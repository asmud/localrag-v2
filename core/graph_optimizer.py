"""
Performance-optimized Neo4j graph operations with minimal properties.
"""

import hashlib
from typing import Dict, List, Optional, Set
from loguru import logger

from .database import db_manager
from .config import settings
from .postgres_storage import postgres_storage


class OptimizedKnowledgeGraphManager:
    """
    Streamlined Knowledge Graph Manager focused on performance and essential relationships.
    
    Design principles:
    - Minimal property storage (avoid content duplication with PostgreSQL)
    - Essential relationships only
    - Optimized for graph traversal and semantic search
    - Batch operations for performance
    """
    
    def __init__(self):
        self.batch_size = settings.kg_batch_size or 50
        self.entity_batch_size = min(self.batch_size, 20)  # Smaller batches for complex entity operations
        self.chunk_batch_size = min(self.batch_size, 30)   # Medium batches for chunk operations
        self.relationship_batch_size = min(self.batch_size, 100)  # Larger batches for simple relationships
        
    async def create_document_node(self, document_data: Dict) -> str:
        """
        Create minimal document node with essential properties only.
        
        Args:
            document_data: Document metadata from ingestion
            
        Returns:
            Document node ID
        """
        try:
            # Generate content hash for deduplication
            content_hash = self._generate_content_hash(document_data.get("content", ""))
            
            query = """
            MERGE (d:Document {file_path: $file_path})
            ON CREATE SET 
                d.id = randomUUID(),
                d.file_name = $file_name,
                d.file_type = $file_type,
                d.content_hash = $content_hash,
                d.chunk_count = $chunk_count,
                d.created_at = datetime(),
                d.language = $language,
                d.region = $region
            ON MATCH SET 
                d.chunk_count = $chunk_count,
                d.updated_at = datetime()
            RETURN d.id as doc_id
            """
            
            metadata = document_data.get("metadata", {})
            
            async with db_manager.get_neo4j_session() as session:
                result = await session.run(query, {
                    "file_path": document_data["file_path"],
                    "file_name": document_data["file_name"],
                    "file_type": metadata.get("file_type", "unknown"),
                    "content_hash": content_hash,
                    "chunk_count": document_data["chunk_count"],
                    "language": "indonesian",  # Default assumption, can be enhanced
                    "region": self._extract_region_from_metadata(metadata)
                })
                
                record = await result.single()
                doc_id = record["doc_id"]
                
                logger.info(f"Created/updated document node: {doc_id}")
                return doc_id
        
        except Exception as e:
            logger.error(f"Failed to create document node: {e}")
            raise
    
    async def create_optimized_chunk_nodes(
        self, 
        document_id: str, 
        chunks_analysis: List[Dict], 
        embeddings: List[List[float]]
    ):
        """
        Create chunk nodes with semantic analysis and essential relationships.
        
        Args:
            document_id: Document node ID
            chunks_analysis: List of chunks with semantic analysis
            embeddings: Corresponding embeddings
        """
        try:
            async with db_manager.get_neo4j_session() as session:
                # Process in batches for performance
                for i in range(0, len(chunks_analysis), self.chunk_batch_size):
                    batch_chunks = chunks_analysis[i:i + self.chunk_batch_size]
                    batch_embeddings = embeddings[i:i + self.chunk_batch_size]
                    
                    await self._create_chunk_batch(session, document_id, batch_chunks, batch_embeddings)
                
                # Create entity and topic relationships
                await self._create_entity_topic_relationships(session, document_id, chunks_analysis)
                
                # Create similarity relationships for top chunks
                await self._create_similarity_relationships(session, document_id, embeddings)
                
                logger.info(f"Created {len(chunks_analysis)} optimized chunk nodes for document {document_id}")
        
        except Exception as e:
            logger.error(f"Failed to create chunk nodes: {e}")
            raise
    
    async def _create_chunk_batch(
        self, 
        session, 
        document_id: str, 
        chunks: List[Dict], 
        embeddings: List[List[float]]
    ):
        """Create a batch of chunk nodes with minimal properties."""
        
        chunk_query = """
        MATCH (d:Document {id: $doc_id})
        UNWIND $chunk_data as chunk
        CREATE (c:Chunk {
            id: randomUUID(),
            doc_id: $doc_id,
            chunk_index: chunk.chunk_index,
            content_hash: chunk.content_hash,
            name: chunk.chunk_title,
            topic: chunk.primary_topic,
            summary: chunk.summary,
            key_phrases: chunk.key_phrases,
            semantic_class: chunk.semantic_class,
            language: chunk.language,
            region: chunk.region,
            formality_level: chunk.formality_level,
            structure_type: chunk.structure_type,
            position: chunk.position,
            entity_count: chunk.entity_count,
            created_at: datetime()
        })
        CREATE (d)-[:HAS_CHUNK {
            index: chunk.chunk_index
        }]->(c)
        RETURN c.id as chunk_id, chunk.chunk_index as idx
        """
        
        # Prepare batch data
        chunk_data = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            # Validate chunk properties before processing
            validated_chunk = self._validate_chunk_properties(chunk)
            analysis = validated_chunk.get('analysis', {})
            
            # Generate meaningful chunk name and primary topic
            chunk_title = validated_chunk.get('title', 'Untitled')
            topics = analysis.get('topics', [])
            primary_topic = topics[0] if topics else analysis.get('semantic_class', 'general')
            
            # Enhance chunk title if it's generic
            if chunk_title in ['Untitled', '']:
                entities = analysis.get('entities', [])
                if entities:
                    # Use first high-confidence entity as part of title
                    high_conf_entities = [e for e in entities if e.get('confidence', 0) > 0.8]
                    if high_conf_entities:
                        chunk_title = f"About {high_conf_entities[0]['text']}"
                    else:
                        chunk_title = f"Content #{i+1}"
                else:
                    chunk_title = f"{primary_topic.title()} Content #{i+1}"
            
            chunk_data.append({
                'chunk_index': validated_chunk.get('chunk_index', i),
                'content_hash': self._generate_content_hash(validated_chunk.get('content', '')),
                'chunk_title': chunk_title,
                'primary_topic': primary_topic,
                'summary': validated_chunk.get('summary', '')[:200],  # Limit summary length
                'key_phrases': validated_chunk.get('key_phrases', [])[:3],  # Limit to top 3 phrases
                'semantic_class': analysis.get('semantic_class', 'general'),
                'language': analysis.get('language', 'indonesian'),
                'region': self._extract_region_from_content(validated_chunk.get('content', '')),
                'formality_level': analysis.get('formality_level', 'neutral'),
                'structure_type': analysis.get('structure_type', 'paragraph'),
                'position': analysis.get('position_context', {}).get('position', 'middle'),
                'topics': analysis.get('topics', [])[:3],  # Limit to top 3 topics
                'entity_count': len(analysis.get('entities', []))
            })
        
        await session.run(chunk_query, {
            'doc_id': document_id,
            'chunk_data': chunk_data
        })
    
    def _validate_chunk_properties(self, chunk: Dict) -> Dict:
        """Validate and ensure essential chunk properties exist with proper defaults."""
        required_properties = {
            'title': 'Untitled',
            'summary': '',
            'key_phrases': [],
            'content': '',
            'chunk_index': 0
        }
        
        # Ensure chunk has all required properties
        for prop, default_value in required_properties.items():
            if prop not in chunk or chunk[prop] is None:
                chunk[prop] = default_value
                logger.debug(f"Missing property '{prop}' in chunk, using default: {default_value}")
        
        # Validate analysis structure
        if 'analysis' not in chunk or not isinstance(chunk['analysis'], dict):
            chunk['analysis'] = {
                'semantic_class': 'general',
                'language': 'unknown',
                'structure_type': 'paragraph',
                'entities': [],
                'topics': chunk.get('key_phrases', [])[:3] if chunk.get('key_phrases') else ['general'],
                'key_phrases': chunk.get('key_phrases', [])
            }
            logger.debug("Missing or invalid analysis in chunk, using default analysis")
        
        # Ensure topics exist in analysis
        analysis = chunk['analysis']
        if 'topics' not in analysis or not analysis['topics']:
            analysis['topics'] = chunk.get('key_phrases', [])[:3] if chunk.get('key_phrases') else ['general']
        
        return chunk
    
    async def _create_entity_topic_relationships(
        self, 
        session, 
        document_id: str, 
        chunks_analysis: List[Dict]
    ):
        """Create entity and topic nodes with relationships."""
        
        # Collect all entities and topics
        all_entities = {}
        all_topics = {}
        
        for chunk in chunks_analysis:
            analysis = chunk.get('analysis', {})
            
            # Process entities
            for entity in analysis.get('entities', []):
                entity_key = f"{entity['text']}_{entity['type']}"
                if entity_key not in all_entities:
                    all_entities[entity_key] = {
                        'name': entity['text'],
                        'type': entity['type'],
                        'confidence': entity['confidence'],
                        'chunks': []
                    }
                all_entities[entity_key]['chunks'].append(chunk.get('chunk_index', 0))
            
            # Process topics
            for topic in analysis.get('topics', []):
                if topic not in all_topics:
                    all_topics[topic] = {
                        'name': topic,
                        'chunks': []
                    }
                all_topics[topic]['chunks'].append(chunk.get('chunk_index', 0))
        
        # Create entity nodes and relationships
        if all_entities:
            await self._create_entities(session, document_id, all_entities)

        # Create topic nodes and relationships
        if all_topics:
            await self._create_topics(session, document_id, all_topics)
    
    async def _create_entities(self, session, document_id: str, entities: Dict):
        """Create Indonesian-specific entity nodes and relationships."""
        
        # Process entities in batches for Indonesian KG optimization
        entity_data = []
        for ent in entities.values():
            # Determine Indonesian-specific entity types
            entity_type = self._classify_indonesian_entity_type(ent['type'], ent['name'])
            normalized_name = self._normalize_entity_name(ent['name'])
            
            entity_data.append({
                'name': ent['name'],
                'type': ent['type'],
                'entity_type': entity_type,
                'normalized_name': normalized_name,
                'confidence': ent['confidence'],
                'chunks': ent['chunks'],
                'language': 'indonesian'
            })
        
        # Batch process entities with optimized batch size
        for i in range(0, len(entity_data), self.entity_batch_size):
            batch = entity_data[i:i + self.entity_batch_size]
            await self._create_entity_batch(session, document_id, batch)
        
        logger.debug(f"Created {len(entity_data)} Indonesian entity relationships")
        
    
    def _map_entity_type_to_node_label(self, entity_type: str, entity_text: str) -> str:
        """Map semantic analyzer entity types to 20 Indonesian NER labels."""
        
        # Direct mapping for NER model outputs (these should pass through unchanged)
        ner_types = {'PER', 'ORG', 'NOR', 'GPE', 'LOC', 'FAC', 'LAW', 'EVT', 'DAT', 'TIM', 
                    'CRD', 'ORD', 'QTY', 'PRC', 'MON', 'PRD', 'REG', 'WOA', 'LAN', 'O'}
        if entity_type in ner_types:
            return entity_type
        
        # Pattern-based entity mapping to Indonesian NER types
        entity_text_lower = entity_text.lower()
        
        # Check for person indicators first
        person_indicators = ['dr.', 'prof.', 'ir.', 'drs.', 'h.', 'hj.', 'kh.', 'bapak', 'ibu', 'pak', 'bu']
        if any(indicator in entity_text_lower for indicator in person_indicators):
            return 'PER'
        
        # Map semantic analyzer types to Indonesian NER schema
        type_mapping = {
            # Organizations - map to ORG or NOR (political)
            'government_org': 'NOR',  # Political/government organizations
            'institution': 'ORG',    # General organizations
            'education': 'ORG',      # Educational institutions
            'healthcare': 'ORG',     # Healthcare organizations
            
            # Locations - map to GPE or LOC
            'regional': 'GPE',       # Geopolitical entities (provinces, cities)
            'location': 'LOC',       # General locations
            
            # Legal entities
            'legal_entity': 'LAW',   # Legal documents
            'legal_section': 'LAW',  # Legal sections
            'legal_reference': 'LAW', # Legal references
            
            # Person-related
            'academic_title': 'PER',
            'religious_title': 'PER',
            'professional_title': 'PER',
            'regional_title': 'PER',
            
            # Events and facilities
            'event': 'EVT',
            'facility': 'FAC',
            
            # Time and date
            'date': 'DAT',
            'time': 'TIM',
            
            # Quantities and money
            'money': 'MON',
            'quantity': 'QTY',
            'percentage': 'PRC',
            'cardinal': 'CRD',
            'ordinal': 'ORD',
            
            # Other types
            'product': 'PRD',
            'religion': 'REG',
            'work_of_art': 'WOA',
            'language': 'LAN'
        }
        
        # Apply contextual mapping based on entity text content
        if entity_type in ['government_org', 'institution']:
            # Check if it's a government/political organization
            if any(keyword in entity_text_lower for keyword in ['kementerian', 'dinas', 'badan', 'lembaga', 'komisi', 'dprd']):
                return 'NOR'  # Political organization
            else:
                return 'ORG'  # General organization
        
        elif entity_type == 'regional':
            # Administrative regions are geopolitical entities
            return 'GPE'
        
        elif entity_type == 'legal_entity':
            # Legal documents and regulations
            return 'LAW'
        
        # Use direct mapping or default to 'O' (Other)
        return type_mapping.get(entity_type, 'O')

    async def _create_entity_batch(self, session, document_id: str, entities: List[Dict]):
        """Create a batch of Indonesian entities with proper node labels."""

        # Group entities by their target node label
        entities_by_label = {}
        for entity in entities:
            node_label = self._map_entity_type_to_node_label(entity['entity_type'], entity['name'])
            if node_label not in entities_by_label:
                entities_by_label[node_label] = []
            entities_by_label[node_label].append(entity)

        # Create entities for each node label type
        for node_label, label_entities in entities_by_label.items():
            await self._create_labeled_entities(session, document_id, node_label, label_entities)

    async def _create_labeled_entities(self, session, document_id: str, node_label: str, entities: List[Dict]):
        """Create entities with specific node labels."""
        
        if not entities:
            logger.debug("No entities to create")
            return
        
        try:
            entity_query = f"""
            MATCH (d:Document {{id: $doc_id}})
            UNWIND $entities as entity

            // Create entity with proper node label: {node_label}
            MERGE (e:{node_label} {{name: entity.name}})
            ON CREATE SET 
                e.id = randomUUID(),
                e.created_at = datetime(),
                e.entity_type = entity.entity_type,
                e.normalized_name = entity.normalized_name,
                e.language = entity.language,
                e.original_type = entity.type
            SET 
                e.confidence = entity.confidence,
                e.frequency = size(entity.chunks),
                e.updated_at = datetime()

            // Create document relationship
            MERGE (d)-[:MENTIONS]->(e)

            // Create relationships with specific chunks
            WITH d, e, entity
            UNWIND entity.chunks as chunk_index
            MATCH (c:Chunk {{doc_id: $doc_id, chunk_index: chunk_index}})
            MERGE (c)-[:MENTIONS {{
                confidence: entity.confidence,
                entity_type: entity.entity_type,
                created_at: datetime()
            }}]->(e)
            
            RETURN count(DISTINCT e) as entities_created, count(DISTINCT c) as chunks_matched
            """
            
            # Execute query with validation
            result = await session.run(entity_query, {
                'doc_id': document_id,
                'entities': entities
            })
            
            # Get result summary
            record = await result.single()
            summary = await result.consume()
            
            # Log detailed results
            entities_created = record['entities_created'] if record else 0
            chunks_matched = record['chunks_matched'] if record else 0
            relationships_created = summary.counters.relationships_created
            
            logger.info(f"Created {entities_created} {node_label} entities, matched {chunks_matched} chunks, created {relationships_created} relationships")
            
            # Validate that relationships were created
            if relationships_created == 0 and entities:
                logger.warning(f"⚠️  No relationships created for {len(entities)} {node_label} entities in document {document_id}")
                # Log the first few entities for debugging
                for i, entity in enumerate(entities[:3]):
                    logger.debug(f"Entity {i+1}: {entity['name']} (chunks: {entity['chunks']})")
            else:
                logger.debug(f"✅ Successfully created {relationships_created} entity relationships for {len(entities)} {node_label} entities")
                
        except Exception as e:
            logger.error(f"Failed to create {node_label} entities: {e}")
            logger.debug(f"Failed entities: {[e['name'] for e in entities[:3]]}")  # Log first 3 for debugging
            raise

    
    async def _create_topics(self, session, document_id: str, topics: Dict):
        """Create topic nodes and relationships."""
        
        if not topics:
            logger.debug("No topics to create")
            return
        
        try:
            topic_query = """
            MATCH (d:Document {id: $doc_id})
            UNWIND $topics as topic
            MERGE (t:Topic {name: topic.name})
            ON CREATE SET t.id = randomUUID(), t.created_at = datetime()
            SET t.frequency = size(topic.chunks)
            WITH d, t, topic
            UNWIND topic.chunks as chunk_idx
            MATCH (c:Chunk {doc_id: $doc_id, chunk_index: chunk_idx})
            MERGE (c)-[:ABOUT]->(t)
            
            RETURN count(DISTINCT t) as topics_created, count(DISTINCT c) as chunks_matched
            """
            
            topic_data = [
                {
                    'name': topic['name'],
                    'chunks': topic['chunks']
                }
                for topic in topics.values()
            ]
            
            result = await session.run(topic_query, {
                'doc_id': document_id,
                'topics': topic_data
            })
            
            # Get result details
            record = await result.single()
            summary = await result.consume()
            
            topics_created = record['topics_created'] if record else 0
            chunks_matched = record['chunks_matched'] if record else 0
            relationships_created = summary.counters.relationships_created
            
            logger.info(f"Created {topics_created} topics, matched {chunks_matched} chunks, created {relationships_created} topic relationships")
            
            # Validate topic relationship creation results
            if relationships_created == 0 and topic_data:
                logger.warning(f"⚠️  No topic relationships created for {len(topic_data)} topics in document {document_id}")
                # Log first few topics for debugging
                for i, topic in enumerate(topic_data[:3]):
                    logger.debug(f"Topic {i+1}: {topic['name']} (chunks: {topic['chunks']})")
            else:
                logger.debug(f"✅ Successfully created {relationships_created} topic relationships for {len(topic_data)} topics")
                
        except Exception as e:
            logger.error(f"Failed to create topics: {e}")
            logger.debug(f"Failed topics: {[t['name'] for t in topic_data[:3]]}")  # Log first 3 for debugging
            raise
    
    async def _create_similarity_relationships(
        self, 
        session, 
        document_id: str, 
        embeddings: List[List[float]]
    ):
        """Create similarity relationships between the most similar chunks."""
        try:
            if len(embeddings) < 2:
                return
            
            # Calculate similarities and create relationships for top similarities only
            similarity_threshold = settings.graph_similarity_threshold  # Use configurable threshold
            max_relationships_per_chunk = 3  # Limit relationships for performance
            
            chunk_similarities = []
            
            for i, emb1 in enumerate(embeddings):
                similarities = []
                for j, emb2 in enumerate(embeddings):
                    if i != j:
                        similarity = self._cosine_similarity(emb1, emb2)
                        if similarity >= similarity_threshold:
                            similarities.append((j, similarity))
                
                # Sort by similarity and keep top relationships
                similarities.sort(key=lambda x: x[1], reverse=True)
                similarities = similarities[:max_relationships_per_chunk]
                
                for j, sim_score in similarities:
                    chunk_similarities.append({
                        'chunk1_idx': i,
                        'chunk2_idx': j,
                        'similarity': round(sim_score, 3)
                    })
            
            if chunk_similarities:
                similarity_query = """
                MATCH (d:Document {id: $doc_id})
                UNWIND $similarities as sim
                MATCH (c1:Chunk {doc_id: $doc_id, chunk_index: sim.chunk1_idx})
                MATCH (c2:Chunk {doc_id: $doc_id, chunk_index: sim.chunk2_idx})
                MERGE (c1)-[:SIMILAR_TO {similarity: sim.similarity}]->(c2)
                """
                
                await session.run(similarity_query, {
                    'doc_id': document_id,
                    'similarities': chunk_similarities
                })
                
                logger.debug(f"Created {len(chunk_similarities)} similarity relationships")
        
        except Exception as e:
            logger.warning(f"Failed to create similarity relationships: {e}")
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        import math
        
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = math.sqrt(sum(a * a for a in vec1))
        magnitude2 = math.sqrt(sum(b * b for b in vec2))
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0
        
        return dot_product / (magnitude1 * magnitude2)
    
    def _generate_content_hash(self, content: str) -> str:
        """Generate SHA256 hash of content for deduplication."""
        return hashlib.sha256(content.encode('utf-8')).hexdigest()[:16]  # Shortened hash
    
    def _classify_indonesian_entity_type(self, entity_type: str, entity_name: str) -> str:
        """Classify entity into Indonesian-specific types."""
        entity_name_lower = entity_name.lower()
        
        # Indonesian government organization patterns
        if entity_type in ['government_org', 'institution']:
            if any(keyword in entity_name_lower for keyword in ['kementerian', 'kemen', 'kemenko']):
                return 'ministry'
            elif any(keyword in entity_name_lower for keyword in ['dinas', 'badan', 'lembaga']):
                return 'agency'
            elif any(keyword in entity_name_lower for keyword in ['kantor', 'balai', 'pusat']):
                return 'office'
            return 'organization'
        
        # Educational institutions
        elif entity_type == 'education':
            if any(keyword in entity_name_lower for keyword in ['universitas', 'institut']):
                return 'university'
            elif any(keyword in entity_name_lower for keyword in ['sekolah', 'akademi']):
                return 'school'
            return 'education'
        
        # Regional/administrative divisions
        elif entity_type == 'regional':
            if 'provinsi' in entity_name_lower:
                return 'province'
            elif any(keyword in entity_name_lower for keyword in ['kota', 'kabupaten']):
                return 'city'
            elif any(keyword in entity_name_lower for keyword in ['kecamatan', 'kelurahan']):
                return 'district'
            return 'location'
        
        # Legal entities
        elif entity_type == 'legal_entity':
            if any(keyword in entity_name_lower for keyword in ['undang-undang', 'uu']):
                return 'law'
            elif any(keyword in entity_name_lower for keyword in ['peraturan', 'permen', 'perpres']):
                return 'regulation'
            return 'legal_entity'
        
        # Person titles
        elif entity_type in ['academic_title', 'religious_title', 'professional_title', 'regional_title']:
            return 'person'
        
        return entity_type
    
    def _normalize_entity_name(self, entity_name: str) -> str:
        """Normalize entity name for Indonesian context."""
        normalized = entity_name.lower()
        
        # Handle Indonesian spelling variations
        variations = {
            'djakarta': 'jakarta',
            'jogjakarta': 'yogyakarta', 
            'djawa': 'jawa',
            'soekarno': 'sukarno'
        }
        
        for old, new in variations.items():
            normalized = normalized.replace(old, new)
        
        # Remove common prefixes for normalization
        prefixes = ['dr.', 'prof.', 'ir.', 'drs.', 'h.', 'hj.', 'kh.']
        for prefix in prefixes:
            if normalized.startswith(prefix):
                normalized = normalized[len(prefix):].strip()
        
        return normalized.strip()
    
    async def find_similar_chunks(
        self, 
        query_embedding: List[float], 
        limit: int = 10,
        semantic_filter: Optional[str] = None
    ) -> List[Dict]:
        """
        Find similar chunks using PostgreSQL for vector similarity and Neo4j for metadata.
        
        Args:
            query_embedding: Query vector
            limit: Maximum results
            semantic_filter: Optional semantic class filter
            
        Returns:
            List of similar chunks with metadata
        """
        try:
            # Use PostgreSQL for vector similarity search
            postgres_results = await postgres_storage.find_similar_chunks(
                query_embedding, 
                limit=limit * 2  # Get more results for filtering
            )
            
            if not postgres_results:
                return []
            
            # Get chunk IDs for Neo4j metadata lookup
            chunk_ids = [str(result.get('id')) for result in postgres_results]
            
            # Get semantic metadata from Neo4j
            metadata_query = """
            MATCH (c:Chunk)
            WHERE c.id IN $chunk_ids
            OPTIONAL MATCH (c)-[:MENTIONS]->(e)
            WHERE e.name IS NOT NULL AND any(label IN labels(e) WHERE NOT label IN ['Chunk', 'Document', 'Topic']) 
            OPTIONAL MATCH (c)-[:ABOUT]->(t:Topic)
            RETURN c.id as chunk_id, 
                   c.doc_id as doc_id,
                   c.semantic_class as semantic_class,
                   c.structure_type as structure_type,
                   collect(DISTINCT e.name) as entities,
                   collect(DISTINCT t.name) as topics
            """
            
            async with db_manager.get_neo4j_session() as session:
                result = await session.run(metadata_query, {"chunk_ids": chunk_ids})
                
                # Create metadata lookup
                metadata_lookup = {}
                async for record in result:
                    metadata_lookup[record['chunk_id']] = {
                        'doc_id': record['doc_id'],
                        'semantic_class': record['semantic_class'],
                        'structure_type': record['structure_type'],
                        'entities': record['entities'],
                        'topics': record['topics']
                    }
                
                # Combine PostgreSQL results with Neo4j metadata
                combined_results = []
                for pg_result in postgres_results:
                    chunk_id = str(pg_result.get('id'))
                    metadata = metadata_lookup.get(chunk_id, {})
                    
                    # Apply semantic filter if specified
                    if semantic_filter and metadata.get('semantic_class') != semantic_filter:
                        continue
                    
                    combined_results.append({
                        'chunk_id': chunk_id,
                        'content': pg_result.get('content', ''),
                        'similarity': pg_result.get('similarity_score', 0.0),
                        'file_name': pg_result.get('file_name', ''),
                        'doc_id': metadata.get('doc_id'),
                        'semantic_class': metadata.get('semantic_class', 'general'),
                        'structure_type': metadata.get('structure_type', 'paragraph'),
                        'entities': metadata.get('entities', []),
                        'topics': metadata.get('topics', [])
                    })
                
                # Sort by similarity and limit results
                combined_results.sort(key=lambda x: x['similarity'], reverse=True)
                final_results = combined_results[:limit]
                
                logger.debug(f"Found {len(final_results)} similar chunks")
                return final_results
        
        except Exception as e:
            logger.error(f"Failed to find similar chunks: {e}")
            return []

    async def find_similar_chunks_optimized(
        self, 
        query_embedding: List[float], 
        limit: int = 10,
        semantic_filter: Optional[str] = None
    ) -> List[Dict]:
        """
        Optimized similarity search using graph relationships.
        
        Args:
            query_embedding: Query vector
            limit: Maximum results
            semantic_filter: Optional semantic class filter
            
        Returns:
            List of similar chunks with metadata
        """
        try:
            # Note: This is a placeholder - actual vector similarity search
            # should be done in PostgreSQL for performance
            # Neo4j is used for relationship traversal and semantic filtering
            
            base_query = """
            MATCH (c:Chunk)
            WHERE c.semantic_class IS NOT NULL
            """
            
            if semantic_filter:
                base_query += f" AND c.semantic_class = '{semantic_filter}'"
            
            base_query += """
            OPTIONAL MATCH (c)-[:MENTIONS]->(e)
            WHERE e.name IS NOT NULL AND any(label IN labels(e) WHERE NOT label IN ['Chunk', 'Document', 'Topic']) 
            OPTIONAL MATCH (c)-[:ABOUT]->(t:Topic)
            RETURN c.id as chunk_id, 
                   c.doc_id as doc_id,
                   c.semantic_class as semantic_class,
                   c.structure_type as structure_type,
                   collect(DISTINCT e.name) as entities,
                   collect(DISTINCT t.name) as topics
            LIMIT $limit
            """
            
            async with db_manager.get_neo4j_session() as session:
                result = await session.run(base_query, {"limit": limit})
                
                chunks = []
                async for record in result:
                    chunks.append({
                        'chunk_id': record['chunk_id'],
                        'doc_id': record['doc_id'],
                        'semantic_class': record['semantic_class'],
                        'structure_type': record['structure_type'],
                        'entities': record['entities'],
                        'topics': record['topics'],
                        'similarity': 0.85  # Placeholder - should come from PostgreSQL
                    })
                
                logger.debug(f"Found {len(chunks)} similar chunks")
                return chunks
        
        except Exception as e:
            logger.error(f"Failed to find similar chunks: {e}")
            return []
    
    def _extract_region_from_metadata(self, metadata: Dict) -> str:
        """Extract Indonesian region from document metadata."""
        # Default region extraction logic
        file_path = metadata.get('file_path', '').lower()
        
        # Indonesian regional keywords
        if any(region in file_path for region in ['jakarta', 'dki']):
            return 'jakarta'
        elif any(region in file_path for region in ['jawa', 'java']):
            return 'java'
        elif any(region in file_path for region in ['sumatra', 'sumatera']):
            return 'sumatra'
        elif any(region in file_path for region in ['kalimantan', 'borneo']):
            return 'kalimantan'
        elif any(region in file_path for region in ['sulawesi', 'celebes']):
            return 'sulawesi'
        elif any(region in file_path for region in ['papua', 'irian']):
            return 'papua'
        
        return 'unknown'
    
    def _extract_region_from_content(self, content: str) -> str:
        """Extract Indonesian region mentions from content."""
        content_lower = content.lower()
        
        # Count regional mentions
        region_scores = {
            'jakarta': sum(1 for keyword in ['jakarta', 'dki', 'betawi'] if keyword in content_lower),
            'java': sum(1 for keyword in ['bandung', 'surabaya', 'semarang', 'yogyakarta'] if keyword in content_lower),
            'sumatra': sum(1 for keyword in ['medan', 'palembang', 'pekanbaru', 'padang'] if keyword in content_lower),
            'kalimantan': sum(1 for keyword in ['pontianak', 'banjarmasin', 'balikpapan'] if keyword in content_lower),
            'sulawesi': sum(1 for keyword in ['makassar', 'manado', 'palu'] if keyword in content_lower),
            'papua': sum(1 for keyword in ['jayapura', 'sorong', 'merauke'] if keyword in content_lower)
        }
        
        # Return region with highest score
        max_region = max(region_scores.items(), key=lambda x: x[1])
        return max_region[0] if max_region[1] > 0 else 'unknown'
    
    async def get_document_statistics(self, document_id: str) -> Dict:
        """Get document statistics from the knowledge graph."""
        query = """
        MATCH (d:Document {id: $doc_id})
        OPTIONAL MATCH (d)-[:HAS_CHUNK]->(c:Chunk)
        OPTIONAL MATCH (c)-[:MENTIONS]->(e)
        WHERE e.name IS NOT NULL AND any(label IN labels(e) WHERE NOT label IN ['Chunk', 'Document', 'Topic']) 
        OPTIONAL MATCH (c)-[:ABOUT]->(t:Topic)
        RETURN d.file_name as file_name,
               d.chunk_count as total_chunks,
               count(DISTINCT c) as processed_chunks,
               count(DISTINCT e) as unique_entities,
               count(DISTINCT t) as unique_topics,
               collect(DISTINCT c.semantic_class) as semantic_classes
        """
        
        async with db_manager.get_neo4j_session() as session:
            result = await session.run(query, {"doc_id": document_id})
            record = await result.single()
            
            if record:
                return {
                    'file_name': record['file_name'],
                    'total_chunks': record['total_chunks'],
                    'processed_chunks': record['processed_chunks'],
                    'unique_entities': record['unique_entities'],
                    'unique_topics': record['unique_topics'],
                    'semantic_classes': record['semantic_classes']
                }
            
            return {}
    
    async def cleanup_document(self, document_id: str):
        """Remove document and all related nodes/relationships."""
        query = """
        MATCH (d:Document {id: $doc_id})
        OPTIONAL MATCH (d)-[:HAS_CHUNK]->(c:Chunk)
        OPTIONAL MATCH (c)-[r]-()
        DELETE r, c, d
        """
        
        async with db_manager.get_neo4j_session() as session:
            await session.run(query, {"doc_id": document_id})
            logger.info(f"Cleaned up document {document_id} from knowledge graph")
    
    async def get_indonesian_kg_metrics(self) -> Dict:
        """Get comprehensive metrics for Indonesian Knowledge Graph."""
        queries = {
            'entity_distribution': """
                MATCH (e)
                WHERE e.name IS NOT NULL 
                    AND e.language = 'indonesian' 
                    AND any(label IN labels(e) WHERE NOT label IN ['Chunk', 'Document', 'Topic']) 
                RETURN labels(e)[0] as entity_type, count(e) as count
                ORDER BY count DESC
            """,
            'high_confidence_entities': """
                MATCH (e)
                WHERE e.name IS NOT NULL 
                    AND e.language = 'indonesian' 
                    AND any(label IN labels(e) WHERE NOT label IN ['Chunk', 'Document', 'Topic']) 
                    AND e.confidence > 0.8
                RETURN count(e) as count
            """,
            'entities_with_relationships': """
                MATCH (e)
                WHERE e.name IS NOT NULL 
                    AND e.language = 'indonesian' 
                    AND any(label IN labels(e) WHERE NOT label IN ['Chunk', 'Document', 'Topic']) 
                    AND exists((e)-[]-())
                RETURN count(e) as count
            """,
            'orphaned_entities': """
                MATCH (e)
                WHERE e.name IS NOT NULL 
                    AND e.language = 'indonesian' 
                    AND any(label IN labels(e) WHERE NOT label IN ['Chunk', 'Document', 'Topic']) 
                    AND exists((e)-[]-())
                RETURN count(e) as count
            """,
            'relationship_types': """
                MATCH (e1)-[r]->(e2)
                WHERE e1.name IS NOT NULL 
                    AND e1.language = 'indonesian' 
                    AND any(label IN labels(e1) WHERE NOT label IN ['Chunk', 'Document', 'Topic']) 
                    AND e2.name IS NOT NULL 
                    AND e2.language = 'indonesian' 
                    AND any(label IN labels(e2) WHERE NOT label IN ['Chunk', 'Document', 'Topic']) 
                RETURN type(r) as relationship_type, count(r) as count
                ORDER BY count DESC
            """,
            'regional_distribution': """
                MATCH (c:Chunk)
                WHERE c.region IS NOT NULL
                RETURN c.region as region, count(c) as count
                ORDER BY count DESC
            """
        }
        
        metrics = {}
        async with db_manager.get_neo4j_session() as session:
            for metric_name, query in queries.items():
                try:
                    result = await session.run(query)
                    if metric_name in ['entity_distribution', 'relationship_types', 'regional_distribution']:
                        metrics[metric_name] = {record[list(record.keys())[0]]: record[list(record.keys())[1]] async for record in result}
                    else:
                        record = await result.single()
                        metrics[metric_name] = record['count'] if record else 0
                except Exception as e:
                    logger.warning(f"Failed to get metric {metric_name}: {e}")
                    metrics[metric_name] = 0
        
        return metrics
    
    async def validate_indonesian_kg_quality(self) -> Dict:
        """Comprehensive quality validation for Indonesian Knowledge Graph."""
        quality_checks = {
            'duplicate_entities': """
                MATCH (e1), (e2)
                WHERE e1.name IS NOT NULL 
                    AND e1.language = 'indonesian' 
                    AND any(label IN labels(e1) WHERE NOT label IN ['Chunk', 'Document', 'Topic']) 
                    AND e2.name IS NOT NULL 
                    AND e2.language = 'indonesian' 
                    AND any(label IN labels(e2) WHERE NOT label IN ['Chunk', 'Document', 'Topic']) 
                    AND e1.normalized_name = e2.normalized_name
                    AND e1.id < e2.id
                RETURN count(*) as duplicates
            """,
            'missing_types': """
                MATCH (e)
                WHERE e.name IS NOT NULL 
                    AND e.language = 'indonesian' 
                    AND any(label IN labels(e) WHERE NOT label IN ['Chunk', 'Document', 'Topic']) 
                    AND (e.entity_type IS NULL OR e.entity_type = '')
                RETURN count(e) as missing_types
            """,
            'low_confidence_entities': """
                MATCH (e)
                WHERE e.name IS NOT NULL 
                    AND e.language = 'indonesian' 
                    AND any(label IN labels(e) WHERE NOT label IN ['Chunk', 'Document', 'Topic']) 
                    AND e.confidence < 0.5
                RETURN count(e) as low_confidence
            """,
            'invalid_names': """
                MATCH (e)
                WHERE e.name IS NOT NULL 
                    AND e.language = 'indonesian' 
                    AND any(label IN labels(e) WHERE NOT label IN ['Chunk', 'Document', 'Topic']) 
                    AND (e.name =~ '.*[0-9]{3,}.*' OR size(e.name) < 2)
                RETURN count(e) as invalid_names
            """,
            'entities_without_normalization': """
                MATCH (e)
                WHERE e.name IS NOT NULL 
                    AND e.language = 'indonesian' 
                    AND any(label IN labels(e) WHERE NOT label IN ['Chunk', 'Document', 'Topic']) 
                    AND (e.normalized_name IS NULL OR e.normalized_name = '')
                RETURN count(e) as count
            """
        }
        
        results = {}
        async with db_manager.get_neo4j_session() as session:
            for check_name, query in quality_checks.items():
                try:
                    result = await session.run(query)
                    record = await result.single()
                    results[check_name] = record.value() if record else 0
                except Exception as e:
                    logger.warning(f"Quality check {check_name} failed: {e}")
                    results[check_name] = 0
        
        return results
    
    async def get_indonesian_entities_stream(self, entity_type: Optional[str] = None, limit: int = 10000):
        """Stream Indonesian entities to avoid memory issues."""
        query = """
        MATCH (e)
        WHERE e.name IS NOT NULL 
            AND e.language = 'indonesian' 
            AND any(label IN labels(e) WHERE NOT label IN ['Chunk', 'Document', 'Topic']) 
            AND ($entity_type IS NULL OR e.entity_type = $entity_type)
        RETURN e.name, e.entity_type, e.confidence, e.normalized_name
        ORDER BY e.confidence DESC
        LIMIT $limit
        """
        
        async with db_manager.get_neo4j_session() as session:
            result = await session.run(query, entity_type=entity_type, limit=limit)
            async for record in result:
                yield {
                    'name': record['e.name'],
                    'entity_type': record['e.entity_type'],
                    'confidence': record['e.confidence'],
                    'normalized_name': record['e.normalized_name']
                }
    
    async def create_indonesian_relationships_batch(self, relationships: List[Dict], batch_size: int = 1000):
        """Create Indonesian entity relationships in batches."""
        for i in range(0, len(relationships), batch_size):
            batch = relationships[i:i+batch_size]
            await self._create_relationship_batch(batch)
    
    async def _create_relationship_batch(self, relationships: List[Dict]):
        """Create a batch of relationships between Indonesian entities."""
        query = """
        UNWIND $relationships AS rel
        MATCH (e1)
        WHERE (e1:PER OR e1:ORG OR e1:NOR OR e1:GPE OR e1:LOC OR e1:FAC OR e1:LAW OR e1:EVT OR e1:DAT OR e1:TIM OR 
               e1:CRD OR e1:ORD OR e1:QTY OR e1:PRC OR e1:MON OR e1:PRD OR e1:REG OR e1:WOA OR e1:LAN OR e1:O)
              AND e1.name = rel.source AND e1.language = 'indonesian'
        MATCH (e2)
        WHERE (e2:PER OR e2:ORG OR e2:NOR OR e2:GPE OR e2:LOC OR e2:FAC OR e2:LAW OR e2:EVT OR e2:DAT OR e2:TIM OR 
               e2:CRD OR e2:ORD OR e2:QTY OR e2:PRC OR e2:MON OR e2:PRD OR e2:REG OR e2:WOA OR e2:LAN OR e2:O)
              AND e2.name = rel.target AND e2.language = 'indonesian'
        MERGE (e1)-[r:RELATED_TO {type: rel.type}]->(e2)
        SET r.confidence = rel.confidence,
            r.created_at = datetime()
        """
        
        async with db_manager.get_neo4j_session() as session:
            await session.run(query, relationships=relationships)


# Global instance
optimized_kg_manager = OptimizedKnowledgeGraphManager()