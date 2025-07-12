"""
Indonesian Knowledge Graph Management with NER optimization.
Specialized manager for Indonesian language processing with performance monitoring.
"""

import re
import time
from typing import Dict, List, Optional, AsyncGenerator
from functools import wraps
from loguru import logger

from .graph_optimizer import optimized_kg_manager
from .semantic_analyzer import semantic_analyzer


def monitor_performance(func):
    """Decorator to monitor function performance."""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        result = await func(*args, **kwargs)
        end_time = time.time()
        
        logger.info(f"{func.__name__} took {end_time - start_time:.2f} seconds")
        return result
    return wrapper


class IndonesianKGManager:
    """
    Specialized manager for Indonesian Knowledge Graph operations.
    Combines semantic analysis with graph optimization for Indonesian content.
    """
    
    def __init__(self):
        self.kg_manager = optimized_kg_manager
        self.semantic_analyzer = semantic_analyzer
        
        # Indonesian regional partitioning for distributed processing
        self.regional_partitions = {
            'java': ['jakarta', 'bandung', 'surabaya', 'semarang', 'yogyakarta'],
            'sumatra': ['medan', 'palembang', 'pekanbaru', 'padang'],
            'kalimantan': ['pontianak', 'banjarmasin', 'balikpapan', 'samarinda'],
            'sulawesi': ['makassar', 'manado', 'palu', 'kendari'],
            'papua': ['jayapura', 'sorong', 'merauke'],
            'others': []
        }
    
    @monitor_performance
    async def process_indonesian_document(self, document_data: Dict, chunks_data: List[Dict]) -> Dict:
        """
        Process Indonesian document with comprehensive semantic analysis.
        
        Args:
            document_data: Document metadata
            chunks_data: List of document chunks with content
            
        Returns:
            Processing results with Indonesian KG statistics
        """
        try:
            # Create document node with Indonesian context
            doc_id = await self.kg_manager.create_document_node(document_data)
            
            # Perform semantic analysis on all chunks while preserving original chunk properties
            analyzed_chunks = []
            for i, chunk_data in enumerate(chunks_data):
                # Extract content for analysis
                content = chunk_data['content']
                
                # Perform semantic analysis
                analysis = await self.semantic_analyzer.analyze_chunk(
                    content, 
                    i, 
                    len(chunks_data)
                )
                
                # Create enhanced chunk preserving all original properties
                enhanced_chunk = chunk_data.copy()  # Preserve all original properties including title, summary, etc.
                enhanced_chunk['analysis'] = analysis
                analyzed_chunks.append(enhanced_chunk)
            
            # Apply Indonesian entity validation and enhancement
            validated_chunks = await self._validate_and_enhance_entities(analyzed_chunks)
            
            # Create chunk nodes with Indonesian entity relationships
            embeddings = [chunk.get('embedding', []) for chunk in chunks_data]
            await self.kg_manager.create_optimized_chunk_nodes(
                doc_id, 
                validated_chunks, 
                embeddings
            )
            
            # Get processing statistics
            stats = await self.get_document_processing_stats(doc_id)
            
            logger.info(f"Successfully processed Indonesian document: {doc_id}")
            return {
                'document_id': doc_id,
                'chunks_processed': len(analyzed_chunks),
                'entities_extracted': stats.get('unique_entities', 0),
                'topics_identified': stats.get('unique_topics', 0),
                'processing_stats': stats
            }
            
        except Exception as e:
            logger.error(f"Failed to process Indonesian document: {e}")
            raise
    
    async def get_document_processing_stats(self, document_id: str) -> Dict:
        """Get comprehensive processing statistics for Indonesian document."""
        return await self.kg_manager.get_document_statistics(document_id)
    
    @monitor_performance
    async def validate_indonesian_entities(self, entities: List[Dict]) -> List[Dict]:
        """
        Validate Indonesian entities against known patterns.
        
        Args:
            entities: List of extracted entities
            
        Returns:
            Validated entities with quality scores
        """
        validation_rules = {
            'person': {
                'min_length': 2,
                'max_length': 50,
                'pattern': r'^[A-Za-z\s\.\-\']+$',
                'required_tokens': ['name']
            },
            'organization': {
                'min_length': 3,
                'max_length': 100,
                'pattern': r'^[A-Za-z0-9\s\.\-\(\)]+$',
                'required_tokens': ['name', 'type']
            },
            'location': {
                'min_length': 2,
                'max_length': 50,
                'pattern': r'^[A-Za-z\s\.\-]+$',
                'required_tokens': ['name', 'type']
            }
        }
        
        validated_entities = []
        for entity in entities:
            entity_type = entity.get('type', 'unknown')
            if entity_type in validation_rules:
                rules = validation_rules[entity_type]
                if self._validate_entity_against_rules(entity, rules):
                    entity['quality_score'] = self._calculate_quality_score(entity)
                    validated_entities.append(entity)
        
        return validated_entities
    
    def _validate_entity_against_rules(self, entity: Dict, rules: Dict) -> bool:
        """Validate entity against specific rules."""
        import re
        
        text = entity.get('text', '')
        
        # Length validation
        if len(text) < rules['min_length'] or len(text) > rules['max_length']:
            return False
        
        # Pattern validation
        if not re.match(rules['pattern'], text):
            return False
        
        return True
    
    def _calculate_quality_score(self, entity: Dict) -> float:
        """Calculate quality score for Indonesian entity."""
        base_score = entity.get('confidence', 0.5)
        
        # Boost for Indonesian-specific patterns
        text = entity.get('text', '').lower()
        entity_type = entity.get('type', '')
        
        # Government organization boost
        if entity_type == 'government_org' and any(
            keyword in text for keyword in ['kementerian', 'dinas', 'badan']
        ):
            base_score += 0.2
        
        # Academic title boost
        if entity_type == 'academic_title' and any(
            title in text for title in ['dr.', 'prof.', 'ir.']
        ):
            base_score += 0.15
        
        # Regional entity boost
        if entity_type == 'regional' and any(
            keyword in text for keyword in ['provinsi', 'kota', 'kabupaten']
        ):
            base_score += 0.1
        
        return min(base_score, 1.0)
    
    async def partition_entities_by_region(self, entities: List[Dict]) -> Dict[str, List[Dict]]:
        """Partition entities by Indonesian regions for distributed processing."""
        partitioned = {region: [] for region in self.regional_partitions.keys()}
        
        for entity in entities:
            assigned = False
            text = entity.get('text', '').lower()
            
            for region, cities in self.regional_partitions.items():
                if any(city in text for city in cities):
                    partitioned[region].append(entity)
                    assigned = True
                    break
            
            if not assigned:
                partitioned['others'].append(entity)
        
        return partitioned
    
    @monitor_performance
    async def get_comprehensive_kg_metrics(self) -> Dict:
        """Get comprehensive Indonesian Knowledge Graph metrics."""
        try:
            # Get basic metrics
            metrics = await self.kg_manager.get_indonesian_kg_metrics()
            
            # Get quality validation results
            quality_metrics = await self.kg_manager.validate_indonesian_kg_quality()
            
            # Calculate derived metrics
            total_entities = sum(metrics.get('entity_distribution', {}).values())
            high_confidence = metrics.get('high_confidence_entities', 0)
            
            derived_metrics = {
                'total_entities': total_entities,
                'high_confidence_ratio': high_confidence / max(total_entities, 1),
                'quality_score': self._calculate_overall_quality_score(quality_metrics, total_entities)
            }
            
            return {
                'basic_metrics': metrics,
                'quality_metrics': quality_metrics,
                'derived_metrics': derived_metrics,
                'timestamp': time.time()
            }
            
        except Exception as e:
            logger.error(f"Failed to get KG metrics: {e}")
            return {}
    
    def _calculate_overall_quality_score(self, quality_metrics: Dict, total_entities: int) -> float:
        """Calculate overall quality score for the Knowledge Graph."""
        if total_entities == 0:
            return 0.0
        
        # Penalty factors
        duplicate_penalty = quality_metrics.get('duplicate_entities', 0) / total_entities
        missing_type_penalty = quality_metrics.get('missing_types', 0) / total_entities
        low_confidence_penalty = quality_metrics.get('low_confidence_entities', 0) / total_entities
        invalid_name_penalty = quality_metrics.get('invalid_names', 0) / total_entities
        
        # Start with perfect score and apply penalties
        quality_score = 1.0
        quality_score -= (duplicate_penalty * 0.3)
        quality_score -= (missing_type_penalty * 0.2)
        quality_score -= (low_confidence_penalty * 0.3)
        quality_score -= (invalid_name_penalty * 0.2)
        
        return max(quality_score, 0.0)
    
    async def stream_indonesian_entities(
        self, 
        entity_type: Optional[str] = None, 
        limit: int = 10000
    ) -> AsyncGenerator[Dict, None]:
        """Stream Indonesian entities for memory-efficient processing."""
        async for entity in self.kg_manager.get_indonesian_entities_stream(entity_type, limit):
            yield entity

    async def _validate_and_enhance_entities(self, chunks: List[Dict]) -> List[Dict]:
        """Apply Indonesian-specific entity validation and enhancement."""
        
        enhanced_chunks = []
        for chunk in chunks:
            enhanced_chunk = chunk.copy()
            analysis = chunk.get('analysis', {})
            
            if 'entities' in analysis:
                # Apply Indonesian entity validation
                validated_entities = self._validate_indonesian_entities(analysis['entities'])
                
                # Enhance entities with Indonesian context
                enhanced_entities = self._enhance_entity_context(validated_entities, chunk['content'])
                
                # Update analysis with validated entities
                enhanced_analysis = analysis.copy()
                enhanced_analysis['entities'] = enhanced_entities
                enhanced_chunk['analysis'] = enhanced_analysis
            
            enhanced_chunks.append(enhanced_chunk)
        
        return enhanced_chunks

    def _validate_indonesian_entities(self, entities: List[Dict]) -> List[Dict]:
        """Validate entities using Indonesian language patterns and context."""
        
        validated_entities = []
        for entity in entities:
            entity_text = entity.get('text', '').strip()
            entity_type = entity.get('type', '')
            
            # Skip very short or invalid entities
            if len(entity_text) < 2:
                continue
            
            # Indonesian-specific validation patterns
            is_valid = True
            
            # Validate government organizations
            if entity_type == 'government_org':
                is_valid = self._validate_government_org(entity_text)
            
            # Validate Indonesian locations
            elif entity_type == 'regional':
                is_valid = self._validate_indonesian_location(entity_text)
            
            # Validate legal entities
            elif entity_type == 'legal_entity':
                is_valid = self._validate_legal_entity(entity_text)
            
            # Validate institutions
            elif entity_type == 'institution':
                is_valid = self._validate_institution(entity_text)
            
            # Add confidence adjustment based on validation
            if is_valid:
                # Boost confidence for validated Indonesian entities
                entity['confidence'] = min(1.0, entity.get('confidence', 0.5) + 0.2)
                validated_entities.append(entity)
            
        return validated_entities

    def _validate_government_org(self, text: str) -> bool:
        """Validate Indonesian government organization names."""
        gov_patterns = [
            r'kementerian', r'kemen\b', r'kemenko', r'badan', r'dinas', 
            r'kantor', r'direktorat', r'pusat', r'balai', r'lembaga'
        ]
        text_lower = text.lower()
        return any(re.search(pattern, text_lower) for pattern in gov_patterns)

    def _validate_indonesian_location(self, text: str) -> bool:
        """Validate Indonesian location names."""
        location_patterns = [
            r'provinsi', r'kota', r'kabupaten', r'kecamatan', 
            r'kelurahan', r'desa', r'nagari', r'gampong'
        ]
        text_lower = text.lower()
        
        # Check for administrative level indicators
        if any(re.search(pattern, text_lower) for pattern in location_patterns):
            return True
        
        # Check against known Indonesian places (basic validation)
        known_places = [
            'jakarta', 'bandung', 'surabaya', 'medan', 'bekasi', 'tangerang',
            'depok', 'semarang', 'palembang', 'makassar', 'bogor', 'batam'
        ]
        return any(place in text_lower for place in known_places)

    def _validate_legal_entity(self, text: str) -> bool:
        """Validate Indonesian legal entity names."""
        legal_patterns = [
            r'undang-undang', r'\buu\b', r'peraturan', r'keputusan', 
            r'instruksi', r'surat', r'perpres', r'permen', r'perda'
        ]
        text_lower = text.lower()
        return any(re.search(pattern, text_lower) for pattern in legal_patterns)

    def _validate_institution(self, text: str) -> bool:
        """Validate institutional entity names."""
        institution_patterns = [
            r'universitas', r'institut', r'sekolah', r'akademi', r'politeknik',
            r'rumah sakit', r'\brs\b', r'puskesmas', r'klinik'
        ]
        text_lower = text.lower()
        return any(re.search(pattern, text_lower) for pattern in institution_patterns)

    def _enhance_entity_context(self, entities: List[Dict], content: str) -> List[Dict]:
        """Enhance entities with Indonesian contextual information."""
        
        enhanced_entities = []
        for entity in entities:
            enhanced_entity = entity.copy()
            
            # Add regional partition information
            if entity.get('type') == 'regional':
                partition = self._determine_regional_partition(entity['text'])
                enhanced_entity['regional_partition'] = partition
            
            # Add administrative level for locations
            if entity.get('type') in ['regional', 'location']:
                admin_level = self._determine_admin_level(entity['text'])
                enhanced_entity['admin_level'] = admin_level
            
            # Add legal hierarchy level
            if entity.get('type') == 'legal_entity':
                legal_level = self._determine_legal_hierarchy(entity['text'])
                enhanced_entity['legal_hierarchy'] = legal_level
            
            # Add Indonesian normalization
            enhanced_entity['normalized_indonesian'] = self._normalize_indonesian_text(entity['text'])
            
            enhanced_entities.append(enhanced_entity)
        
        return enhanced_entities

    def _determine_regional_partition(self, location_text: str) -> str:
        """Determine which Indonesian regional partition a location belongs to."""
        text_lower = location_text.lower()
        
        for region, cities in self.regional_partitions.items():
            if any(city in text_lower for city in cities):
                return region
        
        # Check for regional indicators
        if any(indicator in text_lower for indicator in ['jawa', 'jakarta', 'bandung', 'surabaya']):
            return 'java'
        elif any(indicator in text_lower for indicator in ['sumatra', 'medan', 'palembang']):
            return 'sumatra'
        elif any(indicator in text_lower for indicator in ['kalimantan', 'borneo']):
            return 'kalimantan'
        elif any(indicator in text_lower for indicator in ['sulawesi', 'makassar']):
            return 'sulawesi'
        elif any(indicator in text_lower for indicator in ['papua', 'jayapura']):
            return 'papua'
        
        return 'others'

    def _determine_admin_level(self, location_text: str) -> str:
        """Determine administrative level of Indonesian location."""
        text_lower = location_text.lower()
        
        if 'provinsi' in text_lower:
            return 'province'
        elif any(indicator in text_lower for indicator in ['kota', 'kabupaten']):
            return 'city_regency'
        elif 'kecamatan' in text_lower:
            return 'district'
        elif any(indicator in text_lower for indicator in ['kelurahan', 'desa']):
            return 'village'
        elif any(indicator in text_lower for indicator in ['rt', 'rw']):
            return 'neighborhood'
        
        return 'unknown'

    def _determine_legal_hierarchy(self, legal_text: str) -> str:
        """Determine legal hierarchy level in Indonesian system."""
        text_lower = legal_text.lower()
        
        if any(indicator in text_lower for indicator in ['undang-undang', 'uu']):
            return 'constitutional_law'
        elif any(indicator in text_lower for indicator in ['peraturan pemerintah', 'pp']):
            return 'government_regulation'
        elif any(indicator in text_lower for indicator in ['perpres', 'peraturan presiden']):
            return 'presidential_regulation'
        elif any(indicator in text_lower for indicator in ['permen', 'peraturan menteri']):
            return 'ministerial_regulation'
        elif any(indicator in text_lower for indicator in ['keputusan', 'sk']):
            return 'decree'
        elif 'perda' in text_lower:
            return 'local_regulation'
        
        return 'other_legal'

    def _normalize_indonesian_text(self, text: str) -> str:
        """Normalize Indonesian text for better matching."""
        # Convert to lowercase
        normalized = text.lower()
        
        # Remove common Indonesian affixes for normalization
        normalized = re.sub(r'\b(ke|me|di|ter|ber|pe|per|se)\w*', lambda m: m.group(0)[2:] if len(m.group(0)) > 3 else m.group(0), normalized)
        
        # Normalize common variations
        normalized = normalized.replace('daerah khusus ibukota', 'dki')
        normalized = normalized.replace('daerah istimewa', 'di')
        
        return normalized.strip()
    
    async def create_entity_relationships(self, relationships: List[Dict]):
        """Create relationships between Indonesian entities in batches."""
        await self.kg_manager.create_indonesian_relationships_batch(relationships)
    
    @monitor_performance
    async def update_indonesian_kg_incremental(self, new_entities: List[Dict]) -> int:
        """Incrementally update Indonesian KG with new entities."""
        try:
            # Check for existing entities to avoid duplicates
            existing_names = set()
            
            async for entity in self.stream_indonesian_entities():
                existing_names.add(entity['name'])
            
            # Filter out existing entities
            new_entities_filtered = [
                entity for entity in new_entities 
                if entity['name'] not in existing_names
            ]
            
            # Validate new entities
            validated_entities = await self.validate_indonesian_entities(new_entities_filtered)
            
            # Create entity relationships in batches
            if validated_entities:
                relationships = self._create_entity_relationships(validated_entities)
                await self.create_entity_relationships(relationships)
            
            logger.info(f"Added {len(validated_entities)} new Indonesian entities")
            return len(validated_entities)
            
        except Exception as e:
            logger.error(f"Failed to update Indonesian KG incrementally: {e}")
            return 0
    
    def _create_entity_relationships(self, entities: List[Dict]) -> List[Dict]:
        """Create relationships between entities based on Indonesian context."""
        relationships = []
        
        # Simple co-occurrence based relationships
        for i, entity1 in enumerate(entities):
            for entity2 in entities[i+1:]:
                # Create relationship if entities are related by type or region
                if self._should_create_relationship(entity1, entity2):
                    relationships.append({
                        'source': entity1['name'],
                        'target': entity2['name'],
                        'type': 'CO_OCCURS',
                        'confidence': min(entity1.get('confidence', 0.5), entity2.get('confidence', 0.5))
                    })
        
        return relationships
    
    def _should_create_relationship(self, entity1: Dict, entity2: Dict) -> bool:
        """Determine if two entities should have a relationship."""
        # Same type entities in same document
        if entity1.get('type') == entity2.get('type'):
            return True
        
        # Government organization and location
        if (entity1.get('type') == 'government_org' and entity2.get('type') == 'regional') or \
           (entity1.get('type') == 'regional' and entity2.get('type') == 'government_org'):
            return True
        
        # Person and organization
        if (entity1.get('type') == 'person' and entity2.get('type') in ['government_org', 'organization']) or \
           (entity1.get('type') in ['government_org', 'organization'] and entity2.get('type') == 'person'):
            return True
        
        return False
    
    async def export_indonesian_kg_summary(self) -> Dict:
        """Export comprehensive summary of Indonesian Knowledge Graph."""
        try:
            metrics = await self.get_comprehensive_kg_metrics()
            
            # Get top entities by type
            top_entities = {}
            for entity_type in ['person', 'organization', 'government_org', 'regional', 'legal_entity']:
                entities = []
                async for entity in self.stream_indonesian_entities(entity_type, 10):
                    entities.append(entity)
                top_entities[entity_type] = entities
            
            return {
                'summary': {
                    'total_entities': metrics.get('derived_metrics', {}).get('total_entities', 0),
                    'quality_score': metrics.get('derived_metrics', {}).get('quality_score', 0),
                    'high_confidence_ratio': metrics.get('derived_metrics', {}).get('high_confidence_ratio', 0)
                },
                'metrics': metrics,
                'top_entities': top_entities,
                'export_timestamp': time.time()
            }
            
        except Exception as e:
            logger.error(f"Failed to export Indonesian KG summary: {e}")
            return {}


# Global instance
indonesian_kg_manager = IndonesianKGManager()