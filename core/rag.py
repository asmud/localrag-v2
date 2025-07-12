from typing import Dict, List, Optional

from loguru import logger

from core.config import settings
from core.ingestion import ingestion_pipeline
from core.models import model_manager
from core.postgres_storage import postgres_storage
from core.indonesian_kg_manager import indonesian_kg_manager
from core.llm_service import llm_service


class RAGEngine:
    def __init__(self):
        self.max_tokens = settings.max_tokens
        self.temperature = settings.temperature
        self.top_k = settings.top_k

    def _enhance_query(self, query: str) -> str:
        """Enhance query with conservative regulation pattern detection only."""
        enhanced_query = query.lower().strip()
        
        # Only enhance regulation patterns - no aggressive synonym expansion
        enhanced_query = self._expand_regulation_patterns(enhanced_query)
        
        # Only add minimal key terms if query is very short
        if len(query.split()) <= 2:
            enhanced_query = self._add_minimal_context_terms(enhanced_query)
        
        if enhanced_query != query.lower().strip():
            logger.info(f"Enhanced query: '{query}' → '{enhanced_query}'")
        return enhanced_query
    
    def _expand_regulation_patterns(self, query: str) -> str:
        """Expand regulation references to formal patterns."""
        import re
        
        # Pattern for Permendagri references
        permendagri_pattern = r'permendagri\s*(?:nomor\s*)?(\d+)\s*tahun\s*(\d{4})'
        match = re.search(permendagri_pattern, query)
        
        if match:
            number, year = match.groups()
            formal_ref = f"PERATURAN MENTERI DALAM NEGERI REPUBLIK INDONESIA NOMOR {number} TAHUN {year}"
            # Add both formal and informal patterns
            enhanced = f"{query} {formal_ref} MENDAGRI {number}/{year}"
            logger.info(f"Detected regulation pattern: {number}/{year}")
            return enhanced
        
        # Pattern for general regulation references
        reg_pattern = r'(?:peraturan\s*)?(?:menteri\s*)?(?:dalam\s*negeri\s*)?(?:nomor\s*)?(\d+)\s*tahun\s*(\d{4})'
        match = re.search(reg_pattern, query)
        
        if match:
            number, year = match.groups()
            enhanced = f"{query} NOMOR {number} TAHUN {year}"
            return enhanced
            
        return query
    
    def _add_minimal_context_terms(self, query: str) -> str:
        """Add minimal context terms only for very short queries."""
        # Only add one synonym for key Indonesian terms to avoid query dilution
        key_synonyms = {
            'apa': 'tentang',
            'isi': 'materi', 
            'tujuan': 'maksud',
            'klasifikasi': 'kategori'
        }
        
        words = query.split()
        for word in words:
            if word in key_synonyms:
                return f"{query} {key_synonyms[word]}"
        
        return query

    def _is_indonesian_query(self, query: str) -> bool:
        """Detect if query is in Indonesian language."""
        indonesian_indicators = [
            'apa', 'bagaimana', 'dimana', 'kapan', 'mengapa', 'siapa',
            'yang', 'dengan', 'dalam', 'untuk', 'dari', 'pada', 'oleh', 'sebagai',
            'adalah', 'akan', 'dapat', 'harus', 'perlu', 'tentang', 'kepada',
            'kementerian', 'dinas', 'badan', 'peraturan', 'undang-undang'
        ]
        
        query_lower = query.lower()
        indonesian_count = sum(1 for word in indonesian_indicators if word in query_lower)
        
        # Check for Indonesian question patterns
        question_patterns = ['apa itu', 'bagaimana cara', 'dimana tempat', 'kapan waktu', 'mengapa harus']
        pattern_count = sum(1 for pattern in question_patterns if pattern in query_lower)
        
        total_words = len(query.split())
        if total_words == 0:
            return False
        
        # Consider Indonesian if > 20% indicators or has question patterns
        indonesian_ratio = (indonesian_count + pattern_count * 2) / min(total_words, 20)
        return indonesian_ratio > 0.2

    async def retrieve_context(self, query: str, limit: Optional[int] = None) -> List[Dict]:
        limit = limit or self.top_k
        
        try:
            # Enhance query with regulation patterns and synonyms
            enhanced_query = self._enhance_query(query)
            
            # Generate query embedding using enhanced query
            query_embedding = await model_manager.get_embeddings([enhanced_query], is_query=True)
            query_vec = query_embedding[0]
            
            # Enhanced retrieval for Indonesian queries
            if self._is_indonesian_query(query):
                return await self._retrieve_indonesian_context(enhanced_query, query_vec, limit)
            else:
                return await self._retrieve_standard_context(enhanced_query, query_vec, limit)
            
        except Exception as e:
            logger.error(f"Failed to retrieve context: {e}")
            raise

    async def _retrieve_indonesian_context(self, query: str, query_vec: List[float], limit: int) -> List[Dict]:
        """Enhanced context retrieval for Indonesian queries."""
        context_chunks = []
        is_indonesian = True  # This method is only called for Indonesian queries
        
        # Get Indonesian entity context first
        try:
            indonesian_entities = []
            async for entity in indonesian_kg_manager.stream_indonesian_entities(limit=50):
                if any(keyword in query.lower() for keyword in entity['name'].lower().split()):
                    indonesian_entities.append(entity)
            
            if indonesian_entities:
                logger.info(f"Enhanced query with {len(indonesian_entities)} Indonesian entities")
                
                # Enhance query with entity context
                entity_terms = [entity['name'] for entity in indonesian_entities[:5]]
                enhanced_query = f"{query} {' '.join(entity_terms)}"
                enhanced_embedding = await model_manager.get_embeddings([enhanced_query], is_query=True)
                query_vec = enhanced_embedding[0]  # Use enhanced embedding
                
        except Exception as e:
            logger.warning(f"Failed to enhance query with Indonesian entities: {e}")
        
        # Primary search: PostgreSQL vector similarity
        try:
            pg_chunks = await postgres_storage.find_similar_chunks(
                query_embedding=query_vec,
                limit=limit,
                similarity_threshold=settings.similarity_threshold * settings.indonesian_similarity_multiplier if is_indonesian else settings.similarity_threshold
            )
            
            similarity_scores = []
            for chunk in pg_chunks:
                # Enhanced multi-factor similarity scoring
                base_similarity = chunk["similarity_score"]
                similarity_score = self._calculate_enhanced_similarity(
                    chunk, query, base_similarity
                )
                similarity_scores.append(similarity_score)
                chunk_metadata = chunk.get("metadata", {})
                
                context_chunks.append({
                    "content": chunk["content"],
                    "similarity": similarity_score,
                    "semantic_label": chunk_metadata.get("semantic_label", "content_text"),
                    "source": "postgresql",
                    "file_name": chunk["file_name"],
                    "chunk_index": chunk["chunk_index"],
                    "metadata": chunk_metadata,
                    "language": "indonesian"
                })
            
            if len(pg_chunks) > 0:
                threshold_used = settings.similarity_threshold * settings.indonesian_similarity_multiplier if is_indonesian else settings.similarity_threshold
                logger.info(f"Retrieved {len(pg_chunks)} chunks from PostgreSQL (threshold: {threshold_used:.2f})")
            
        except Exception as e:
            logger.warning(f"PostgreSQL search failed: {e}")
        
        # Enhanced Neo4j search for Indonesian content
        if len(context_chunks) < limit:
            try:
                remaining_limit = limit - len(context_chunks)
                neo4j_chunks = await ingestion_pipeline.search_similar_content(query, remaining_limit)
                
                for chunk in neo4j_chunks:
                    # Avoid duplicates
                    is_duplicate = any(
                        abs(existing["similarity"] - chunk["similarity"]) < 0.001 and 
                        existing["content"][:100] == chunk["content"][:100]
                        for existing in context_chunks
                    )
                    
                    if not is_duplicate:
                        context_chunks.append({
                            "content": chunk["content"],
                            "similarity": chunk["similarity"],
                            "semantic_label": chunk["semantic_label"],
                            "source": "neo4j",
                            "language": "indonesian"
                        })
                
                logger.info(f"Retrieved {len(neo4j_chunks)} additional Indonesian chunks from Neo4j")
                
            except Exception as e:
                logger.warning(f"Neo4j search failed: {e}")
        
        # Sort by similarity and limit results
        context_chunks.sort(key=lambda x: x["similarity"], reverse=True)
        context_chunks = context_chunks[:limit]
        
        # Filter out corrupted chunks before returning
        filtered_chunks = self._filter_corrupted_chunks(context_chunks)
        
        # Special debugging for DTSEN queries
        if any("dtsen" in query.lower() for query in [query] if "dtsen" in query.lower()):
            logger.info(f"DTSEN QUERY DEBUG - Indonesian context: {len(context_chunks)} total, {len(filtered_chunks)} after filtering")
            if context_chunks and not filtered_chunks:
                logger.warning(f"DTSEN QUERY DEBUG - All chunks were filtered out as corrupted!")
                for i, chunk in enumerate(context_chunks[:3]):  # Log first 3 chunks
                    logger.info(f"DTSEN QUERY DEBUG - Filtered chunk {i}: {chunk.get('content', '')[:100]}...")
        
        logger.info(f"Retrieved {len(context_chunks)} total Indonesian context chunks, {len(filtered_chunks)} after corruption filtering")
        return filtered_chunks

    async def _retrieve_standard_context(self, query: str, query_vec: List[float], limit: int) -> List[Dict]:
        """Standard context retrieval for non-Indonesian queries."""
        context_chunks = []
        
        # Primary search: PostgreSQL vector similarity (faster and more accurate)
        try:
            pg_chunks = await postgres_storage.find_similar_chunks(
                query_embedding=query_vec,
                limit=limit,
                similarity_threshold=settings.similarity_threshold
            )
            
            similarity_scores = []
            for chunk in pg_chunks:
                # Enhanced multi-factor similarity scoring
                base_similarity = chunk["similarity_score"]
                similarity_score = self._calculate_enhanced_similarity(
                    chunk, query, base_similarity
                )
                similarity_scores.append(similarity_score)
                chunk_metadata = chunk.get("metadata", {})
                
                context_chunks.append({
                    "content": chunk["content"],
                    "similarity": similarity_score,
                    "semantic_label": chunk_metadata.get("semantic_label", "content_text"),
                    "source": "postgresql",
                    "file_name": chunk["file_name"],
                    "chunk_index": chunk["chunk_index"],
                    "metadata": chunk_metadata
                })
            
            if len(pg_chunks) > 0:
                logger.info(f"Retrieved {len(pg_chunks)} chunks from PostgreSQL (threshold: {settings.similarity_threshold:.2f})")
            
        except Exception as e:
            logger.warning(f"PostgreSQL search failed: {e}")
        
        # Fallback/supplementary search: Neo4j (if PostgreSQL didn't return enough results)
        if len(context_chunks) < limit:
            try:
                remaining_limit = limit - len(context_chunks)
                neo4j_chunks = await ingestion_pipeline.search_similar_content(query, remaining_limit)
                
                for chunk in neo4j_chunks:
                    # Avoid duplicates by checking content similarity
                    is_duplicate = any(
                        abs(existing["similarity"] - chunk["similarity"]) < 0.001 and 
                        existing["content"][:100] == chunk["content"][:100]
                        for existing in context_chunks
                    )
                    
                    if not is_duplicate:
                        context_chunks.append({
                            "content": chunk["content"],
                            "similarity": chunk["similarity"],
                            "semantic_label": chunk["semantic_label"],
                            "source": "neo4j"
                        })
                
                logger.info(f"Retrieved {len(neo4j_chunks)} additional chunks from Neo4j")
                
            except Exception as e:
                logger.warning(f"Neo4j search failed: {e}")
        
        # Sort by similarity and limit results
        context_chunks.sort(key=lambda x: x["similarity"], reverse=True)
        context_chunks = context_chunks[:limit]
        
        # Filter out corrupted chunks before returning
        filtered_chunks = self._filter_corrupted_chunks(context_chunks)
        pg_count = len([c for c in filtered_chunks if c.get('source') == 'postgresql'])
        neo4j_count = len([c for c in filtered_chunks if c.get('source') == 'neo4j'])
        
        # Special debugging for DTSEN queries
        if "dtsen" in query.lower():
            logger.info(f"DTSEN QUERY DEBUG - Standard context: {len(context_chunks)} total, {len(filtered_chunks)} after filtering")
            if context_chunks and not filtered_chunks:
                logger.warning(f"DTSEN QUERY DEBUG - All standard chunks were filtered out as corrupted!")
                for i, chunk in enumerate(context_chunks[:3]):  # Log first 3 chunks
                    logger.info(f"DTSEN QUERY DEBUG - Filtered standard chunk {i}: {chunk.get('content', '')[:100]}...")
        
        logger.info(f"Retrieved {len(context_chunks)} total chunks, {len(filtered_chunks)} after corruption filtering (PG: {pg_count}, Neo4j: {neo4j_count})")
        return filtered_chunks

    async def generate_response(self, query: str, context_chunks: List[Dict], force_extractive: bool = False, use_llm: Optional[bool] = None) -> Dict:
        try:
            context_text = self._build_context_text(context_chunks)
            prompt = self._build_prompt(query, context_text)
            
            # Determine whether to use LLM based on parameters
            should_use_llm = self._should_use_llm(force_extractive, use_llm)
            
            # Try LLM generation if requested and available
            if should_use_llm and await llm_service.is_available():
                try:
                    response = await self._generate_llm_response(query, context_chunks)
                    generation_method = "llm_generation"
                    logger.info(f"Generated response using LLM ({settings.llm_provider})")
                except Exception as llm_error:
                    logger.warning(f"LLM generation failed: {llm_error}")
                    if settings.llm_fallback_to_extractive and not force_extractive:
                        logger.info("Falling back to extractive summarization")
                        response = await self._generate_extractive_response(query, context_chunks)
                        generation_method = "extractive_fallback"
                    else:
                        raise llm_error
            else:
                # Use extractive summarization
                response = await self._generate_extractive_response(query, context_chunks)
                if force_extractive:
                    generation_method = "extractive_forced"
                    logger.info("Using extractive summarization (forced)")
                elif use_llm is False:
                    generation_method = "extractive_requested"
                    logger.info("Using extractive summarization (requested)")
                else:
                    generation_method = "extractive_summarization"
                    if should_use_llm:
                        logger.info("LLM not available, using extractive summarization")
                    else:
                        logger.info("Using extractive summarization (LLM disabled)")
            
            # Build enhanced metadata
            metadata = {
                "retrieval_count": len(context_chunks),
                "average_similarity": sum(chunk["similarity"] for chunk in context_chunks) / len(context_chunks) if context_chunks else 0,
                "max_similarity": max((chunk["similarity"] for chunk in context_chunks), default=0),
                "min_similarity": min((chunk["similarity"] for chunk in context_chunks), default=0),
                "generation_method": generation_method,
                "llm_info": llm_service.get_provider_info() if should_use_llm else None,
                "extractive_forced": force_extractive,
                "llm_requested": use_llm,
                "llm_available": await llm_service.is_available()
            }
            
            # Add extractive-specific metrics if using extractive mode
            if generation_method.startswith("extractive"):
                extractive_metrics = self._calculate_extractive_metrics(response, query, context_chunks)
                metadata.update(extractive_metrics)
            
            response_data = {
                "query": query,
                "context_chunks": len(context_chunks),
                "context_text": context_text,
                "prompt": prompt,
                "response": response,
                "metadata": metadata
            }
            
            return response_data
            
        except Exception as e:
            logger.error(f"Failed to generate response: {e}")
            raise
    
    def _should_use_llm(self, force_extractive: bool, use_llm: Optional[bool]) -> bool:
        """Determine whether to use LLM based on parameters and settings."""
        if force_extractive:
            return False
        
        if use_llm is not None:
            return use_llm
        
        # If extractive mode is set as default, prefer extractive over LLM
        if settings.extractive_default_mode:
            return False
            
        return settings.enable_llm_chat
    
    def _calculate_extractive_metrics(self, response: str, query: str, context_chunks: List[Dict]) -> Dict:
        """Calculate metrics specific to extractive summarization."""
        try:
            # Basic response metrics
            response_length = len(response)
            response_words = len(response.split())
            response_sentences = len([s for s in response.split('.') if s.strip()])
            
            # Calculate overlap with source content
            response_words_set = set(response.lower().split())
            context_words_set = set()
            for chunk in context_chunks:
                context_words_set.update(chunk.get("content", "").lower().split())
            
            overlap_ratio = len(response_words_set.intersection(context_words_set)) / len(response_words_set) if response_words_set else 0
            
            # Calculate query coverage (how well the response addresses the query)
            query_words = set(query.lower().split())
            query_coverage = len(response_words_set.intersection(query_words)) / len(query_words) if query_words else 0
            
            # Estimate extractive confidence based on similarity and coverage
            avg_similarity = sum(chunk["similarity"] for chunk in context_chunks) / len(context_chunks) if context_chunks else 0
            extractive_confidence = (avg_similarity * 0.6 + overlap_ratio * 0.3 + query_coverage * 0.1)
            
            # Check for Indonesian content quality
            indonesian_quality = self._assess_indonesian_quality(response)
            
            return {
                "extractive_confidence": round(extractive_confidence, 3),
                "response_length": response_length,
                "response_words": response_words,
                "response_sentences": response_sentences,
                "content_overlap_ratio": round(overlap_ratio, 3),
                "query_coverage": round(query_coverage, 3),
                "indonesian_quality": indonesian_quality,
                "extractive_source_chunks": len(context_chunks)
            }
            
        except Exception as e:
            logger.warning(f"Failed to calculate extractive metrics: {e}")
            return {"extractive_confidence": 0.5}
    
    def _assess_indonesian_quality(self, text: str) -> float:
        """Assess the quality of Indonesian text."""
        try:
            # Basic quality indicators for Indonesian text
            indonesian_words = {
                'dan', 'yang', 'dengan', 'untuk', 'dari', 'pada', 'dalam', 'atau', 'oleh',
                'kepada', 'tentang', 'sebagai', 'adalah', 'akan', 'dapat', 'harus', 'perlu',
                'pemerintah', 'daerah', 'kabupaten', 'kota', 'provinsi', 'kementerian'
            }
            
            words = text.lower().split()
            if not words:
                return 0.0
                
            indonesian_count = sum(1 for word in words if word in indonesian_words)
            indonesian_ratio = indonesian_count / len(words)
            
            # Check for corruption patterns
            corruption_penalty = 0.5 if self._is_corrupted_text(text) else 0.0
            
            quality_score = max(0.0, indonesian_ratio - corruption_penalty)
            return round(quality_score, 3)
            
        except Exception:
            return 0.5

    def _build_context_text(self, context_chunks: List[Dict]) -> str:
        if not context_chunks:
            return "No relevant context found."
        
        context_parts = []
        for i, chunk in enumerate(context_chunks, 1):
            context_parts.append(f"[Context {i}] (Similarity: {chunk['similarity']:.3f})")
            content = chunk["content"]
            context_parts.append(content)
            context_parts.append("")
            
        
        final_context = "\n".join(context_parts)
        logger.info(f"Built context: {len(final_context)} chars, {len(context_chunks)} chunks")
        
        return final_context

    def _build_prompt(self, query: str, context: str) -> str:
        prompt_template = """You are a helpful AI assistant with access to relevant context information. 
Use the provided context to answer the user's question accurately and comprehensively.

Context Information:
{context}

User Question: {query}

Instructions:
- Base your answer primarily on the provided context
- If the context doesn't contain enough information, clearly state what's missing
- Be specific and cite relevant parts of the context when appropriate
- If you're uncertain about something, express that uncertainty
- Provide a helpful and well-structured response

Answer:"""
        
        return prompt_template.format(context=context, query=query)
    
    async def _generate_extractive_response(self, query: str, context_chunks: List[Dict]) -> str:
        """Generate response using extractive summarization from context chunks"""
        if not context_chunks:
            return "I couldn't find any relevant information to answer your question."
        
        try:
            # Validate and sanitize query
            query = self._sanitize_text(query)
            if not query:
                return "I couldn't process your question properly. Please try rephrasing it."
            
            # Get embeddings for the query to find most relevant sentences
            query_embedding = await model_manager.get_embeddings([query], is_query=True)
            if not query_embedding or not query_embedding[0]:
                logger.warning("Failed to generate query embedding, using fallback")
                return self._generate_fallback_response(query, context_chunks)
            
            query_vec = query_embedding[0]
            
            # Extract sentences from context chunks and score them
            all_sentences = []
            for chunk in context_chunks:
                content = chunk["content"]
                if not content or not isinstance(content, str):
                    continue
                
                # Sanitize content
                content = self._sanitize_text(content)
                if not content:
                    continue
                
                # Better sentence splitting for Indonesian text
                sentences = self._split_sentences_safe(content)
                for sentence in sentences:
                    if sentence and len(sentence.strip()) > 20:
                        # Validate sentence before adding
                        clean_sentence = self._sanitize_text(sentence)
                        if clean_sentence and not self._is_corrupted_text(clean_sentence):
                            all_sentences.append({
                                "text": clean_sentence,
                                "chunk_similarity": chunk["similarity"],
                                "sentence": clean_sentence
                            })
            
            if not all_sentences:
                return "I found relevant content but couldn't extract specific information to answer your question."
            
            # Get embeddings for all sentences
            sentence_texts = [s["text"] for s in all_sentences]
            logger.debug(f"Generating embeddings for {len(sentence_texts)} sentences")
            sentence_embeddings = await model_manager.get_embeddings(sentence_texts)
            
            if not sentence_embeddings or len(sentence_embeddings) != len(sentence_texts):
                logger.warning(f"Embedding generation failed or mismatch: expected {len(sentence_texts)}, got {len(sentence_embeddings) if sentence_embeddings else 0}")
                return self._generate_fallback_response(query, context_chunks)
            
            # Calculate similarity scores
            scored_sentences = []
            for i, sentence_data in enumerate(all_sentences):
                # Calculate cosine similarity with query
                sent_vec = sentence_embeddings[i]
                similarity = self._cosine_similarity(query_vec, sent_vec)
                
                # Combine with chunk similarity (weighted)
                combined_score = (similarity * 0.7) + (sentence_data["chunk_similarity"] * 0.3)
                
                scored_sentences.append({
                    "text": sentence_data["text"],
                    "score": combined_score,
                    "query_similarity": similarity,
                    "chunk_similarity": sentence_data["chunk_similarity"]
                })
            
            # Sort by combined score and take top sentences
            scored_sentences.sort(key=lambda x: x["score"], reverse=True)
            top_sentences = scored_sentences[:3]  # Take top 3 sentences
            
            # Build response
            if not top_sentences:
                logger.info("No top sentences found, using fallback")
                return self._generate_fallback_response(query, context_chunks)
            
            # Filter high confidence sentences
            high_conf_sentences = [s for s in top_sentences if s["score"] > 0.5]
            
            if not high_conf_sentences:
                logger.info(f"No high confidence sentences (best score: {max(s['score'] for s in top_sentences):.2f})")
                return "I found some potentially relevant information, but I'm not confident it directly answers your question. Please try rephrasing your query or provide more specific details."
            
            response_parts = []
            response_parts.append("")
            
            for sent in high_conf_sentences:
                sentence_text = sent['text'].strip()
                # Final validation of sentence before including
                if sentence_text and not self._is_corrupted_text(sentence_text):
                    if not sentence_text.endswith('.'):
                        sentence_text += '.'
                    response_parts.append(sentence_text)
                    logger.debug(f"Including sentence with score {sent['score']:.2f}: {sentence_text[:50]}...")
                else:
                    logger.warning(f"Skipping corrupted sentence: {sentence_text[:50]}...")
            
            # Check if we have any valid sentences
            if len(response_parts) <= 2:  # Only header and empty line
                logger.warning("All sentences were filtered out as corrupted")
                return self._generate_fallback_response(query, context_chunks)
            
            response_parts.append("")
            #response_parts.append(f"(Response generated from {len(context_chunks)} relevant document sections)")
            
            final_response = "\n".join(response_parts)
            logger.info(f"Generated extractive response with {len(high_conf_sentences)} sentences")
            
            return final_response
            
        except Exception as e:
            logger.error(f"Failed to generate extractive response: {e}")
            # Fallback to simple context concatenation
            return self._generate_fallback_response(query, context_chunks)
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        import math
        
        # Calculate dot product
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        
        # Calculate magnitudes
        mag1 = math.sqrt(sum(a * a for a in vec1))
        mag2 = math.sqrt(sum(b * b for b in vec2))
        
        if mag1 == 0 or mag2 == 0:
            return 0
        
        return dot_product / (mag1 * mag2)
    
    def _generate_fallback_response(self, query: str, context_chunks: List[Dict]) -> str:
        """Fallback response generation using simple text processing"""
        if not context_chunks:
            return "I couldn't find any relevant information to answer your question."
        
        try:
            # Use the highest similarity chunk as primary response
            best_chunk = max(context_chunks, key=lambda x: x.get("similarity", 0))
            
            response_parts = []
            response_parts.append("")
            
            # Extract and sanitize content
            content = best_chunk.get("content", "")
            if not content:
                return "I found relevant content but couldn't extract readable information from it."
            
            # Sanitize the content
            content = self._sanitize_text(content)
            if not content:
                return "I found relevant content but it appears to be corrupted or unreadable."
            
            # Check for corruption
            if self._is_corrupted_text(content):
                logger.warning("Fallback content appears corrupted, attempting repair")
                content = self._attempt_text_repair(content)
                if not content:
                    return "I found relevant content but it appears to be corrupted. Please try rephrasing your question."
            
            # Smart content extraction
            if len(content) > 300:
                # Use safe sentence splitting
                sentences = self._split_sentences_safe(content)
                if sentences:
                    # Find the most relevant sentence
                    best_sentence = self._find_best_sentence(sentences, query)
                    if best_sentence:
                        response_parts.append(f"{best_sentence}")
                    else:
                        # Fallback to safe truncation
                        safe_content = self._safe_truncate(content, 300)
                        response_parts.append(safe_content)
                else:
                    # Safe truncation at word boundary
                    safe_content = self._safe_truncate(content, 300)
                    response_parts.append(safe_content)
            else:
                response_parts.append(content)
            
            response_parts.append("")
            response_parts.append(f"(Response based on content with {best_chunk.get('similarity', 0):.1%} relevance)")
            
            final_response = "\n".join(response_parts)
            
            # Final validation
            if self._is_corrupted_text(final_response):
                return "I found relevant information but encountered issues processing it. Please try rephrasing your question."
            
            return final_response
            
        except Exception as e:
            logger.error(f"Fallback response generation failed: {e}")
            return "I encountered an error while processing the available information. Please try rephrasing your question."

    def _safe_truncate(self, text: str, max_length: int) -> str:
        """Safely truncate text at word boundaries"""
        if len(text) <= max_length:
            return text
        
        # Find last complete word within limit
        truncated = text[:max_length]
        last_space = truncated.rfind(' ')
        
        if last_space > max_length * 0.8:  # If we don't lose too much
            return truncated[:last_space] + "..."
        else:
            return truncated + "..."

    def _find_best_sentence(self, sentences: List[str], query: str) -> str:
        """Find the sentence most relevant to the query"""
        if not sentences or not query:
            return ""
        
        query_words = set(query.lower().split())
        best_sentence = ""
        best_score = 0
        
        for sentence in sentences:
            if len(sentence.strip()) < 20:  # Skip very short sentences
                continue
            
            sentence_words = set(sentence.lower().split())
            # Simple overlap scoring
            overlap = len(query_words.intersection(sentence_words))
            score = overlap / max(len(query_words), 1)
            
            if score > best_score:
                best_score = score
                best_sentence = sentence.strip()
        
        return best_sentence

    def _attempt_text_repair(self, text: str) -> str:
        """Attempt to repair corrupted text"""
        if not text:
            return ""
        
        try:
            # Handle the specific garbled pattern we've seen
            import re
            
            # Pattern like "U M A R G O R P" -> try to reverse or fix
            if re.match(r'^([A-Z]\s){5,}', text):
                # Remove spaces between single letters and reverse
                letters = re.findall(r'[A-Z]', text)
                if letters:
                    reversed_text = ''.join(reversed(letters))
                    # Check if this makes more sense
                    if len(reversed_text) > 5:
                        logger.info(f"Attempted text repair: {text[:30]} -> {reversed_text}")
                        return reversed_text
            
            # Remove excessive whitespace
            repaired = ' '.join(text.split())
            
            # If still looks corrupted, return empty
            if self._is_corrupted_text(repaired):
                return ""
            
            return repaired
            
        except Exception as e:
            logger.warning(f"Text repair failed: {e}")
            return ""

    def _sanitize_text(self, text: str) -> str:
        """Sanitize text to prevent corruption and encoding issues"""
        if not text or not isinstance(text, str):
            return ""
        
        try:
            # Ensure proper UTF-8 encoding
            if isinstance(text, bytes):
                text = text.decode('utf-8', errors='replace')
            
            # Remove null bytes and other problematic characters
            text = text.replace('\x00', '').replace('\ufffd', '')
            
            # Normalize whitespace but preserve Indonesian characters
            text = ' '.join(text.split())
            
            # Remove extremely long sequences of repeated characters (likely corruption)
            import re
            text = re.sub(r'(.)\1{10,}', r'\1\1\1', text)
            
            return text.strip()
        
        except Exception as e:
            logger.warning(f"Text sanitization failed: {e}")
            return ""

    def _split_sentences_safe(self, text: str) -> List[str]:
        """Safely split text into sentences, handling Indonesian text properly"""
        if not text:
            return []
        
        try:
            import re
            
            # Indonesian sentence boundary patterns
            # Handle periods followed by spaces and capital letters
            # Also handle common Indonesian punctuation
            sentence_pattern = r'(?<=[.!?])\s+(?=[A-Z\d])'
            
            # Split but be careful with abbreviations and numbers
            sentences = re.split(sentence_pattern, text)
            
            # Clean and validate each sentence
            clean_sentences = []
            for sentence in sentences:
                sentence = sentence.strip()
                if sentence and len(sentence) > 15:  # Minimum meaningful sentence length
                    # Check if sentence looks reasonable (not all caps, not garbled)
                    if self._is_valid_sentence(sentence):
                        clean_sentences.append(sentence)
            
            return clean_sentences
            
        except Exception as e:
            logger.warning(f"Sentence splitting failed: {e}")
            # Fallback to simple splitting
            return [s.strip() for s in text.split('.') if s.strip() and len(s.strip()) > 15]

    def _is_valid_sentence(self, sentence: str) -> bool:
        """Check if a sentence appears to be valid Indonesian text"""
        if not sentence or len(sentence) < 15:
            return False
        
        # Check for reasonable character distribution
        alpha_chars = sum(1 for c in sentence if c.isalpha())
        total_chars = len(sentence)
        
        # Sentence should be mostly alphabetic characters
        if total_chars > 0 and alpha_chars / total_chars < 0.6:
            return False
        
        # Check for excessive repetition (sign of corruption)
        import re
        if re.search(r'(.{3,})\1{3,}', sentence):
            return False
        
        # Check for reasonable word structure
        words = sentence.split()
        if len(words) < 3:  # Very short sentences might be headers/corrupted
            return False
        
        return True

    def _is_corrupted_text(self, text: str) -> bool:
        """Detect if text appears to be corrupted or garbled"""
        if not text:
            return True
        
        # Don't flag content that contains important acronyms as corrupted
        important_acronyms = ['dtsen', 'pkb', 'pdip', 'bssn', 'bappenas', 'kemenko', 'kemensos']
        if any(acronym in text.lower() for acronym in important_acronyms):
            logger.debug(f"Text contains important acronym, skipping corruption check: {text[:50]}...")
            return False
        
        # Check for patterns that indicate corruption
        import re
        
        # More specific patterns to avoid false positives with Indonesian text
        corruption_patterns = [
            # Very specific garbled patterns we've actually seen
            r'^\s*([A-Z]\s){15,}',  # 15+ single letters with spaces at start (more specific)
            r'([A-Z]\s){20,}',     # 20+ single letters with spaces anywhere (more specific)
            
            # Excessive repetition (clear corruption)
            r'(.{2,})\1{6,}',      # Same 2+ char sequence repeated 6+ times (more specific)
            
            # Lines that are mostly non-alphabetic (corrupted structure)
            r'^[^a-zA-Z]{25,}$',   # 25+ non-letters only (more specific)
            
            # Very long all-caps with spaces (clear garbled text)
            r'^[A-Z\s]{80,}$',     # 80+ chars of only caps and spaces (much more specific)
        ]
        
        for pattern in corruption_patterns:
            if re.search(pattern, text):
                logger.debug(f"Detected corrupted text pattern: {pattern} in: {text[:50]}...")
                return True
        
        # Additional validation for Indonesian text
        if self._is_likely_garbled_indonesian(text):
            logger.debug(f"Detected garbled Indonesian pattern in: {text[:50]}...")
            return True
        
        return False

    def _is_likely_garbled_indonesian(self, text: str) -> bool:
        """Check for specific Indonesian garbled patterns we've seen"""
        import re
        
        # The specific corrupted patterns from Indonesian government docs
        indonesian_garbled_patterns = [
            # Common reversed patterns found in actual data
            r'K\s+B\s+U\s+S\s+N\s+A\s+T\s+A\s+I\s+G\s+E\s+K',  # "KEGIATAN SUBKEGIATAN" reversed
            r'A\s+R\s+G\s+O\s+R\s+P\s+N\s+A\s+T\s+A\s+I\s+G\s+E\s+K',  # "KEGIATAN PROGRAM" reversed  
            r'S\s+U\s+R\s+U\s+M\s+A\s+R\s+G\s+O\s+R\s+P',      # "PROGRAM URUSAN" reversed
            r'O\s+R\s+P\s+N\s+A\s+T\s+A\s+I\s+G\s+E\s+K',      # "KEGIATAN PRO" reversed
            r'M\s+A\s+R\s+G\s+O\s+R\s+P\s+N\s+A\s+T\s+A\s+I\s+G\s+E\s+K',  # "KEGIATAN PROGRAM" reversed
            
            # Specific spacing patterns that indicate corruption
            r'^[A-Z]\s+[A-Z]\s+[A-Z]\s+[A-Z]\s+[A-Z]\s+[A-Z]\s+[A-Z]\s+[A-Z]\s+[A-Z]\s+[A-Z]',  # 10+ spaced letters at start
            r'([A-Z]\s){12,}(NOMENKLATUR|URUSAN|KABUPATEN|KOTA)',  # 12+ letters + keywords
            
            # Full corrupted sequences we've seen
            r'MARGORP\s+NATAIGEK\s+NATAIGEBUSK',        # Exact reversed pattern
            r'^\s*([A-Z]\s){15,}[A-Z]?\s*$',           # 15+ single letters only
        ]
        
        for pattern in indonesian_garbled_patterns:
            if re.search(pattern, text.upper()):
                return True
        
        return False

    def _filter_corrupted_chunks(self, chunks: List[Dict]) -> List[Dict]:
        """Filter out chunks with corrupted content to improve result quality."""
        if not chunks:
            return chunks
        
        filtered_chunks = []
        corrupted_count = 0
        
        for chunk in chunks:
            content = chunk.get("content", "")
            if not content:
                continue
                
            # Special debugging for DTSEN content
            if "dtsen" in content.lower():
                logger.info(f"DTSEN QUERY DEBUG - Found chunk with DTSEN content: {content[:100]}...")
                logger.info(f"DTSEN QUERY DEBUG - Is corrupted: {self._is_corrupted_text(content)}")
                logger.info(f"DTSEN QUERY DEBUG - Has excessive spacing: {self._has_excessive_spacing(content)}")
                
            # Check if chunk content is corrupted
            if self._is_corrupted_text(content):
                corrupted_count += 1
                if "dtsen" in content.lower():
                    logger.warning(f"DTSEN QUERY DEBUG - Filtering out corrupted DTSEN chunk: {content[:100]}...")
                else:
                    logger.debug(f"Filtered out corrupted chunk: {content[:50]}...")
                continue
            
            # Additional quick checks for common corruption patterns
            if self._has_excessive_spacing(content):
                corrupted_count += 1
                if "dtsen" in content.lower():
                    logger.warning(f"DTSEN QUERY DEBUG - Filtering out spaced DTSEN chunk: {content[:100]}...")
                else:
                    logger.debug(f"Filtered out spaced-out chunk: {content[:50]}...")
                continue
                
            filtered_chunks.append(chunk)
        
        if corrupted_count > 0:
            logger.info(f"Filtered out {corrupted_count} corrupted chunks from results")
        
        return filtered_chunks
    
    def _has_excessive_spacing(self, text: str) -> bool:
        """Quick check for excessive spacing that indicates corruption."""
        # Count single letters followed by spaces
        import re
        single_letter_spaces = len(re.findall(r'\b[A-Z]\s+', text))
        total_words = len(text.split())
        
        # If more than 50% of "words" are single letters with spaces, likely corrupted
        if total_words > 0 and (single_letter_spaces / total_words) > 0.5:
            return True
            
        return False
    
    def _is_clean_readable_content(self, text: str) -> bool:
        """Check if content is clean and readable (bonus for good content)."""
        if not text or len(text.strip()) < 20:
            return False
        
        # Check for good indicators of readable content
        import re
        
        # Count sentences (periods, exclamation marks, question marks)
        sentences = len(re.findall(r'[.!?]+', text))
        words = len(text.split())
        
        # Readable content should have reasonable sentence structure
        if words > 0 and sentences > 0:
            words_per_sentence = words / sentences
            # Good range: 5-25 words per sentence
            if 5 <= words_per_sentence <= 25:
                return True
        
        # Check for proper Indonesian/formal language indicators
        formal_indicators = [
            'yang', 'dengan', 'dalam', 'untuk', 'dari', 'pada', 'oleh', 'sebagai',
            'adalah', 'akan', 'dapat', 'harus', 'perlu', 'tentang', 'kepada',
            'berdasarkan', 'sesuai', 'mengenai', 'terhadap', 'melalui'
        ]
        
        text_lower = text.lower()
        formal_count = sum(1 for indicator in formal_indicators if indicator in text_lower)
        
        # If content has multiple formal language indicators, it's likely good content
        if formal_count >= 3:
            return True
        
        return False

    async def _generate_llm_response(self, query: str, context_chunks: List[Dict]) -> str:
        """Generate response using LLM with retrieved context"""
        try:
            response = await llm_service.generate_response(query, context_chunks)
            return response
        except Exception as e:
            logger.error(f"Failed to generate LLM response: {e}")
            raise

    async def query(self, question: str, limit: Optional[int] = None, force_extractive: bool = False, use_llm: Optional[bool] = None) -> Dict:
        try:
            logger.info(f"Processing RAG query: {question}")
            
            # Check if this is an Indonesian query and enhance accordingly
            is_indonesian = self._is_indonesian_query(question)
            
            context_chunks = await self.retrieve_context(question, limit)
            
            if not context_chunks:
                logger.warning(f"No context chunks found for query: '{question}' (Indonesian: {is_indonesian})")
                return {
                    "query": question,
                    "response": "I couldn't find any relevant information to answer your question." if not is_indonesian else "Saya tidak dapat menemukan informasi yang relevan untuk menjawab pertanyaan Anda.",
                    "context_chunks": 0,
                    "language": "indonesian" if is_indonesian else "other",
                    "metadata": {
                        "retrieval_count": 0,
                        "average_similarity": 0,
                        "max_similarity": 0,
                        "min_similarity": 0,
                        "language_detected": "indonesian" if is_indonesian else "other",
                        "query_enhanced": False,
                        "threshold_used": settings.similarity_threshold * settings.indonesian_similarity_multiplier if is_indonesian else settings.similarity_threshold
                    }
                }
            
            response_data = await self.generate_response(question, context_chunks, force_extractive=force_extractive, use_llm=use_llm)
            
            # Add Indonesian-specific enhancements
            if is_indonesian:
                response_data = await self._enhance_indonesian_response(response_data, question, context_chunks)
            
            logger.info(f"Successfully processed RAG query with {len(context_chunks)} context chunks")
            return response_data
            
        except Exception as e:
            logger.error(f"Failed to process RAG query: {e}")
            raise

    async def _enhance_indonesian_response(self, response_data: Dict, question: str, context_chunks: List[Dict]) -> Dict:
        """Enhance response with Indonesian-specific information."""
        try:
            # Get Indonesian KG metrics for additional context
            kg_metrics = await indonesian_kg_manager.get_comprehensive_kg_metrics()
            
            # Add Indonesian entity validation
            response_text = response_data.get('response', '')
            if response_text:
                # Extract potential entities from response for validation
                response_entities = []
                words = response_text.split()
                
                # Simple entity extraction from response
                for i, word in enumerate(words):
                    if word.istitle() and len(word) > 3:  # Potential entity
                        # Check if it's a known Indonesian entity
                        async for entity in indonesian_kg_manager.stream_indonesian_entities(limit=100):
                            if word.lower() in entity['name'].lower():
                                response_entities.append({
                                    'text': word,
                                    'matched_entity': entity['name'],
                                    'entity_type': entity.get('entity_type', 'unknown'),
                                    'confidence': entity.get('confidence', 0.0)
                                })
                                break
                
                # Add Indonesian-specific metadata
                response_data['indonesian_context'] = {
                    'entities_in_response': response_entities,
                    'kg_quality_score': kg_metrics.get('derived_metrics', {}).get('quality_score', 0),
                    'total_indonesian_entities': kg_metrics.get('derived_metrics', {}).get('total_entities', 0),
                    'confidence_distribution': kg_metrics.get('basic_metrics', {}).get('entity_distribution', {})
                }
                
                # Add language marker
                response_data['language'] = 'indonesian'
                response_data['metadata']['language_detected'] = 'indonesian'
                
                logger.info(f"Enhanced Indonesian response with {len(response_entities)} validated entities")
            
            return response_data
            
        except Exception as e:
            logger.warning(f"Failed to enhance Indonesian response: {e}")
            return response_data

    def _calculate_enhanced_similarity(self, chunk: Dict, query: str, base_similarity: float) -> float:
        """Calculate enhanced similarity score with simplified factors."""
        score = base_similarity
        chunk_metadata = chunk.get("metadata", {})
        content = chunk.get("content", "")
        
        # Factor 0: Content quality penalty (most important)
        if self._is_corrupted_text(content):
            score *= 0.3  # Heavy penalty for corrupted content
            logger.debug(f"Applied corruption penalty: {base_similarity:.3f} → {score:.3f}")
        elif self._has_excessive_spacing(content):
            score *= 0.5  # Moderate penalty for spaced content
            logger.debug(f"Applied spacing penalty: {base_similarity:.3f} → {score:.3f}")
        
        # Factor 1: Document structure priority (simplified)
        section_type = chunk_metadata.get("section_type", "")
        if section_type == "document_header":
            score *= 1.15  # Moderate boost for headers
        elif section_type == "introduction":
            score *= 1.1   # Small boost for introductions
        elif section_type == "appendix":
            score *= 0.9   # Small reduction for appendix
        
        # Factor 2: Exact phrase matching (most important)
        if self._has_exact_phrase_match(content, query):
            score *= 1.2   # Significant boost for exact matches
            logger.debug(f"Applied exact phrase boost: {base_similarity:.3f} → {score:.3f}")
        
        # Factor 3: Document position priority (simplified)
        chunk_index = chunk.get("chunk_index", 0)
        if chunk_index <= 3:  # First 3 chunks only
            score *= 1.05  # Small boost for early content
        
        # Factor 4: Content readability bonus
        if self._is_clean_readable_content(content):
            score *= 1.1   # Small boost for clean, readable content
            logger.debug(f"Applied readability bonus: {score:.3f}")
        
        # Ensure score doesn't exceed 1.0
        final_score = min(score, 1.0)
        
        if abs(final_score - base_similarity) > 0.01:  # Only log significant changes
            logger.debug(f"Enhanced similarity: {base_similarity:.3f} → {final_score:.3f}")
        
        return final_score
    
    def _contains_regulation_reference(self, content: str, query: str) -> bool:
        """Check if content contains regulation references mentioned in query."""
        import re
        
        # Extract regulation patterns from query
        query_lower = query.lower()
        content_lower = content.lower()
        
        # Pattern for Permendagri references
        permendagri_pattern = r'permendagri.*nomor.*(\d+).*tahun.*(\d{4})|nomor.*(\d+).*tahun.*(\d{4})'
        
        query_matches = re.findall(permendagri_pattern, query_lower)
        if query_matches:
            # Check if same regulation number appears in content
            for match in query_matches:
                number = match[0] or match[2]  # First or third capture group
                year = match[1] or match[3]    # Second or fourth capture group
                
                if number and year:
                    # Look for the same number and year in content
                    content_pattern = f"(nomor.*{number}.*tahun.*{year}|{number}.*tahun.*{year})"
                    if re.search(content_pattern, content_lower):
                        return True
        
        # Check for direct regulation title matches
        regulation_keywords = ["peraturan menteri", "permendagri", "mendagri"]
        for keyword in regulation_keywords:
            if keyword in query_lower and keyword in content_lower:
                return True
        
        return False
    
    def _has_exact_phrase_match(self, content: str, query: str) -> bool:
        """Check if content contains exact phrases from query."""
        import re
        
        # Clean and normalize text
        content_clean = re.sub(r'\s+', ' ', content.lower().strip())
        query_clean = re.sub(r'\s+', ' ', query.lower().strip())
        
        # Extract meaningful phrases (3+ words) from query
        query_words = query_clean.split()
        phrases_to_check = []
        
        # Generate 3-word phrases
        for i in range(len(query_words) - 2):
            phrase = ' '.join(query_words[i:i+3])
            if len(phrase) > 10:  # Skip very short phrases
                phrases_to_check.append(phrase)
        
        # Generate 4-word phrases for better accuracy
        for i in range(len(query_words) - 3):
            phrase = ' '.join(query_words[i:i+4])
            phrases_to_check.append(phrase)
        
        # Check if any phrase exists in content
        for phrase in phrases_to_check:
            if phrase in content_clean:
                return True
        
        # Also check for key Indonesian question terms with context
        key_terms = {
            'intisari': ['intisari', 'ringkasan', 'isi pokok'],
            'nomenklatur': ['nomenklatur', 'nama', 'penamaan'],
            'klasifikasi': ['klasifikasi', 'penggolongan', 'kategori']
        }
        
        for term, synonyms in key_terms.items():
            if term in query_clean:
                for synonym in synonyms:
                    if synonym in content_clean:
                        return True
        
        return False


class TemporalEngine:
    """Handles temporal awareness and time-based context"""
    
    def __init__(self):
        self.temporal_window = 30  # days
    
    async def get_temporal_context(self, query: str, timestamp: Optional[str] = None) -> Dict:
        """Get temporal context for a query"""
        try:
            logger.info(f"Getting temporal context for query: {query}")
            
            # For now, return empty context - can be enhanced later
            return {
                "temporal_relevance": 0.0,
                "time_references": [],
                "context_age": None,
                "freshness_score": 1.0
            }
        except Exception as e:
            logger.error(f"Failed to get temporal context: {e}")
            return {}
    
    async def enhance_with_temporal_info(self, response: str, context_chunks: List[Dict]) -> str:
        """Enhance response with temporal information"""
        # Simple implementation - can be expanded later
        return response


class HallucinationReducer:
    """Reduces hallucination by validating responses against knowledge base"""
    
    def __init__(self):
        self.confidence_threshold = 0.7
        self.validation_enabled = True
    
    async def validate_response(self, response: str, context_chunks: List[Dict]) -> Dict:
        """Validate response against provided context to reduce hallucinations"""
        try:
            if not self.validation_enabled:
                return {"validated": False, "confidence": 1.0, "issues": []}
            
            validation_result = {
                "validated": True,
                "confidence": 1.0,
                "issues": [],
                "recommendations": []
            }
            
            # Basic validation checks
            if not response or len(response.strip()) < 10:
                validation_result["issues"].append("Response too short")
                validation_result["confidence"] *= 0.5
            
            if not context_chunks:
                validation_result["issues"].append("No context available for validation")
                validation_result["confidence"] *= 0.7
            
            # Check if response seems to be based on context
            if context_chunks and response:
                context_text = " ".join([chunk.get("content", "") for chunk in context_chunks])
                if len(context_text) > 50:
                    # Simple keyword overlap check
                    response_words = set(response.lower().split())
                    context_words = set(context_text.lower().split())
                    overlap = len(response_words.intersection(context_words))
                    overlap_ratio = overlap / len(response_words) if response_words else 0
                    
                    if overlap_ratio < 0.1:
                        validation_result["issues"].append("Low overlap with provided context")
                        validation_result["confidence"] *= 0.8
                        validation_result["recommendations"].append("Consider using more context-based information")
            
            # Final validation status
            validation_result["validated"] = validation_result["confidence"] >= self.confidence_threshold
            
            logger.info(f"Response validation completed: confidence={validation_result['confidence']:.2f}")
            return validation_result
            
        except Exception as e:
            logger.error(f"Failed to validate response: {e}")
            return {"validated": False, "confidence": 0.0, "issues": [f"Validation error: {e}"]}
    
    async def enhance_response_reliability(self, response: str, context_chunks: List[Dict]) -> Dict:
        """Enhance response reliability with additional checks"""
        try:
            validation = await self.validate_response(response, context_chunks)
            
            enhanced_response = response
            if validation["issues"]:
                # Add disclaimer if issues found
                disclaimer = "\n\n[Note: This response may have limited accuracy due to validation concerns]"
                enhanced_response += disclaimer
            
            return {
                "original_response": response,
                "enhanced_response": enhanced_response,
                "validation": validation,
                "reliability_score": validation["confidence"]
            }
            
        except Exception as e:
            logger.error(f"Failed to enhance response reliability: {e}")
            return {
                "original_response": response,
                "enhanced_response": response,
                "validation": {"validated": False, "confidence": 0.0, "issues": [str(e)]},
                "reliability_score": 0.0
            }


# Initialize engines
rag_engine = RAGEngine()
temporal_engine = TemporalEngine()
hallucination_reducer = HallucinationReducer()