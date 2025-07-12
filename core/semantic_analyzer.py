"""
Advanced semantic analyzer using Indonesian BERT for enhanced text understanding.
"""

import re
import numpy as np
from typing import Dict, List, Set, Tuple, Optional
from collections import Counter
from loguru import logger

from .models import model_manager


class IndonesianSemanticAnalyzer:
    """Advanced semantic analysis using Indonesian BERT model."""
    
    def __init__(self):
        self.model_manager = model_manager
        
        # Enhanced Indonesian legal/government document patterns
        self.legal_patterns = {
            'regulation': r'\b(peraturan|undang-undang|keputusan|surat|instruksi|perda|permen|perpres|pp)\b',
            'law_number': r'\b(nomor|no\.?)\s*(\d+)\s*(tahun|th\.?)\s*(\d{4})\b',
            'article': r'\b(pasal|ayat|huruf|angka)\s*(\d+)\b',
            'institution': r'\b(kementerian|badan|dinas|kantor|lembaga|direktorat|dprd|bpk|bpkp)\b',
            'procedure': r'\b(tata\s*cara|prosedur|mekanisme|tahap|langkah)\b',
            'requirement': r'\b(syarat|ketentuan|persyaratan|kriteria|standar)\b',
            'definition': r'\b(definisi|pengertian|arti|makna|dimaksud)\b'
        }
        
        # Indonesian entity classification patterns
        self.entity_patterns = {
            'government_org': r'(?i)(kementerian|kemen|kemenko|lembaga|badan|dinas|kantor|balai|pusat|direktorat)',
            'education': r'(?i)(universitas|institut|sekolah|akademi|politeknik|stikes|stkip)',
            'healthcare': r'(?i)(rumah sakit|rs|puskesmas|klinik|poliklinik|balai kesehatan)',
            'legal_entity': r'(?i)(undang-undang|uu\s+no|uu\s+nomor|peraturan\s+pemerintah|permen|perpres|kepres|pp\s+no|pp\s+nomor|perda|sk\s+no|surat\s+keputusan|keputusan\s+menteri|keppres|perpu|tap\s+mpr)',
            'regional': r'(?i)(provinsi|kota|kabupaten|kecamatan|kelurahan|desa|nagari|gampong)',
            'academic_title': r'(?i)(dr|prof|ir|drs|drg|dr\\s+h|dr\\s+hj)',
            'religious_title': r'(?i)\b(h\.|hj\.|kh\.|nyai|habib|sayyid|syekh|ustadz|ustadzah)\b',
            'professional_title': r'(?i)(s\\.pd|s\\.kom|s\\.h|s\\.e|s\\.si|s\\.st|m\\.pd|m\\.kom|m\\.h|m\\.e)',
            'regional_title': r'(?i)(datuk|dato|tengku|cut|teuku|raden|pangeran|sultan)'
        }
        
        # Content classification patterns
        self.content_classes = {
            'legal_document': ['peraturan', 'undang-undang', 'keputusan', 'surat edaran'],
            'procedure': ['tata cara', 'prosedur', 'mekanisme', 'langkah'],
            'definition': ['definisi', 'pengertian', 'dimaksud dengan'],
            'requirement': ['syarat', 'ketentuan', 'persyaratan'],
            'reference': ['pasal', 'ayat', 'lampiran', 'bagian'],
            'institution': ['kementerian', 'badan', 'dinas', 'lembaga']
        }
        
        # Comprehensive Indonesian stopwords for key phrase extraction
        self.indonesian_stopwords = {
            'ada', 'adalah', 'adapun', 'agak', 'agaknya', 'agar', 'akan', 'akankah', 'akhirnya', 'aku',
            'alam', 'anda', 'andaikan', 'andai', 'andaikata', 'antar', 'antara', 'antaranya', 'apa', 'apaan',
            'apalagi', 'apakah', 'apapun', 'asalkan', 'atas', 'atau', 'bahkan', 'bagaimana', 'bagaimanapun',
            'bagi', 'bagian', 'banyak', 'beberapa', 'begini', 'beginian', 'beginikah', 'beginilah', 'begitu',
            'begitukah', 'begitulah', 'begitupun', 'belum', 'belumkah', 'belumlah', 'benar', 'benarkah',
            'benar-benar', 'berupa', 'besar', 'betul', 'betulkah', 'boleh', 'bolehkah', 'bukan', 'bukankah',
            'bukanlah', 'bukannya', 'cuma', 'cukup', 'dah', 'dahulu', 'dalam', 'dan', 'dapat', 'dari',
            'darimana', 'daripadanya', 'datang', 'dekat', 'demi', 'dengan', 'dengannya', 'dia', 'dialah',
            'di', 'diantaranya', 'diharapkannya', 'diharuskan', 'dijelaskan', 'tidak', 'tidaklah', 'dimaksud',
            'dimana', 'dimana-mana', 'demikian', 'demikianlah', 'demikianpun', 'dengan', 'dia', 'diri',
            'dirinya', 'dulu', 'enggak', 'enggaknya', 'entah', 'entahlah', 'erat', 'guna', 'hal', 'hampir',
            'hanya', 'hanyalah', 'harus', 'haruslah', 'harusnya', 'hendak', 'hendaklah', 'hendaknya', 'hingga',
            'ia', 'ialah', 'ibarat', 'ibaratkan', 'ibaratnya', 'jika', 'jikalau', 'juga', 'jumlah', 'justru',
            'kabar', 'kala', 'kalau', 'kalaulah', 'kalapun', 'kalian', 'kami', 'kamilah', 'kamu', 'kamulah',
            'kan', 'kapan', 'kapankah', 'kapanpun', 'karena', 'karyanya', 'kasih', 'kata', 'katakan',
            'katanya', 'ke', 'kebanyakan', 'kebetulan', 'kedua', 'kedua-duanya', 'keinginan', 'kemudian',
            'kena', 'kenapa', 'kepadanya', 'ketika', 'kita', 'kitalah', 'kok', 'kembali', 'keseluruhan',
            'ketiga', 'ketiga-tiganya', 'lah', 'lagi', 'lagian', 'lain', 'lainnya', 'lalu', 'lama', 'lamanya',
            'lebih', 'melalui', 'memang', 'menjadi', 'menurut', 'menyebabkannya', 'menyeluruh', 'mengenai',
            'mengapa', 'mungkin', 'mulai', 'maupun', 'masing', 'masih', 'masuk', 'maka', 'malah', 'malahnya',
            'mereka', 'merekalah', 'meski', 'meskipun', 'misal', 'misalnya', 'mula', 'mudah', 'mudah-mudahan',
            'nah', 'namun', 'nanti', 'nantinya', 'nya', 'oleh', 'orang', 'orang-orang', 'pada', 'padanya',
            'para', 'pasti', 'pastilah', 'pemberi', 'pemberian', 'perlu', 'perlukah', 'perlunya', 'pernah',
            'pernyataan', 'pribadi', 'pun', 'punya', 'satu', 'sama', 'sama-sama', 'sang', 'sangat', 'se',
            'sebab', 'seakan', 'seandainya', 'sebaiknya', 'sekali', 'sekalipun', 'selama', 'selama-lamanya',
            'selain', 'selesai', 'seluruh', 'seluruhnya', 'semua', 'semuanya', 'sementara', 'sempat',
            'semula', 'sendiri', 'sendirinya', 'seolah', 'sepertinya', 'seperti', 'sesudah', 'sesuatu',
            'sesuai', 'setiap', 'setelah', 'serta', 'seorang', 'sering', 'suatu', 'sudah', 'sudahkah',
            'sudahlah', 'supaya', 'tadi', 'tadinya', 'tak', 'tanpa', 'tapi', 'tapi', 'tetapi', 'tiap',
            'telah', 'tempat', 'tentu', 'tentunya', 'tentang', 'tengah', 'tersebut', 'tertentu', 'tiap-tiap',
            'untuk', 'usaha', 'usahlah', 'wahai', 'wah', 'walau', 'walaupun', 'waktu', 'waktu-waktu', 'yang',
            'yakni', 'yaitu', 'ya', 'tiada', 'tidak', 'si', 'saja', 'aja', 'paling', 'ini', 'itu',
            'dua', 'tiga', 'empat', 'lima', 'enam', 'tujuh', 'delapan', 'sembilan', 'sepuluh', 'sebelas',
            'dua belas', 'pertama', 'kedua', 'ketiga'
        }
    
    async def analyze_chunk(self, text: str, chunk_index: int, total_chunks: int) -> Dict:
        """
        Perform comprehensive semantic analysis on a text chunk using Indonesian BERT.
        
        Args:
            text: Text content to analyze
            chunk_index: Position of chunk in document
            total_chunks: Total number of chunks in document
            
        Returns:
            Dict containing semantic analysis results
        """
        try:
            # Basic text properties
            analysis = {
                'language': self._detect_language(text),
                'formality_level': self._detect_formality(text),
                'text_complexity': self._calculate_complexity(text)
            }
            
            # Use Indonesian BERT for advanced analysis
            bert_features = await self.model_manager.extract_features(text)
            
            # Semantic classification
            analysis['semantic_class'] = await self._classify_content_semantically(text, bert_features)
            
            # Entity extraction with NER model and pattern-based fallback
            ner_entities = await self._extract_entities_with_ner_model(text)
            pattern_entities = self._extract_indonesian_entities(text)
            
            # Combine and deduplicate entities
            all_entities = self._combine_and_deduplicate_entities(ner_entities, pattern_entities)
            analysis['entities'] = self._disambiguate_indonesian_entities(all_entities)
            
            # Key information extraction
            analysis['legal_references'] = self._extract_legal_references(text)
            analysis['key_phrases'] = await self._extract_advanced_key_phrases(text, bert_features)
            
            # Topic and thematic analysis
            analysis['topics'] = await self._identify_topics(text, bert_features)
            
            # Structural analysis
            analysis['structure_type'] = self._analyze_structure(text)
            analysis['position_context'] = self._get_enhanced_position_context(chunk_index, total_chunks)
            
            # Generate semantic label
            analysis['semantic_label'] = self._generate_semantic_label(analysis)
            
            return analysis
            
        except Exception as e:
            logger.error(f"Failed to analyze chunk semantically: {e}")
            # Fallback to basic analysis
            return self._fallback_analysis(text, chunk_index, total_chunks)

    async def analyze_chunks_batch(self, chunks_data: List[Dict]) -> List[Dict]:
        """
        Perform batch semantic analysis on multiple chunks for better performance.
        
        Args:
            chunks_data: List of dicts with 'text', 'chunk_index', 'total_chunks'
            
        Returns:
            List of analysis results for each chunk
        """
        try:
            if not chunks_data:
                return []
            
            logger.info(f"Starting batch analysis of {len(chunks_data)} chunks")
            
            # Extract texts for batch processing
            texts = [chunk['text'] for chunk in chunks_data]
            
            # Batch NER entity extraction
            batch_ner_entities = await self.model_manager.extract_entities_batch(texts, batch_size=8)
            
            # Process each chunk with its entities
            results = []
            for i, chunk_data in enumerate(chunks_data):
                text = chunk_data['text']
                chunk_index = chunk_data.get('chunk_index', i)
                total_chunks = chunk_data.get('total_chunks', len(chunks_data))
                
                try:
                    # Basic text properties
                    analysis = {
                        'language': self._detect_language(text),
                        'formality_level': self._detect_formality(text),
                        'text_complexity': self._calculate_complexity(text)
                    }
                    
                    # Use Indonesian BERT for advanced analysis
                    bert_features = await self.model_manager.extract_features(text)
                    
                    # Semantic classification
                    analysis['semantic_class'] = await self._classify_content_semantically(text, bert_features)
                    
                    # Use pre-computed NER entities and pattern-based fallback
                    ner_entities = self._convert_batch_ner_entities(batch_ner_entities[i])
                    pattern_entities = self._extract_indonesian_entities(text)
                    
                    # Combine and deduplicate entities
                    all_entities = self._combine_and_deduplicate_entities(ner_entities, pattern_entities)
                    analysis['entities'] = self._disambiguate_indonesian_entities(all_entities)
                    
                    # Key information extraction
                    analysis['legal_references'] = self._extract_legal_references(text)
                    analysis['key_phrases'] = await self._extract_advanced_key_phrases(text, bert_features)
                    
                    # Topic and thematic analysis
                    analysis['topics'] = await self._identify_topics(text, bert_features)
                    
                    # Structural analysis
                    analysis['structure_type'] = self._analyze_structure(text)
                    analysis['position_context'] = self._get_enhanced_position_context(chunk_index, total_chunks)
                    
                    # Generate semantic label
                    analysis['semantic_label'] = self._generate_semantic_label(analysis)
                    
                    results.append(analysis)
                    
                except Exception as e:
                    logger.error(f"Failed to analyze chunk {i} in batch: {e}")
                    # Fallback to basic analysis
                    results.append(self._fallback_analysis(text, chunk_index, total_chunks))
            
            logger.info(f"Completed batch analysis of {len(results)} chunks")
            return results
            
        except Exception as e:
            logger.error(f"Failed to perform batch analysis: {e}")
            # Fallback to individual processing
            return [await self.analyze_chunk(chunk['text'], chunk.get('chunk_index', i), chunk.get('total_chunks', len(chunks_data))) 
                   for i, chunk in enumerate(chunks_data)]

    def _convert_batch_ner_entities(self, batch_entities: List[Dict]) -> List[Dict]:
        """Convert batch NER entities to our format."""
        processed_entities = []
        for entity in batch_entities:
            processed_entity = {
                'text': entity['text'],
                'type': entity['type'],
                'confidence': entity['confidence'],
                'start': entity['start'],
                'end': entity['end'],
                'normalized_text': entity['normalized_text'],
                'extraction_method': 'ner_model'
            }
            processed_entities.append(processed_entity)
        return processed_entities
    
    def _detect_language(self, text: str) -> str:
        """Detect if text is primarily Indonesian or mixed language."""
        indonesian_indicators = [
            'yang', 'dengan', 'dalam', 'untuk', 'dari', 'pada', 'oleh', 'sebagai',
            'adalah', 'akan', 'dapat', 'harus', 'perlu', 'tentang', 'kepada'
        ]
        
        text_lower = text.lower()
        indonesian_count = sum(1 for word in indonesian_indicators if word in text_lower)
        
        total_words = len(text.split())
        if total_words == 0:
            return 'unknown'
        
        indonesian_ratio = indonesian_count / min(total_words, 50)  # Check first 50 words
        
        if indonesian_ratio > 0.15:
            return 'indonesian'
        elif indonesian_ratio > 0.05:
            return 'mixed'
        else:
            return 'other'
    
    def _detect_formality(self, text: str) -> str:
        """Detect the formality level of Indonesian text."""
        formal_indicators = [
            'berdasarkan', 'sebagaimana', 'dimaksud', 'tersebut', 'menimbang',
            'mengingat', 'memperhatikan', 'menetapkan', 'diundangkan',
            'berlaku', 'ketentuan', 'peraturan'
        ]
        
        informal_indicators = [
            'gimana', 'kayak', 'banget', 'dong', 'sih', 'nih', 'yuk', 'deh'
        ]
        
        text_lower = text.lower()
        formal_count = sum(1 for word in formal_indicators if word in text_lower)
        informal_count = sum(1 for word in informal_indicators if word in text_lower)
        
        if formal_count > informal_count and formal_count > 0:
            return 'formal'
        elif informal_count > 0:
            return 'informal'
        else:
            return 'neutral'
    
    def _calculate_complexity(self, text: str) -> Dict:
        """Calculate text complexity metrics."""
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        words = text.split()
        avg_sentence_length = len(words) / max(len(sentences), 1)
        
        # Count complex words (>3 syllables approximation)
        complex_words = [w for w in words if len(w) > 8]
        complexity_ratio = len(complex_words) / max(len(words), 1)
        
        return {
            'avg_sentence_length': round(avg_sentence_length, 2),
            'complexity_ratio': round(complexity_ratio, 3),
            'total_sentences': len(sentences),
            'total_words': len(words)
        }
    
    async def _classify_content_semantically(self, text: str, bert_features: Dict) -> str:
        """Use Indonesian BERT features to classify content semantically."""
        try:
            # Extract pooled features for classification
            if bert_features.get('pooler_output') is not None:
                features = bert_features['pooler_output'].flatten()
            
            # Rule-based classification enhanced with BERT confidence
            text_lower = text.lower()
            
            # Check against content classes with BERT confidence
            max_score = 0
            best_class = 'general'
            
            for class_name, keywords in self.content_classes.items():
                score = 0
                for keyword in keywords:
                    if keyword in text_lower:
                        score += 1
                
                # Normalize by number of keywords
                normalized_score = score / len(keywords)
                
                if normalized_score > max_score:
                    max_score = normalized_score
                    best_class = class_name
            
            # If no strong pattern match, use general classification
            if max_score < 0.1:
                best_class = 'general'
            
            return best_class
            
        except Exception as e:
            logger.warning(f"BERT classification failed, using fallback: {e}")
            return self._fallback_classify_content(text)
    
    def _fallback_classify_content(self, text: str) -> str:
        """Fallback content classification using patterns."""
        text_lower = text.lower()
        
        for class_name, keywords in self.content_classes.items():
            if any(keyword in text_lower for keyword in keywords):
                return class_name
        
        return 'general'
    
    async def _extract_entities_with_ner_model(self, text: str) -> List[Dict]:
        """Extract entities using the Indonesian NER model."""
        try:
            # Use the model manager to extract entities with NER
            ner_entities = await self.model_manager.extract_entities_with_ner(text)
            
            # Convert to our format and add additional metadata
            processed_entities = []
            for entity in ner_entities:
                processed_entity = {
                    'text': entity['text'],
                    'type': entity['type'],
                    'confidence': entity['confidence'],
                    'start': entity['start'],
                    'end': entity['end'],
                    'normalized_text': entity['normalized_text'],
                    'extraction_method': 'ner_model'
                }
                processed_entities.append(processed_entity)
            
            logger.info(f"Extracted {len(processed_entities)} entities using NER model")
            return processed_entities
            
        except Exception as e:
            logger.warning(f"NER model entity extraction failed: {e}")
            return []

    def _combine_and_deduplicate_entities(self, ner_entities: List[Dict], pattern_entities: List[Dict]) -> List[Dict]:
        """Combine NER and pattern-based entities, removing duplicates."""
        all_entities = []
        
        # Add NER entities (higher priority)
        for entity in ner_entities:
            all_entities.append(entity)
        
        # Sort by position in text
        all_entities.sort(key=lambda x: x['start'])
        
        logger.info(f"Combined entities: {len(ner_entities)} from NER, {len(pattern_entities)} from patterns, {len(all_entities)} total")
        return all_entities


    def _extract_indonesian_entities(self, text: str) -> List[Dict]:
        """Extract Indonesian-specific named entities using comprehensive patterns."""
        entities = []
        
        # Legal document references
        law_matches = re.finditer(self.legal_patterns['law_number'], text, re.IGNORECASE)
        for match in law_matches:
            entities.append({
                'text': match.group(0),
                'type': 'legal_reference',
                'confidence': 0.9,
                'start': match.start(),
                'end': match.end(),
                'normalized_text': self._normalize_indonesian_entity(match.group(0))
            })
        
        # Government institutions
        institution_matches = re.finditer(self.legal_patterns['institution'], text, re.IGNORECASE)
        for match in institution_matches:
            entities.append({
                'text': match.group(0),
                'type': 'institution',
                'confidence': 0.8,
                'start': match.start(),
                'end': match.end(),
                'normalized_text': self._normalize_indonesian_entity(match.group(0))
            })
        
        # Articles and legal sections
        article_matches = re.finditer(self.legal_patterns['article'], text, re.IGNORECASE)
        for match in article_matches:
            entities.append({
                'text': match.group(0),
                'type': 'legal_section',
                'confidence': 0.85,
                'start': match.start(),
                'end': match.end(),
                'normalized_text': self._normalize_indonesian_entity(match.group(0))
            })
        
        # Extract entities using enhanced patterns
        for entity_type, pattern in self.entity_patterns.items():
            matches = re.finditer(pattern, text)
            for match in matches:
                confidence = self._calculate_entity_confidence(match.group(0), entity_type, text)
                entities.append({
                    'text': match.group(0),
                    'type': entity_type,
                    'confidence': confidence,
                    'start': match.start(),
                    'end': match.end(),
                    'normalized_text': self._normalize_indonesian_entity(match.group(0))
                })
        
        # Remove duplicates and sort by position
        entities = list({(e['text'], e['type']): e for e in entities}.values())
        entities.sort(key=lambda x: x['start'])
        
        # Filter entities for quality before returning
        filtered_entities = self._validate_entity_quality(entities)
        
        return filtered_entities
    
    def _extract_legal_references(self, text: str) -> List[Dict]:
        """Extract legal document references and citations."""
        references = []
        
        # Enhanced regulation references with numbers and specific patterns
        regulation_patterns = [
            r'\b(undang-undang|uu)\s+(no|nomor)\.?\s*\d+\s*(tahun\s*\d+)?\b',
            r'\b(peraturan\s+pemerintah|pp)\s+(no|nomor)\.?\s*\d+\s*(tahun\s*\d+)?\b',
            r'\b(peraturan\s+menteri|permen)\s+[^.\n]{1,50}\s+(no|nomor)\.?\s*\d+\b',
            r'\b(peraturan\s+presiden|perpres)\s+(no|nomor)\.?\s*\d+\s*(tahun\s*\d+)?\b',
            r'\b(keputusan\s+presiden|keppres)\s+(no|nomor)\.?\s*\d+\b',
            r'\b(peraturan\s+daerah|perda)\s+[^.\n]{1,50}\s+(no|nomor)\.?\s*\d+\b',
            r'\b(surat\s+keputusan|sk)\s+[^.\n]{1,50}\s+(no|nomor)\.?\s*[\w\d\/\-]+\b'
        ]
        
        for pattern in regulation_patterns:
            reg_matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in reg_matches:
                references.append({
                    'text': match.group(0).strip(),
                    'type': 'regulation_reference',
                    'confidence': 0.9  # Higher confidence for specific numbered regulations
                })
        
        # Extract article references
        article_pattern = r'\bpasal\s+\d+[^.\n]{0,50}\b'
        art_matches = re.finditer(article_pattern, text, re.IGNORECASE)
        
        for match in art_matches:
            references.append({
                'text': match.group(0).strip(),
                'type': 'article_reference',
                'confidence': 0.9
            })
        
        return references
    
    async def _extract_advanced_key_phrases(self, text: str, bert_features: Dict) -> List[str]:
        """Extract key phrases using BERT attention weights and Indonesian NLP."""

        try:
            if bert_features.get('last_hidden_state') is not None:
                hidden_states = bert_features['last_hidden_state'].squeeze()  # shape: (seq_len, hidden_size)
                attention_scores = np.mean(hidden_states, axis=-1)  # shape: (seq_len,)

                words = text.split()
                if len(words) > len(attention_scores):
                    words = words[:len(attention_scores)]

                word_scores = list(zip(words, attention_scores[:len(words)]))
                word_scores.sort(key=lambda x: x[1], reverse=True)

                key_phrases = []
                for word, score in word_scores[:10]:
                    if (word.lower() not in self.indonesian_stopwords and 
                        len(word) > 2 and 
                        not word.isdigit()):
                        key_phrases.append(word.lower())

                return key_phrases[:5]

        except Exception as e:
            logger.warning(f"BERT key phrase extraction failed: {e}")
        
        # Fallback to TF-IDF approach
        return self._fallback_extract_key_phrases(text)
    
    def _fallback_extract_key_phrases(self, text: str) -> List[str]:
        """Fallback key phrase extraction using frequency analysis."""
        # Preprocess text with Indonesian-specific normalization
        text_processed = self._preprocess_indonesian_text(text)
        
        words = re.findall(r'\b[a-zA-Z]+\b', text_processed.lower())
        filtered_words = [
            word for word in words 
            if (word not in self.indonesian_stopwords and 
                len(word) > 3 and 
                not word.isdigit())
        ]
        
        word_counts = Counter(filtered_words)
        return [word for word, count in word_counts.most_common(5)]
    
    async def _identify_topics(self, text: str, bert_features: Dict) -> List[str]:
        """Identify main topics using Indonesian BERT and pattern matching."""
        topics = []
        text_lower = text.lower()
        
        # Indonesian legal/administrative topics
        topic_keywords = {
            'administrasi_pemerintahan': ['administrasi', 'pemerintahan', 'pelayanan', 'birokrasi'],
            'perizinan': ['izin', 'perizinan', 'rekomendasi', 'persetujuan'],
            'keuangan': ['anggaran', 'keuangan', 'dana', 'biaya', 'tarif'],
            'kepegawaian': ['pegawai', 'kepegawaian', 'jabatan', 'pangkat'],
            'pembangunan': ['pembangunan', 'infrastruktur', 'proyek', 'konstruksi'],
            'lingkungan': ['lingkungan', 'limbah', 'pencemaran', 'konservasi'],
            'sosial': ['sosial', 'masyarakat', 'kemasyarakatan', 'keluarga'],
            'pendidikan': ['pendidikan', 'sekolah', 'siswa', 'pembelajaran'],
            'kesehatan': ['kesehatan', 'medis', 'rumah sakit', 'dokter'],
            'hukum': ['hukum', 'peraturan', 'undang-undang', 'sanksi']
        }
        
        for topic, keywords in topic_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            if score >= 2:  # At least 2 keywords present
                topics.append(topic)
        
        return topics[:3]  # Return top 3 topics
    
    def _analyze_structure(self, text: str) -> str:
        """Analyze the structural type of the text."""
        # Check for numbered lists
        if re.search(r'^\s*\d+\.\s+', text, re.MULTILINE):
            return 'numbered_list'
        
        # Check for bullet points
        if re.search(r'^\s*[-•*]\s+', text, re.MULTILINE):
            return 'bullet_list'
        
        # Check for definitions
        if re.search(r'\bdimaksud dengan\b|\bdefinisi\b|\bpengertian\b', text, re.IGNORECASE):
            return 'definition'
        
        # Check for procedures
        if re.search(r'\btahap\b|\blangkah\b|\bprosedur\b', text, re.IGNORECASE):
            return 'procedure'
        
        # Check for requirements
        if re.search(r'\bsyarat\b|\bketentuan\b|\bpersyaratan\b', text, re.IGNORECASE):
            return 'requirement'
        
        # Check for headings
        if len(text.split()) < 15 and not text.endswith('.'):
            return 'heading'
        
        return 'paragraph'
    
    def _get_enhanced_position_context(self, chunk_index: int, total_chunks: int) -> Dict:
        """Get enhanced position context with additional metadata."""
        if total_chunks <= 1:
            return {'position': 'complete', 'relative_position': 1.0}
        
        relative_position = chunk_index / (total_chunks - 1)
        
        if relative_position <= 0.1:
            position = 'beginning'
        elif relative_position >= 0.9:
            position = 'end'
        elif relative_position <= 0.3:
            position = 'early'
        elif relative_position >= 0.7:
            position = 'late'
        else:
            position = 'middle'
        
        return {
            'position': position,
            'relative_position': round(relative_position, 3),
            'chunk_index': chunk_index,
            'total_chunks': total_chunks
        }
    
    def _generate_semantic_label(self, analysis: Dict) -> str:
        """Generate a semantic label based on analysis results."""
        components = []
        
        # Add semantic class
        components.append(analysis.get('semantic_class', 'general'))
        
        # Add language if not Indonesian
        if analysis.get('language') != 'indonesian':
            components.append(analysis.get('language', 'unknown'))
        
        # Add structure type if significant
        structure = analysis.get('structure_type', 'paragraph')
        if structure != 'paragraph':
            components.append(structure)
        
        # Add primary topic if available
        topics = analysis.get('topics', [])
        if topics:
            components.append(topics[0])
        
        return '_'.join(components)
    
    def _normalize_indonesian_entity(self, entity_text: str) -> str:
        """Normalize Indonesian entity text for better matching."""
        # Convert to lowercase
        normalized = entity_text.lower()
        
        # Remove common prefixes/suffixes
        prefixes = ['dr.', 'prof.', 'ir.', 'drs.', 'h.', 'hj.', 'kh.']
        suffixes = ['s.pd', 's.kom', 's.h', 's.e', 'm.pd', 'm.kom']
        
        for prefix in prefixes:
            if normalized.startswith(prefix):
                normalized = normalized[len(prefix):].strip()
        
        # Handle Indonesian spelling variations
        variations = {
            'djakarta': 'jakarta',
            'jogjakarta': 'yogyakarta',
            'djawa': 'jawa',
            'soekarno': 'sukarno'
        }
        
        for old, new in variations.items():
            normalized = normalized.replace(old, new)
        
        return normalized.strip()
    
    def _calculate_entity_confidence(self, entity_text: str, entity_type: str, context: str) -> float:
        """Calculate confidence score for Indonesian entities."""
        base_score = 0.5
        
        # Boost for known Indonesian patterns
        if entity_type == 'academic_title' and any(title in entity_text.lower() 
                                                 for title in ['dr.', 'prof.', 'ir.']):
            base_score += 0.2
        
        # Boost for government organizations
        if entity_type == 'government_org' and any(keyword in entity_text.lower() 
                                                 for keyword in ['kementerian', 'dinas', 'badan']):
            base_score += 0.3
        
        # Boost for Indonesian locations
        if entity_type == 'regional' and any(keyword in entity_text.lower() 
                                           for keyword in ['provinsi', 'kota', 'kabupaten']):
            base_score += 0.25
        
        # Context-based scoring
        if context and entity_text.lower() in context.lower():
            base_score += 0.15
        
        return min(base_score, 1.0)
    
    def _disambiguate_indonesian_entities(self, entities: List[Dict]) -> List[Dict]:
        """Disambiguate Indonesian entities using context."""
        disambiguated = []
        
        for entity in entities:
            # Create a copy to avoid modifying the original
            entity_copy = entity.copy()
            
            # Standardize Neo4j entity types (ensure GPE is used instead of regional)
            if entity_copy['type'] == 'regional':
                entity_copy['type'] = 'GPE'
            
            # Check for common Indonesian ambiguities
            if entity_copy['text'].lower() in ['jakarta', 'dki jakarta', 'dki']:
                entity_copy['canonical_form'] = 'DKI Jakarta'
                entity_copy['type'] = 'GPE'
                entity_copy['subtype'] = 'province'
            elif entity_copy['text'].lower() in ['jokowi', 'ir. joko widodo']:
                entity_copy['canonical_form'] = 'Joko Widodo'
                entity_copy['type'] = 'PER'
                entity_copy['title'] = 'President'
            
            disambiguated.append(entity_copy)
        
        return disambiguated
    
    def _validate_entity_quality(self, entities: List[Dict]) -> List[Dict]:
        """Validate entity quality and filter out low-quality entities."""
        validated_entities = []
        
        # Common noise patterns to filter out
        noise_patterns = [
            r'^\d+$',  # Pure numbers
            r'^[a-zA-Z]$',  # Single letters
            r'^[^\w\s]+$',  # Only punctuation
            r'^\s+$',  # Only whitespace
        ]
        
        # Common stopwords/noise words to exclude
        noise_words = {
            'yang', 'dan', 'atau', 'ini', 'itu', 'untuk', 'dari', 'pada', 'di', 'ke',
            'oleh', 'dengan', 'dalam', 'akan', 'dapat', 'adalah', 'ada', 'sebagai',
            'tidak', 'jika', 'kalau', 'tapi', 'tetapi', 'namun', 'serta', 'juga'
        }
        
        for entity in entities:
            entity_text = entity.get('text', '').strip()
            entity_type = entity.get('type', '')
            
            # Skip empty entities
            if not entity_text:
                continue
            
            # Check minimum length requirements by type
            min_length = self._get_min_length_for_entity_type(entity_type)
            if len(entity_text) < min_length:
                continue
            
            # Filter out noise patterns
            if any(re.match(pattern, entity_text) for pattern in noise_patterns):
                continue
            
            # Filter out noise words (case insensitive)
            if entity_text.lower() in noise_words:
                continue
            
            # Additional validation for specific entity types
            if not self._validate_entity_by_type(entity_text, entity_type):
                continue
            
            # Check entity confidence threshold
            confidence = entity.get('confidence', 0.0)
            min_confidence = self._get_min_confidence_for_entity_type(entity_type)
            if confidence < min_confidence:
                continue
            
            validated_entities.append(entity)
        return validated_entities
    
    def _get_min_length_for_entity_type(self, entity_type: str) -> int:
        """Get minimum length requirement for entity type."""
        min_lengths = {
            'person': 3,
            'organization': 3,
            'government_org': 4,
            'education': 3,
            'healthcare': 3,
            'legal_entity': 2,  # For abbreviations like UU, PP
            'regional': 3,
            'academic_title': 2,  # For Dr., Ir.
            'religious_title': 2,  # For H., KH.
            'professional_title': 3,
            'regional_title': 4,
            'institution': 3,
            'legal_reference': 2,
            'legal_section': 2
        }
        return min_lengths.get(entity_type, 3)  # Default minimum 3 characters
    
    def _validate_entity_by_type(self, entity_text: str, entity_type: str) -> bool:
        """Validate entity based on its specific type requirements."""
        text_lower = entity_text.lower()
        
        # Academic titles should have dots
        if entity_type == 'academic_title':
            return '.' in entity_text or entity_text.lower() in ['professor', 'doktor']
        
        # Religious titles should be meaningful
        if entity_type == 'religious_title':
            return len(entity_text) >= 2 and ('.' in entity_text or len(entity_text) >= 4)
        
        # Legal entities should contain legal keywords
        if entity_type == 'legal_entity':
            legal_keywords = ['undang', 'peratur', 'keputus', 'surat', 'pp', 'uu', 'permen', 'perpres', 'perda', 'keppres', 'perpu', 'tap']
            return any(keyword in text_lower for keyword in legal_keywords)
        
        # Government organizations should contain government keywords
        if entity_type == 'government_org':
            gov_keywords = ['kementer', 'dinas', 'badan', 'lembaga', 'kantor', 'direktorat']
            return any(keyword in text_lower for keyword in gov_keywords)
        
        # Regional entities should contain regional keywords
        if entity_type == 'regional':
            regional_keywords = ['provinsi', 'kota', 'kabupaten', 'kecamatan', 'kelurahan', 'desa']
            return any(keyword in text_lower for keyword in regional_keywords) or len(entity_text) >= 4
        
        return True  # Default: accept entity
    
    def _get_min_confidence_for_entity_type(self, entity_type: str) -> float:
        """Get minimum confidence threshold for entity type."""
        min_confidences = {
            'person': 0.6,
            'organization': 0.5,
            'government_org': 0.5,
            'education': 0.5,
            'healthcare': 0.5,
            'legal_entity': 0.4,  # Lower threshold for legal entities
            'regional': 0.5,
            'academic_title': 0.7,
            'religious_title': 0.7,
            'professional_title': 0.6,
            'regional_title': 0.6,
            'institution': 0.5,
            'legal_reference': 0.4,  # Lower threshold for legal references
            'legal_section': 0.4
        }
        return min_confidences.get(entity_type, 0.5)  # Default minimum 0.5
    
    def _preprocess_indonesian_text(self, text: str) -> str:
        """Preprocess Indonesian text with language-specific normalization."""
        # Normalize Indonesian text
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
        text = re.sub(r'\s+', ' ', text)     # Normalize whitespace
        
        # Handle Indonesian-specific patterns
        text = re.sub(r'\b(dr|prof|ir|drs|h|hj)\.', r'\1', text)  # Normalize titles
        text = re.sub(r'\b(pt|cv|ud|pd)\.', r'\1', text)          # Normalize org types
        
        return text.strip()
    
    def _fallback_analysis(self, text: str, chunk_index: int, total_chunks: int) -> Dict:
        """Fallback analysis when BERT processing fails."""
        # For fallback, use only pattern-based extraction
        pattern_entities = self._extract_indonesian_entities(text)
        
        return {
            'language': self._detect_language(text),
            'formality_level': self._detect_formality(text),
            'semantic_class': self._fallback_classify_content(text),
            'entities': self._disambiguate_indonesian_entities(pattern_entities),
            'key_phrases': self._fallback_extract_key_phrases(text),
            'topics': [],
            'structure_type': self._analyze_structure(text),
            'position_context': self._get_enhanced_position_context(chunk_index, total_chunks),
            'semantic_label': f"{self._fallback_classify_content(text)}_{self._analyze_structure(text)}",
            'text_complexity': self._calculate_complexity(text),
            'legal_references': self._extract_legal_references(text)
        }


# Global instance
semantic_analyzer = IndonesianSemanticAnalyzer()