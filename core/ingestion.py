import asyncio
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Union

from loguru import logger
from markitdown import MarkItDown

from core.config import settings
from core.database import db_manager
from core.models import model_manager
from core.postgres_storage import postgres_storage
from core.semantic_analyzer import semantic_analyzer
from core.graph_optimizer import optimized_kg_manager
from core.indonesian_kg_manager import indonesian_kg_manager
from core.enhanced_ocr import enhanced_ocr_engine


class ChunkLabeler:
    """Generates semantic labels and metadata for text chunks."""
    
    def __init__(self):
        # Common section headers and patterns
        self.section_patterns = {
            'introduction': r'\b(introduction|intro|overview|background|summary)\b',
            'methodology': r'\b(methodology|method|approach|procedure|process)\b',
            'results': r'\b(results|findings|outcome|analysis|conclusion)\b',
            'discussion': r'\b(discussion|interpretation|implication)\b',
            'conclusion': r'\b(conclusion|summary|final|end)\b',
            'references': r'\b(references|bibliography|citations)\b',
            'appendix': r'\b(appendix|supplement|additional)\b'
        }
        
        # Content type patterns
        self.content_patterns = {
            'list': r'^\s*[-•*]\s+|^\s*\d+\.\s+',
            'table': r'\|.*\||^\s*\w+\s*:\s*\w+',
            'heading': r'^#+\s+|^[A-Z][^.!?]*$',
            'numbered_section': r'^\d+(\.\d+)*\s+[A-Z]'
        }
    
    def extract_key_phrases(self, text: str, max_phrases: int = 5) -> List[str]:
        """Extract key phrases using simple TF-IDF-like approach."""
        # Clean and tokenize
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        
        # Remove common stop words
        stop_words = {
            'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
            'this', 'that', 'these', 'those', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
            'may', 'might', 'must', 'can', 'shall', 'not', 'no', 'yes', 'all', 'any', 'some'
        }
        
        filtered_words = [word for word in words if word not in stop_words and len(word) > 3]
        
        # Count frequency and get top phrases
        word_counts = Counter(filtered_words)
        top_words = [word for word, count in word_counts.most_common(max_phrases)]
        
        return top_words
    
    def detect_content_type(self, text: str) -> str:
        """Detect the type of content in the chunk."""
        text_lower = text.lower().strip()
        
        # Check for specific patterns
        for content_type, pattern in self.content_patterns.items():
            if re.search(pattern, text, re.MULTILINE | re.IGNORECASE):
                return content_type
        
        # Check text characteristics
        if len(text.split('.')) <= 2 and len(text.split()) <= 20:
            return 'heading'
        elif text.count('\n') > 3 and any(char in text for char in ['•', '-', '*']):
            return 'list'
        elif len(text.split()) > 100:
            return 'paragraph'
        else:
            return 'text'
    
    def detect_section_type(self, text: str) -> str:
        """Detect what section this chunk likely belongs to."""
        text_lower = text.lower()
        
        for section, pattern in self.section_patterns.items():
            if re.search(pattern, text_lower):
                return section
        
        return 'content'
    
    def generate_summary(self, text: str, max_length: int = 100, analysis: Dict = None) -> str:
        """Generate a brief summary of the chunk using AI-enhanced analysis."""
        # Use AI-enhanced summary generation if analysis is available
        if analysis:
            ai_summary = self._generate_ai_summary(text, analysis, max_length)
            if ai_summary and self._validate_generated_content(ai_summary, 'summary', text):
                return ai_summary
        
        # Fallback to original logic
        sentences = re.split(r'[.!?]+', text.strip())
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            return text[:max_length] + "..." if len(text) > max_length else text
        
        # Take first sentence or combine short sentences
        summary = sentences[0]
        if len(summary) < 50 and len(sentences) > 1:
            summary += ". " + sentences[1]
        
        # Truncate if too long
        if len(summary) > max_length:
            summary = summary[:max_length-3] + "..."
        
        return summary
    
    def _generate_ai_summary(self, text: str, analysis: Dict, max_length: int = 100) -> str:
        """Generate AI-powered summary using semantic analysis."""
        try:
            # Extract key components from analysis
            semantic_class = analysis.get('semantic_class', 'general')
            entities = analysis.get('entities', [])
            topics = analysis.get('topics', [])
            legal_refs = analysis.get('legal_references', [])
            key_phrases = analysis.get('key_phrases', [])
            structure_type = analysis.get('structure_type', 'paragraph')
            
            # Indonesian summary templates based on semantic class
            summary_generators = {
                'legal_document': self._summarize_legal_content,
                'procedure': self._summarize_procedure_content,
                'definition': self._summarize_definition_content,
                'requirement': self._summarize_requirement_content,
                'reference': self._summarize_reference_content,
                'institution': self._summarize_institution_content
            }
            
            # Use specific summarizer if available
            if semantic_class in summary_generators:
                summary = summary_generators[semantic_class](text, analysis, max_length)
                if summary:
                    return summary
            
            # Generic intelligent summarization
            return self._generate_generic_ai_summary(text, analysis, max_length)
            
        except Exception as e:
            logger.warning(f"AI summary generation failed: {e}")
            return None
    
    def _summarize_legal_content(self, text: str, analysis: Dict, max_length: int) -> str:
        """Summarize legal document content."""
        legal_refs = analysis.get('legal_references', [])
        entities = analysis.get('entities', [])
        key_phrases = analysis.get('key_phrases', [])
        
        summary_parts = []
        
        # Add legal reference info
        if legal_refs:
            ref_text = legal_refs[0].get('text', '')[:30]
            summary_parts.append(f"Mengatur {ref_text}")
        
        # Add key legal entities
        legal_entities = [e for e in entities if e.get('type') in ['legal_reference', 'institution']]
        if legal_entities:
            entity_name = legal_entities[0]['text'][:20]
            summary_parts.append(f"terkait {entity_name}")
        
        # Add main topics
        topics = analysis.get('topics', [])
        if topics:
            topic_name = topics[0].replace('_', ' ')
            summary_parts.append(f"bidang {topic_name}")
        
        if summary_parts:
            summary = " ".join(summary_parts)
            if len(summary) > max_length:
                summary = summary[:max_length-3] + "..."
            return summary
        
        return None
    
    def _summarize_procedure_content(self, text: str, analysis: Dict, max_length: int) -> str:
        """Summarize procedure content."""
        topics = analysis.get('topics', [])
        key_phrases = analysis.get('key_phrases', [])
        
        # Look for numbered steps
        step_pattern = r'(\d+)\.\s*([^.]+)'
        steps = re.findall(step_pattern, text)
        
        if steps:
            first_step = steps[0][1].strip()[:40]
            total_steps = len(steps)
            return f"Prosedur {total_steps} langkah dimulai dengan {first_step}..."
        
        # Fallback to topic-based summary
        if topics:
            topic = topics[0].replace('_', ' ')
            return f"Tata cara untuk {topic}"
        
        return "Prosedur atau langkah-langkah kerja"
    
    def _summarize_definition_content(self, text: str, analysis: Dict, max_length: int) -> str:
        """Summarize definition content."""
        # Extract what's being defined
        definition_patterns = [
            r'dimaksud dengan ([^adalah]+) adalah ([^.]+)',
            r'definisi ([^adalah]+) adalah ([^.]+)',
            r'pengertian ([^adalah]+) adalah ([^.]+)',
            r'([^adalah]+) adalah ([^.]+)'
        ]
        
        text_lower = text.lower()
        original_text = text  # Keep original case
        
        for pattern in definition_patterns:
            match = re.search(pattern, text_lower)
            if match:
                # Get the original text positions
                start_pos = match.start()
                end_pos = match.end()
                original_match = original_text[start_pos:end_pos]
                
                # Extract term and its definition
                groups = re.search(pattern, original_match, re.IGNORECASE)
                if groups:
                    term = groups.group(1).strip()
                    definition = groups.group(2).strip()[:50]
                    return f"Definisi {term}: {definition}..."
        
        return "Definisi atau pengertian istilah"
    
    def _summarize_requirement_content(self, text: str, analysis: Dict, max_length: int) -> str:
        """Summarize requirement content."""
        topics = analysis.get('topics', [])
        key_phrases = analysis.get('key_phrases', [])
        
        # Look for enumerated requirements
        req_pattern = r'([a-z]\.|[0-9]+\.|\*|\-)\s*([^.]+)'
        requirements = re.findall(req_pattern, text, re.IGNORECASE)
        
        if requirements:
            total_reqs = len(requirements)
            first_req = requirements[0][1].strip()[:40]
            return f"Persyaratan {total_reqs} poin termasuk {first_req}..."
        
        # Topic-based summary
        if topics:
            topic = topics[0].replace('_', ' ')
            return f"Syarat dan ketentuan untuk {topic}"
        
        return "Persyaratan atau ketentuan yang harus dipenuhi"
    
    def _summarize_reference_content(self, text: str, analysis: Dict, max_length: int) -> str:
        """Summarize reference content."""
        legal_refs = analysis.get('legal_references', [])
        
        if legal_refs:
            ref_text = legal_refs[0].get('text', '')
            if ref_text:
                return f"Rujukan pada {ref_text[:50]}..."
        
        return "Rujukan atau referensi hukum"
    
    def _summarize_institution_content(self, text: str, analysis: Dict, max_length: int) -> str:
        """Summarize institution content."""
        entities = analysis.get('entities', [])
        topics = analysis.get('topics', [])
        
        # Extract institution entities
        institutions = [e for e in entities if e.get('type') == 'institution']
        if institutions:
            inst_name = institutions[0]['text']
            if topics:
                topic = topics[0].replace('_', ' ')
                return f"Informasi tentang {inst_name} terkait {topic}"
            else:
                return f"Informasi tentang {inst_name}"
        
        return "Informasi kelembagaan"
    
    def _generate_generic_ai_summary(self, text: str, analysis: Dict, max_length: int) -> str:
        """Generate generic intelligent summary using analysis."""
        key_phrases = analysis.get('key_phrases', [])
        topics = analysis.get('topics', [])
        entities = analysis.get('entities', [])
        structure_type = analysis.get('structure_type', 'paragraph')
        
        # Build summary components
        summary_parts = []
        
        # Add primary topic
        if topics:
            topic = topics[0].replace('_', ' ')
            summary_parts.append(f"Membahas {topic}")
        
        # Add key entities
        if entities:
            entity_names = [e['text'] for e in entities[:2]]
            if entity_names:
                summary_parts.append(f"melibatkan {', '.join(entity_names)}")
        
        # Add key phrases context
        if key_phrases and len(key_phrases) > 1:
            phrase = key_phrases[1].replace('_', ' ')  # Use second phrase for context
            summary_parts.append(f"dengan fokus pada {phrase}")
        
        # Add structure context
        if structure_type in ['numbered_list', 'bullet_list']:
            summary_parts.append("dalam bentuk daftar poin")
        elif structure_type == 'procedure':
            summary_parts.append("dalam bentuk prosedur")
        
        if summary_parts:
            summary = " ".join(summary_parts)
            if len(summary) > max_length:
                summary = summary[:max_length-3] + "..."
            return summary
        
        # Fallback: use first meaningful sentence
        sentences = re.split(r'[.!?]+', text.strip())
        if sentences:
            first_sentence = sentences[0].strip()
            if len(first_sentence) > 10:
                if len(first_sentence) > max_length:
                    return first_sentence[:max_length-3] + "..."
                return first_sentence
        
        return "Konten informasi"
    
    def _validate_generated_content(self, content: str, content_type: str, original_text: str) -> bool:
        """Validate quality of AI-generated titles and summaries."""
        if not content or not content.strip():
            return False
        
        # Basic length validation
        if content_type == 'title':
            # Title should be between 5-80 characters
            if len(content) < 5 or len(content) > 80:
                return False
            # Title shouldn't end with punctuation except colon
            if content.endswith(('.', '!', '?')):
                return False
        elif content_type == 'summary':
            # Summary should be between 20-200 characters
            if len(content) < 20 or len(content) > 200:
                return False
        
        # Content quality checks
        # Should not be repetitive (same word repeated > 3 times)
        words = content.lower().split()
        word_counts = {}
        for word in words:
            if len(word) > 3:  # Only check meaningful words
                word_counts[word] = word_counts.get(word, 0) + 1
                if word_counts[word] > 3:
                    return False
        
        # Should not be too generic
        generic_patterns = [
            r'^(content|konten|informasi|data|text|teks)$',
            r'^(general|umum|biasa)$',
            r'^(document|dokumen)$'
        ]
        
        content_lower = content.lower().strip()
        for pattern in generic_patterns:
            if re.match(pattern, content_lower):
                return False
        
        # Should contain some meaningful content from original text
        # Extract meaningful words from both
        original_words = set(re.findall(r'\b[a-zA-Z]{4,}\b', original_text.lower()))
        content_words = set(re.findall(r'\b[a-zA-Z]{4,}\b', content.lower()))
        
        # At least one meaningful word should be shared (relevance check)
        if original_words and content_words:
            shared_words = original_words.intersection(content_words)
            if not shared_words:
                return False
        
        return True
    
    def generate_title(self, text: str, key_phrases: List[str], section_type: str, analysis: Dict = None) -> str:
        """Generate a human-readable title for the chunk using AI-enhanced analysis."""
        # Try to find a natural heading first
        lines = text.split('\n')
        first_line = lines[0].strip()
        
        # If first line looks like a heading
        if (len(first_line) < 80 and 
            len(first_line.split()) <= 10 and 
            not first_line.endswith('.') and
            first_line[0].isupper()):
            return first_line
        
        # Use AI-enhanced title generation if analysis is available
        if analysis:
            ai_title = self._generate_ai_title(text, analysis)
            if ai_title and self._validate_generated_content(ai_title, 'title', text):
                return ai_title
        
        # Fallback to original logic
        if key_phrases:
            main_topic = key_phrases[0].replace('_', ' ').title()
            if section_type != 'content':
                return f"{section_type.title()}: {main_topic}"
            else:
                return main_topic
        
        # Final fallback to section type
        return section_type.title() if section_type != 'content' else "Content"
    
    def _generate_ai_title(self, text: str, analysis: Dict) -> str:
        """Generate AI-powered title using semantic analysis."""
        try:
            # Extract key components from analysis
            semantic_class = analysis.get('semantic_class', 'general')
            entities = analysis.get('entities', [])
            topics = analysis.get('topics', [])
            legal_refs = analysis.get('legal_references', [])
            structure_type = analysis.get('structure_type', 'paragraph')
            key_phrases = analysis.get('key_phrases', [])
            
            # Indonesian title templates based on semantic class
            title_templates = {
                'legal_document': self._generate_legal_title,
                'procedure': self._generate_procedure_title,
                'definition': self._generate_definition_title,
                'requirement': self._generate_requirement_title,
                'reference': self._generate_reference_title,
                'institution': self._generate_institution_title
            }
            
            # Use specific template if available
            if semantic_class in title_templates:
                title = title_templates[semantic_class](text, analysis)
                if title:
                    return title
            
            # Generic intelligent title generation
            return self._generate_generic_ai_title(text, analysis)
            
        except Exception as e:
            logger.warning(f"AI title generation failed: {e}")
            return None
    
    def _generate_legal_title(self, text: str, analysis: Dict) -> str:
        """Generate title for legal documents."""
        legal_refs = analysis.get('legal_references', [])
        entities = analysis.get('entities', [])
        
        # Extract regulation references
        for ref in legal_refs:
            if ref.get('type') == 'regulation_reference':
                return f"Peraturan: {ref['text'][:50]}..."
        
        # Extract legal entities
        legal_entities = [e for e in entities if e.get('type') in ['legal_reference', 'institution']]
        if legal_entities:
            return f"Ketentuan {legal_entities[0]['text'].title()}"
        
        return "Dokumen Hukum"
    
    def _generate_procedure_title(self, text: str, analysis: Dict) -> str:
        """Generate title for procedure documents."""
        key_phrases = analysis.get('key_phrases', [])
        topics = analysis.get('topics', [])
        
        # Look for procedure-specific keywords
        procedure_keywords = ['tata cara', 'prosedur', 'langkah', 'tahap', 'mekanisme']
        text_lower = text.lower()
        
        for keyword in procedure_keywords:
            if keyword in text_lower:
                # Extract what follows the keyword
                pattern = rf'{keyword}\s+([^.]+)'
                match = re.search(pattern, text_lower)
                if match:
                    return f"Tata Cara {match.group(1)[:30].title()}"
        
        # Use topics if available
        if topics:
            return f"Prosedur {topics[0].replace('_', ' ').title()}"
        
        return "Prosedur"
    
    def _generate_definition_title(self, text: str, analysis: Dict) -> str:
        """Generate title for definition content."""
        # Look for definition patterns
        definition_patterns = [
            r'dimaksud dengan ([^adalah]+)',
            r'definisi ([^adalah]+)',
            r'pengertian ([^adalah]+)',
            r'([^adalah]+) adalah'
        ]
        
        text_lower = text.lower()
        for pattern in definition_patterns:
            match = re.search(pattern, text_lower)
            if match:
                term = match.group(1).strip()[:30]
                return f"Definisi {term.title()}"
        
        return "Definisi"
    
    def _generate_requirement_title(self, text: str, analysis: Dict) -> str:
        """Generate title for requirement content."""
        topics = analysis.get('topics', [])
        key_phrases = analysis.get('key_phrases', [])
        
        # Look for requirement-specific keywords
        requirement_keywords = ['syarat', 'ketentuan', 'persyaratan', 'kriteria']
        text_lower = text.lower()
        
        for keyword in requirement_keywords:
            if keyword in text_lower:
                if topics:
                    return f"Syarat {topics[0].replace('_', ' ').title()}"
                elif key_phrases:
                    return f"Ketentuan {key_phrases[0].replace('_', ' ').title()}"
        
        return "Persyaratan"
    
    def _generate_reference_title(self, text: str, analysis: Dict) -> str:
        """Generate title for reference content."""
        legal_refs = analysis.get('legal_references', [])
        
        # Extract article references
        for ref in legal_refs:
            if ref.get('type') == 'article_reference':
                return f"Rujukan: {ref['text']}"
        
        return "Rujukan"
    
    def _generate_institution_title(self, text: str, analysis: Dict) -> str:
        """Generate title for institution content."""
        entities = analysis.get('entities', [])
        topics = analysis.get('topics', [])
        
        # Extract institution entities
        institutions = [e for e in entities if e.get('type') == 'institution']
        if institutions:
            inst_name = institutions[0]['text'].title()
            if topics:
                return f"{inst_name}: {topics[0].replace('_', ' ').title()}"
            else:
                return inst_name
        
        return "Lembaga"
    
    def _generate_generic_ai_title(self, text: str, analysis: Dict) -> str:
        """Generate generic intelligent title using analysis."""
        key_phrases = analysis.get('key_phrases', [])
        topics = analysis.get('topics', [])
        structure_type = analysis.get('structure_type', 'paragraph')
        
        # Combine most relevant elements
        title_parts = []
        
        # Add primary topic
        if topics:
            title_parts.append(topics[0].replace('_', ' ').title())
        
        # Add key phrase if different from topic
        if key_phrases and (not topics or key_phrases[0] not in topics[0]):
            title_parts.append(key_phrases[0].replace('_', ' ').title())
        
        # Add structure context if significant
        if structure_type in ['numbered_list', 'bullet_list', 'procedure']:
            title_parts.append(f"({structure_type.replace('_', ' ').title()})")
        
        if title_parts:
            return " - ".join(title_parts[:2])  # Keep it concise
        
        # Last resort: use first meaningful phrase from text
        sentences = re.split(r'[.!?]+', text.strip())
        if sentences:
            first_sentence = sentences[0].strip()
            if len(first_sentence) > 10:
                return first_sentence[:50] + ("..." if len(first_sentence) > 50 else "")
        
        return "Konten"
    
    def get_position_context(self, chunk_index: int, total_chunks: int) -> str:
        """Determine position context within document."""
        if total_chunks <= 1:
            return 'complete'
        
        position_ratio = chunk_index / (total_chunks - 1)
        
        if position_ratio <= 0.1:
            return 'beginning'
        elif position_ratio >= 0.9:
            return 'end'
        elif position_ratio <= 0.3:
            return 'early'
        elif position_ratio >= 0.7:
            return 'late'
        else:
            return 'middle'
    
    async def label_chunk(self, chunk: Dict, chunk_index: int, total_chunks: int) -> Dict:
        """Generate comprehensive labels and metadata for a chunk using advanced semantic analysis."""
        text = chunk['content']
        
        try:
            # Use advanced semantic analyzer with Indonesian BERT
            analysis = await semantic_analyzer.analyze_chunk(text, chunk_index, total_chunks)
            
            # Enhanced labeling with semantic analysis
            enhanced_title = self.generate_title(text, analysis.get('key_phrases', []), analysis.get('semantic_class', 'general'), analysis)
            enhanced_summary = self.generate_summary(text, analysis=analysis)
            
            # Enhanced chunk with semantic analysis
            enhanced_chunk = chunk.copy()
            enhanced_chunk.update({
                'title': enhanced_title,
                'summary': enhanced_summary,
                'key_phrases': analysis.get('key_phrases', [])[:3],
                'content_type': analysis.get('structure_type', 'paragraph'),
                'section_type': analysis.get('semantic_class', 'general'),
                'semantic_label': analysis.get('semantic_label', 'general_paragraph'),
                'position_context': analysis.get('position_context', {}).get('position', 'middle'),
                'analysis': analysis  # Store full analysis for Neo4j optimization
            })
            
            return enhanced_chunk
            
        except Exception as e:
            logger.warning(f"Advanced semantic analysis failed for chunk {chunk_index}, using fallback: {e}")
            return self._fallback_label_chunk(chunk, chunk_index, total_chunks)
    
    def _fallback_label_chunk(self, chunk: Dict, chunk_index: int, total_chunks: int) -> Dict:
        """Fallback to basic labeling when advanced analysis fails."""
        text = chunk['content']
        
        # Extract features using basic methods
        key_phrases = self.extract_key_phrases(text)
        content_type = self.detect_content_type(text)
        section_type = self.detect_section_type(text)
        summary = self.generate_summary(text)
        title = self.generate_title(text, key_phrases, section_type)
        position_context = self.get_position_context(chunk_index, total_chunks)
        
        # Create semantic label
        semantic_label = f"{section_type}_{content_type}"
        if key_phrases:
            semantic_label += f"_{key_phrases[0]}"
        
        # Add new properties to chunk
        enhanced_chunk = chunk.copy()
        enhanced_chunk.update({
            'title': title,
            'summary': summary,
            'key_phrases': key_phrases[:3],  # Limit to top 3
            'content_type': content_type,
            'section_type': section_type,
            'semantic_label': semantic_label,
            'position_context': position_context,
            'analysis': {  # Minimal analysis for consistency
                'semantic_class': section_type,
                'language': 'unknown',
                'structure_type': content_type,
                'entities': [],
                'topics': key_phrases[:3] if key_phrases else ['general'],  # Use key_phrases as topics
                'key_phrases': key_phrases
            }
        })
        
        return enhanced_chunk


class DocumentProcessor:
    def __init__(self):
        self.markitdown = MarkItDown()  # Keep as fallback
        self.enhanced_ocr = enhanced_ocr_engine
        self.chunk_size = settings.chunk_size
        self.chunk_overlap = settings.chunk_overlap
        self.labeler = ChunkLabeler()

    async def process_document(self, file_path: Union[str, Path]) -> Dict:
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        logger.info(f"Processing document with enhanced OCR: {file_path}")
        
        try:
            # Use page-by-page processing for PDFs, fallback for other files
            if file_path.suffix.lower() == '.pdf':
                logger.info(f"Using page-by-page PDF processing for {file_path.name}")
                ocr_result = self.enhanced_ocr.extract_text_pdf_pages(file_path)
            else:
                # Use standard extraction for non-PDF files
                ocr_result = self.enhanced_ocr.extract_text_with_fallback(file_path)
            text_content = ocr_result.text
            
            chunks = await self._create_chunks(text_content)
            
            document_data = {
                "file_path": str(file_path),
                "file_name": file_path.name,
                "file_size": file_path.stat().st_size,
                "content": text_content,
                "chunks": chunks,
                "chunk_count": len(chunks),
                "metadata": {
                    "file_type": file_path.suffix,
                    "processed_with": ocr_result.engine,
                    "ocr_confidence": ocr_result.confidence,
                    "ocr_metadata": ocr_result.metadata
                }
            }
            
            logger.info(f"Document processed successfully with {ocr_result.engine}, confidence: {ocr_result.confidence:.3f}")
            return document_data
            
        except Exception as e:
            logger.error(f"Failed to process document {file_path}: {e}")
            # Try fallback to original MarkItDown if enhanced OCR fails
            try:
                logger.info(f"Attempting fallback to MarkItDown for {file_path}")
                result = self.markitdown.convert(str(file_path))
                text_content = result.text_content
                
                chunks = await self._create_chunks(text_content)
                
                document_data = {
                    "file_path": str(file_path),
                    "file_name": file_path.name,
                    "file_size": file_path.stat().st_size,
                    "content": text_content,
                    "chunks": chunks,
                    "chunk_count": len(chunks),
                    "metadata": {
                        "file_type": file_path.suffix,
                        "processed_with": "markitdown_fallback",
                        "ocr_confidence": 0.5,  # Default confidence for fallback
                        "fallback_reason": str(e)
                    }
                }
                
                logger.info(f"Document processed with MarkItDown fallback: {file_path}")
                return document_data
                
            except Exception as fallback_error:
                logger.error(f"Both enhanced OCR and MarkItDown fallback failed for {file_path}: {fallback_error}")
                raise

    async def _create_chunks(self, text: str) -> List[Dict]:
        chunks = []
        lines = text.split('\n')
        
        # For Indonesian regulation documents, prioritize the beginning (title, intro, main provisions)
        important_content_lines = min(5000, len(lines))  # First 5000 lines likely contain key content
        
        # Create a high-priority chunk from the document beginning (title, purpose, etc.)
        beginning_lines = lines[:min(100, len(lines))]  # First 100 lines for title/intro
        beginning_text = '\n'.join(line.strip() for line in beginning_lines if line.strip())
        
        if beginning_text and len(beginning_text) > 100:
            chunks.append({
                "content": beginning_text,
                "start_index": 0,
                "end_index": len(beginning_text),
                "word_count": len(beginning_text.split()),
                "chunk_index": 0,
                "metadata": {
                    "section_type": "document_header",
                    "priority": "high",
                    "contains_title": True
                }
            })
            logger.info(f"Created high-priority header chunk with {len(beginning_text)} characters")
        
        # Process the rest with regular word-based chunking
        words = text.split()
        
        if len(words) <= self.chunk_size and len(chunks) == 0:
            chunk_text = text
            if settings.enable_chunk_splitting and len(chunk_text) > settings.max_chunk_characters:
                sub_chunks = self._split_oversized_chunk(chunk_text, 0)
                chunks.extend(sub_chunks)
            else:
                chunks.append({
                    "content": chunk_text,
                    "start_index": 0,
                    "end_index": len(text),
                    "word_count": len(words),
                    "chunk_index": 0,
                    "metadata": {"section_type": "full_document"}
                })
            return chunks
        
        chunk_index = len(chunks)  # Continue from header chunks
        for i in range(0, len(words), self.chunk_size - self.chunk_overlap):
            chunk_words = words[i:i + self.chunk_size]
            chunk_text = " ".join(chunk_words)
            
            start_char = len(" ".join(words[:i])) + (1 if i > 0 else 0)
            end_char = start_char + len(chunk_text)
            
            # Determine if this chunk is from important early content
            is_early_content = i < important_content_lines * 10  # Rough estimate for word position
            priority = "medium" if is_early_content else "low"
            
            # Validate chunk size and split if necessary
            if settings.enable_chunk_splitting and len(chunk_text) > settings.max_chunk_characters:
                sub_chunks = self._split_oversized_chunk(chunk_text, chunk_index)
                chunks.extend(sub_chunks)
                chunk_index += len(sub_chunks)
            else:
                chunks.append({
                    "content": chunk_text,
                    "start_index": start_char,
                    "end_index": end_char,
                    "word_count": len(chunk_words),
                    "chunk_index": chunk_index,
                    "metadata": {
                        "section_type": "body_content",
                        "priority": priority,
                        "contains_title": False
                    }
                })
                chunk_index += 1
            
            if i + self.chunk_size >= len(words):
                break
        
        # Apply semantic labeling to all chunks asynchronously
        labeled_chunks = []
        total_chunks = len(chunks)
        for i, chunk in enumerate(chunks):
            labeled_chunk = await self.labeler.label_chunk(chunk, i, total_chunks)
            
            # Validate that essential labeling properties were created
            if not labeled_chunk.get('title') or not labeled_chunk.get('summary'):
                logger.warning(f"Chunk {i} labeling incomplete - title: '{labeled_chunk.get('title')}', summary: '{labeled_chunk.get('summary')}'")
            
            labeled_chunks.append(labeled_chunk)
        
        logger.info(f"Completed labeling {len(labeled_chunks)} chunks")
        return labeled_chunks

    def _split_oversized_chunk(self, chunk_text: str, base_chunk_index: int) -> List[Dict]:
        """Split an oversized chunk into smaller chunks that fit within character limits."""
        sub_chunks = []
        max_chars = settings.max_chunk_characters
        
        # Try to split on sentence boundaries first
        sentences = chunk_text.split('. ')
        if len(sentences) > 1:
            current_chunk = ""
            sub_chunk_index = 0
            
            for sentence in sentences:
                # Add sentence separator back except for last sentence
                sentence_with_period = sentence + ('. ' if sentence != sentences[-1] else '')
                
                # Check if adding this sentence would exceed the limit
                if len(current_chunk + sentence_with_period) > max_chars:
                    if current_chunk:  # Save current chunk if not empty
                        words = current_chunk.strip().split()
                        sub_chunks.append({
                            "content": current_chunk.strip(),
                            "start_index": 0,  # Will be recalculated if needed
                            "end_index": len(current_chunk.strip()),
                            "word_count": len(words),
                            "chunk_index": base_chunk_index + sub_chunk_index
                        })
                        sub_chunk_index += 1
                        
                        # Check if the sentence itself is too large
                        if len(sentence_with_period) > max_chars:
                            word_chunks = self._split_by_words(sentence_with_period, base_chunk_index + sub_chunk_index)
                            sub_chunks.extend(word_chunks)
                            sub_chunk_index += len(word_chunks)
                            current_chunk = ""
                        else:
                            current_chunk = sentence_with_period
                    else:
                        # Single sentence is too long, split by words
                        word_chunks = self._split_by_words(sentence_with_period, base_chunk_index + sub_chunk_index)
                        sub_chunks.extend(word_chunks)
                        sub_chunk_index += len(word_chunks)
                else:
                    current_chunk += sentence_with_period
            
            # Add remaining chunk with safety validation
            if current_chunk.strip():
                # Safety check: if remaining chunk is still too large, split by words
                if len(current_chunk.strip()) > max_chars:
                    word_chunks = self._split_by_words(current_chunk.strip(), base_chunk_index + sub_chunk_index)
                    sub_chunks.extend(word_chunks)
                else:
                    words = current_chunk.strip().split()
                    sub_chunks.append({
                        "content": current_chunk.strip(),
                        "start_index": 0,
                        "end_index": len(current_chunk.strip()),
                        "word_count": len(words),
                        "chunk_index": base_chunk_index + sub_chunk_index
                    })
        else:
            # No sentence boundaries, split by words
            sub_chunks = self._split_by_words(chunk_text, base_chunk_index)
        
        return sub_chunks

    def _split_by_words(self, text: str, base_chunk_index: int) -> List[Dict]:
        """Split text by words when sentence splitting isn't sufficient."""
        sub_chunks = []
        words = text.split()
        max_chars = settings.max_chunk_characters
        
        current_chunk = ""
        current_words = []
        sub_chunk_index = 0
        
        for word in words:
            test_chunk = current_chunk + (" " if current_chunk else "") + word
            
            if len(test_chunk) > max_chars and current_chunk:
                # Save current chunk
                sub_chunks.append({
                    "content": current_chunk,
                    "start_index": 0,
                    "end_index": len(current_chunk),
                    "word_count": len(current_words),
                    "chunk_index": base_chunk_index + sub_chunk_index
                })
                sub_chunk_index += 1
                current_chunk = word
                current_words = [word]
            else:
                if current_chunk:
                    current_chunk += " " + word
                else:
                    current_chunk = word
                current_words.append(word)
        
        # Add remaining chunk
        if current_chunk:
            sub_chunks.append({
                "content": current_chunk,
                "start_index": 0,
                "end_index": len(current_chunk),
                "word_count": len(current_words),
                "chunk_index": base_chunk_index + sub_chunk_index
            })
        
        return sub_chunks


class KnowledgeGraphManager:
    def __init__(self):
        self.batch_size = settings.kg_batch_size

    async def create_document_node(self, document_data: Dict) -> str:
        query = """
        CREATE (d:Document {
            file_path: $file_path,
            file_name: $file_name,
            file_size: $file_size,
            chunk_count: $chunk_count,
            file_type: $file_type,
            processed_with: $processed_with,
            created_at: datetime()
        })
        RETURN id(d) as node_id
        """
        
        metadata = document_data.get("metadata", {})
        async with db_manager.get_neo4j_session() as session:
            result = await session.run(query, {
                "file_path": document_data["file_path"],
                "file_name": document_data["file_name"],
                "file_size": document_data["file_size"],
                "chunk_count": document_data["chunk_count"],
                "file_type": metadata.get("file_type", "unknown"),
                "processed_with": metadata.get("processed_with", "unknown")
            })
            
            record = await result.single()
            return str(record["node_id"])

    async def create_chunk_nodes(self, document_node_id: str, chunks: List[Dict], embeddings: List[List[float]]):
        query = """
        MATCH (d:Document) WHERE id(d) = $doc_id
        CREATE (c:Chunk {
            content: $content,
            start_index: $start_index,
            end_index: $end_index,
            word_count: $word_count,
            embedding: $embedding,
            title: $title,
            summary: $summary,
            key_phrases: $key_phrases,
            content_type: $content_type,
            section_type: $section_type,
            semantic_label: $semantic_label,
            position_context: $position_context,
            created_at: datetime()
        })
        CREATE (d)-[:HAS_CHUNK]->(c)
        """
        
        async with db_manager.get_neo4j_session() as session:
            for i, chunk in enumerate(chunks):
                await session.run(query, {
                    "doc_id": int(document_node_id),
                    "content": chunk["content"],
                    "start_index": chunk["start_index"],
                    "end_index": chunk["end_index"],
                    "word_count": chunk["word_count"],
                    "embedding": embeddings[i],
                    "title": chunk.get("title", "Untitled"),
                    "summary": chunk.get("summary", ""),
                    "key_phrases": chunk.get("key_phrases", []),
                    "content_type": chunk.get("content_type", "text"),
                    "section_type": chunk.get("section_type", "content"),
                    "semantic_label": chunk.get("semantic_label", "content_text"),
                    "position_context": chunk.get("position_context", "middle")
                })

    async def find_similar_chunks(self, query_embedding: List[float], limit: int = 10) -> List[Dict]:
        query = """
        MATCH (c:Chunk)
        WITH c, 
             reduce(dot = 0.0, i in range(0, size(c.embedding)-1) | 
                 dot + c.embedding[i] * $query_embedding[i]) as dot_product,
             sqrt(reduce(norm_c = 0.0, val in c.embedding | norm_c + val * val)) as norm_c,
             sqrt(reduce(norm_q = 0.0, val in $query_embedding | norm_q + val * val)) as norm_q
        WITH c, dot_product / (norm_c * norm_q) as similarity
        WHERE similarity > $threshold
        RETURN c.content as content, 
               c.semantic_label as semantic_label,
               similarity
        ORDER BY similarity DESC
        LIMIT $limit
        """
        
        async with db_manager.get_neo4j_session() as session:
            result = await session.run(query, {
                "query_embedding": query_embedding,
                "threshold": settings.similarity_threshold,
                "limit": limit
            })
            
            return [dict(record) async for record in result]


class IngestionPipeline:
    def __init__(self):
        self.processor = DocumentProcessor()
        self.kg_manager = optimized_kg_manager  # Use optimized knowledge graph manager

    def _is_indonesian_content(self, content: str) -> bool:
        """Detect if content is primarily Indonesian language."""
        indonesian_indicators = [
            'yang', 'dengan', 'dalam', 'untuk', 'dari', 'pada', 'oleh', 'sebagai',
            'adalah', 'akan', 'dapat', 'harus', 'perlu', 'tentang', 'kepada',
            'kementerian', 'dinas', 'badan', 'peraturan', 'undang-undang'
        ]
        
        content_lower = content.lower()
        indonesian_count = sum(1 for word in indonesian_indicators if word in content_lower)
        
        # Check for Indonesian administrative patterns
        admin_patterns = ['nomor', 'tahun', 'pasal', 'ayat', 'huruf', 'angka']
        admin_count = sum(1 for pattern in admin_patterns if pattern in content_lower)
        
        total_words = len(content.split())
        if total_words == 0:
            return False
        
        # Consider Indonesian if > 15% indicators or has admin patterns
        indonesian_ratio = (indonesian_count + admin_count) / min(total_words, 100)
        return indonesian_ratio > 0.15

    async def ingest_document(self, file_path: Union[str, Path]) -> Dict:
        try:
            file_path = Path(file_path)
            
            # Fast duplicate detection BEFORE expensive processing
            if settings.duplicate_detection_enabled:
                # Calculate file hash immediately (much faster than OCR)
                content_hash = postgres_storage.calculate_file_hash(file_path)
                
                should_skip = False
                existing_doc = None
                duplicate_reason = None
                
                if settings.duplicate_detection_method in ["content_hash", "both"]:
                    # First, check if document was successfully processed (has chunks)
                    existing_doc = await postgres_storage.check_successfully_processed_document_by_hash(content_hash)
                    
                    if existing_doc and settings.duplicate_detection_skip_duplicates:
                        # Successfully processed document exists - skip processing
                        should_skip = True
                        duplicate_reason = f"identical content already successfully processed (hash: {content_hash[:8]}...)"
                    else:
                        # No successfully processed document found
                        # Check if there's a failed/incomplete document with same hash
                        failed_doc = await postgres_storage.check_document_exists_by_hash(content_hash)
                        if failed_doc and failed_doc.get('chunk_count', 0) == 0:
                            if settings.duplicate_detection_cleanup_failed:
                                logger.info(f"Found incomplete document with same content hash, cleaning up: {failed_doc['file_path']}")
                                # Clean up the incomplete document and continue processing
                                await postgres_storage.delete_document(failed_doc['id'])
                            else:
                                logger.info(f"Found incomplete document with same content hash, keeping it: {failed_doc['file_path']}")
                        # Continue with processing (should_skip remains False)
                
                if should_skip:
                    log_msg = f"Skipping duplicate document: {file_path} ({duplicate_reason})"
                    if settings.duplicate_detection_log_level == "debug":
                        logger.debug(log_msg)
                    else:
                        logger.info(log_msg)
                    
                    return {
                        "status": "skipped_duplicate",
                        "message": f"Document content already successfully processed as {existing_doc['file_name']} with {existing_doc['chunk_count']} chunks",
                        "existing_document": existing_doc,
                        "file_path": str(file_path),
                        "content_hash": content_hash,
                        "duplicate_reason": duplicate_reason
                    }
            
            # Only process document if no duplicate found (expensive operation)
            document_data = await self.processor.process_document(file_path)
            
            # Update document_data with the calculated hash (if duplicate detection enabled)
            if settings.duplicate_detection_enabled:
                document_data["content_hash"] = content_hash
            else:
                # Calculate hash from processed content for storage (fallback method)
                document_data["content_hash"] = postgres_storage.calculate_content_hash(document_data["content"])
            
            chunk_texts = [chunk["content"] for chunk in document_data["chunks"]]
            embeddings = await model_manager.get_embeddings(chunk_texts)
            
            # Store in PostgreSQL first
            pg_document_id = await postgres_storage.store_document(
                file_path=document_data["file_path"],
                file_name=document_data["file_name"],
                file_size=document_data["file_size"],
                content_hash=content_hash,
                metadata=document_data.get("metadata", {})
            )
            
            # Prepare chunks data for PostgreSQL storage
            pg_chunks_data = []
            for i, (chunk, embedding) in enumerate(zip(document_data["chunks"], embeddings)):
                pg_chunk_data = {
                    "content": chunk["content"],
                    "start_index": chunk.get("start_index", 0),
                    "end_index": chunk.get("end_index", len(chunk["content"])),
                    "word_count": chunk.get("word_count", len(chunk["content"].split())),
                    "chunk_index": i,
                    "embedding": embedding,
                    "metadata": {
                        "title": chunk.get("title", ""),
                        "summary": chunk.get("summary", ""),
                        "key_phrases": chunk.get("key_phrases", []),
                        "content_type": chunk.get("content_type", "text"),
                        "section_type": chunk.get("section_type", "content"),
                        "semantic_label": chunk.get("semantic_label", "content_text"),
                        "position_context": chunk.get("position_context", "middle")
                    }
                }
                pg_chunks_data.append(pg_chunk_data)
            
            # Store chunks with embeddings in PostgreSQL
            pg_chunk_ids = await postgres_storage.store_chunks_with_embeddings(
                document_id=pg_document_id,
                chunks_data=pg_chunks_data
            )
            
            # Enhanced processing for Indonesian documents
            if self._is_indonesian_content(document_data["content"]):
                logger.info(f"Detected Indonesian content, using enhanced processing: {file_path}")
                
                # Use Indonesian KG manager for Indonesian documents
                # Pass complete chunk data including all labeling properties
                chunks_with_embeddings = []
                for chunk, emb in zip(document_data["chunks"], embeddings):
                    chunk_with_embedding = chunk.copy()  # Preserve all chunk properties (title, summary, key_phrases, etc.)
                    chunk_with_embedding["embedding"] = emb
                    chunks_with_embeddings.append(chunk_with_embedding)
                
                indonesian_result = await indonesian_kg_manager.process_indonesian_document(
                    document_data, 
                    chunks_with_embeddings
                )
                
                # Get Indonesian-specific statistics
                indonesian_metrics = await indonesian_kg_manager.get_comprehensive_kg_metrics()
                
                result = {
                    "document_id": indonesian_result['document_id'],
                    "pg_document_id": pg_document_id,
                    "pg_chunk_ids": pg_chunk_ids,
                    "file_path": str(file_path),
                    "chunks_created": len(document_data["chunks"]),
                    "status": "success",
                    "language": "indonesian",
                    "indonesian_stats": {
                        "entities_extracted": indonesian_result.get('entities_extracted', 0),
                        "topics_identified": indonesian_result.get('topics_identified', 0),
                        "processing_stats": indonesian_result.get('processing_stats', {})
                    },
                    "kg_quality": {
                        "total_entities": indonesian_metrics.get('derived_metrics', {}).get('total_entities', 0),
                        "quality_score": indonesian_metrics.get('derived_metrics', {}).get('quality_score', 0),
                        "high_confidence_ratio": indonesian_metrics.get('derived_metrics', {}).get('high_confidence_ratio', 0)
                    }
                }
            else:
                # Standard processing for non-Indonesian documents
                document_node_id = await self.kg_manager.create_document_node(document_data)
                
                await self.kg_manager.create_optimized_chunk_nodes(
                    document_node_id, 
                    document_data["chunks"], 
                    embeddings
                )
                
                result = {
                    "document_id": document_node_id,
                    "pg_document_id": pg_document_id,
                    "pg_chunk_ids": pg_chunk_ids,
                    "file_path": str(file_path),
                    "chunks_created": len(document_data["chunks"]),
                    "status": "success",
                    "language": "other"
                }
            
            logger.info(f"Successfully ingested document: {file_path} (PG ID: {pg_document_id})")
            return result
            
        except Exception as e:
            logger.error(f"Failed to ingest document {file_path}: {e}")
            raise

    async def ingest_directory(self, directory_path: Union[str, Path], 
                             file_patterns: Optional[List[str]] = None) -> List[Dict]:
        directory_path = Path(directory_path)
        
        if not directory_path.exists() or not directory_path.is_dir():
            raise ValueError(f"Invalid directory: {directory_path}")
        
        file_patterns = file_patterns or ["*.txt", "*.md", "*.pdf", "*.docx", "*.html"]
        
        files_to_process = []
        for pattern in file_patterns:
            files_to_process.extend(directory_path.glob(f"**/{pattern}"))
        
        results = []
        for file_path in files_to_process:
            try:
                result = await self.ingest_document(file_path)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to process {file_path}: {e}")
                results.append({
                    "file_path": str(file_path),
                    "status": "error",
                    "error": str(e)
                })
        
        return results

    async def search_similar_content(self, query: str, limit: int = 10) -> List[Dict]:
        query_embedding = await model_manager.get_embeddings([query], is_query=True)
        return await self.kg_manager.find_similar_chunks(query_embedding[0], limit)


ingestion_pipeline = IngestionPipeline()