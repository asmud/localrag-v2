import os
import warnings
from pathlib import Path
from typing import Optional, List, Dict

import torch
import numpy as np
from loguru import logger
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoModelForTokenClassification, AutoTokenizer

from core.config import settings


class ModelManager:
    def __init__(self):
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.modules.module")

        self.models_path = Path(settings.models_path)
        self.models_path.mkdir(parents=True, exist_ok=True)
        
        self.embedding_model: Optional[SentenceTransformer] = None
        self.extraction_model: Optional[AutoModel] = None
        self.extraction_tokenizer: Optional[AutoTokenizer] = None
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")

    async def initialize(self):
        if settings.auto_download_models:
            await self._load_embedding_model()
            await self._load_extraction_model()
        else:
            logger.info("Auto-download disabled. Models will be loaded on demand.")

    async def _load_embedding_model(self):
        try:
            model_path = self.models_path / "embedding"
            
            if model_path.exists() and any(model_path.iterdir()):
                logger.info(f"Loading embedding model from {model_path}")
                self.embedding_model = SentenceTransformer(str(model_path))
            else:
                logger.info(f"Downloading embedding model: {settings.embedding_model}")
                self.embedding_model = SentenceTransformer(
                    settings.embedding_model,
                    device=self.device,
                    token=settings.hf_token
                )
                
                model_path.mkdir(parents=True, exist_ok=True)
                self.embedding_model.save(str(model_path))
                logger.info(f"Embedding model saved to {model_path}")
                
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise

    async def _load_extraction_model(self):
        try:
            model_path = self.models_path / "extraction"
            is_torch = True
            
            if model_path.exists() and any(model_path.iterdir()):
                logger.info(f"Loading extraction model from {model_path}")
                self.extraction_tokenizer = AutoTokenizer.from_pretrained(str(model_path))
                
                # Check if model has TensorFlow or PyTorch weights
                has_pytorch_weights = (model_path / "pytorch_model.bin").exists() or (model_path / "model.safetensors").exists()
                has_tf_weights = any(model_path.glob("tf_model.h5")) or any(model_path.glob("*.h5"))
                
                if has_pytorch_weights:
                    logger.info("Loading from PyTorch weights")
                    self.extraction_model = AutoModelForTokenClassification.from_pretrained(
                        str(model_path)
                    )
                elif has_tf_weights:
                    logger.info("Loading from TensorFlow weights")
                    is_torch = False
                    self.extraction_model = AutoModelForTokenClassification.from_pretrained(
                        str(model_path),
                        from_tf=True
                    )
                else:
                    # Fallback: try PyTorch first, then TensorFlow
                    logger.info("Weight format unclear, trying PyTorch first")
                    try:
                        self.extraction_model = AutoModelForTokenClassification.from_pretrained(
                            str(model_path)
                        )
                    except Exception as pytorch_error:
                        logger.warning(f"PyTorch loading failed: {pytorch_error}")
                        logger.info("Trying TensorFlow loading")
                        is_torch = False
                        self.extraction_model = AutoModelForTokenClassification.from_pretrained(
                            str(model_path),
                            from_tf=True
                        )
            else:
                logger.info(f"Downloading extraction model: {settings.extraction_model}")
                self.extraction_tokenizer = AutoTokenizer.from_pretrained(
                    settings.extraction_model,
                    token=settings.hf_token
                )

                self.extraction_model = AutoModelForTokenClassification.from_pretrained(
                    settings.extraction_model,
                    token=settings.hf_token
                )
                
                model_path.mkdir(parents=True, exist_ok=True)
                self.extraction_tokenizer.save_pretrained(str(model_path))
                self.extraction_model.save_pretrained(str(model_path))
                logger.info(f"Extraction model saved to {model_path}")
                
            # Move model to appropriate device
            try:
                if is_torch:
                    self.extraction_model.to(self.device)
                    logger.info(f"Extraction model moved to device: {self.device}")
                else:
                    logger.info("TensorFlow model — skipping device placement")

            except Exception as device_error:
                logger.warning(f"Could not move model to device {self.device}: {device_error}")
                logger.info("Model will remain on default device")
            
        except Exception as e:
            logger.error(f"Failed to load extraction model: {e}")
            raise

    async def get_embeddings(self, texts: list[str], is_query: bool = False) -> list[list[float]]:
        if not self.embedding_model:
            await self._load_embedding_model()
        
        try:
            # Validate and sanitize input texts
            clean_texts = []
            for text in texts:
                if text and isinstance(text, str):
                    # Sanitize text to prevent encoding issues
                    clean_text = self._sanitize_text_for_embedding(text, is_query)
                    if clean_text:
                        clean_texts.append(clean_text)
                    else:
                        # Add placeholder for empty/corrupted text
                        clean_texts.append("text unavailable")
                        logger.warning("Empty or corrupted text replaced with placeholder")
                else:
                    clean_texts.append("text unavailable")
                    logger.warning("Invalid text input replaced with placeholder")
            
            if not clean_texts:
                logger.error("No valid texts provided for embedding")
                return []
            
            embeddings = self.embedding_model.encode(
                clean_texts,
                convert_to_tensor=False,
                show_progress_bar=False,
                batch_size=32
            )
            
            # Validate embedding output
            embedding_list = embeddings.tolist()
            
            # Check for NaN or invalid embeddings
            valid_embeddings = []
            for i, embedding in enumerate(embedding_list):
                if self._is_valid_embedding(embedding):
                    valid_embeddings.append(embedding)
                else:
                    logger.warning(f"Invalid embedding generated for text: {clean_texts[i][:50]}...")
                    # Create zero embedding as fallback
                    valid_embeddings.append([0.0] * len(embedding))
            
            return valid_embeddings
            
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            raise

    def _is_indonesian_text(self, text: str) -> bool:
        """Detect if text is primarily Indonesian language."""
        indonesian_indicators = [
            'yang', 'dengan', 'dalam', 'untuk', 'dari', 'pada', 'oleh', 'sebagai',
            'adalah', 'akan', 'dapat', 'harus', 'perlu', 'tentang', 'kepada',
            'kementerian', 'dinas', 'badan', 'peraturan', 'undang-undang'
        ]
        
        text_lower = text.lower()
        indonesian_count = sum(1 for word in indonesian_indicators if word in text_lower)
        
        total_words = len(text.split())
        if total_words == 0:
            return False
        
        indonesian_ratio = indonesian_count / min(total_words, 50)
        return indonesian_ratio > 0.15

    def _sanitize_text_for_embedding(self, text: str, is_query: bool = False) -> str:
        """Sanitize text specifically for embedding generation"""
        if not text or not isinstance(text, str):
            return ""
        
        try:
            # Handle bytes
            if isinstance(text, bytes):
                text = text.decode('utf-8', errors='replace')
            
            # Remove problematic characters
            text = text.replace('\x00', '').replace('\ufffd', '')
            
            # Normalize excessive whitespace
            text = ' '.join(text.split())
            
            # Truncate very long texts to prevent model issues
            max_length = 8000  # Safe limit for most embedding models
            if len(text) > max_length:
                text = text[:max_length].rsplit(' ', 1)[0]  # Cut at word boundary
            
            # Check for corruption patterns
            import re
            
            # Remove excessive repetition that could confuse embeddings
            text = re.sub(r'(.{2,})\1{3,}', r'\1', text)
            
            # Check if text is mostly meaningful
            # Allow short queries (common search terms) but filter out very short corrupted content
            stripped_text = text.strip()
            if is_query:
                # For queries, allow very short text (2+ characters for search terms)
                if len(stripped_text) < 2:
                    return ""
            else:
                # For document content, maintain stricter filtering (minimum 5 characters)
                if len(stripped_text) < 5:
                    return ""
            
            # Check for garbled patterns (specific patterns found in actual data)
            garbled_patterns = [
                # Specific Indonesian corruption patterns
                r'K\s+B\s+U\s+S\s+N\s+A\s+T\s+A\s+I\s+G\s+E\s+K',  # KEGIATAN SUBKEGIATAN reversed
                r'A\s+R\s+G\s+O\s+R\s+P\s+N\s+A\s+T\s+A\s+I\s+G\s+E\s+K',  # KEGIATAN PROGRAM reversed
                r'S\s+U\s+R\s+U\s+M\s+A\s+R\s+G\s+O\s+R\s+P',      # PROGRAM URUSAN reversed
                r'O\s+R\s+P\s+N\s+A\s+T\s+A\s+I\s+G\s+E\s+K',      # KEGIATAN PRO reversed
                r'M\s+A\s+R\s+G\s+O\s+R\s+P\s+N\s+A\s+T\s+A\s+I\s+G\s+E\s+K',  # KEGIATAN PROGRAM reversed
                
                # General corruption patterns (more specific)
                r'^([A-Z]\s){10,}',  # 10+ single letters with spaces at start
                r'^[A-Z\s]{60,}$',  # 60+ chars of only caps and spaces (more specific)
                r'(.{1,2})\1{8,}',  # Same 1-2 char sequence repeated 8+ times
                
                # Excessive single letter spacing (likely corruption)
                r'([A-Z]\s){15,}',  # 15+ single letters with spaces anywhere
            ]
            
            for pattern in garbled_patterns:
                if re.search(pattern, text.strip()):
                    logger.warning(f"Detected garbled text pattern, skipping: {text[:50]}...")
                    return ""
            
            return text.strip()
            
        except Exception as e:
            logger.warning(f"Text sanitization for embedding failed: {e}")
            return ""

    def _is_valid_embedding(self, embedding: list) -> bool:
        """Check if embedding vector is valid"""
        if not embedding or not isinstance(embedding, list):
            return False
        
        try:
            # Check for NaN or infinite values
            for value in embedding:
                if not isinstance(value, (int, float)):
                    return False
                if value != value:  # NaN check
                    return False
                if abs(value) == float('inf'):
                    return False
            
            # Check for zero vector (might indicate failure)
            if all(abs(v) < 1e-10 for v in embedding):
                return False
            
            # Check for reasonable magnitude
            magnitude = sum(v * v for v in embedding) ** 0.5
            if magnitude < 1e-10 or magnitude > 1e10:
                return False
            
            return True
            
        except Exception:
            return False

    async def extract_features(self, text: str) -> dict:
        if not self.extraction_model or not self.extraction_tokenizer:
            await self._load_extraction_model()
        
        try:
            inputs = self.extraction_tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            )
            
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.extraction_model(**inputs)
                
            features = {
                "pooler_output": outputs.pooler_output.cpu().numpy() if hasattr(outputs, 'pooler_output') else None
            }
            
            return features
            
        except Exception as e:
            logger.error(f"Failed to extract features: {e}")
            raise

    async def extract_entities_with_ner(self, text: str) -> List[Dict]:
        """Extract Indonesian entities using the NER model with proper token classification."""
        if not self.extraction_model or not self.extraction_tokenizer:
            await self._load_extraction_model()
        
        try:
            # Tokenize input text
            inputs = self.extraction_tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
                return_offsets_mapping=True
            )
            
            # Move to appropriate device
            device_inputs = {k: v.to(self.device) for k, v in inputs.items() if k != 'offset_mapping'}
            offset_mapping = inputs['offset_mapping']
            
            # Get model predictions
            with torch.no_grad():
                outputs = self.extraction_model(**device_inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
                predicted_token_class = torch.argmax(predictions, dim=-1)
            
            # Convert predictions to entities
            entities = self._convert_ner_predictions_to_entities(
                text, 
                predicted_token_class[0].cpu().numpy(), 
                predictions[0].cpu().numpy(),
                offset_mapping[0].cpu().numpy(),
                inputs['input_ids'][0].cpu().numpy()
            )
            
            return entities
            
        except Exception as e:
            logger.error(f"Failed to extract entities with NER: {e}")
            return []

    async def extract_entities_batch(self, texts: List[str], batch_size: int = 8) -> List[List[Dict]]:
        """Extract entities from multiple texts in batches for better performance."""
        if not self.extraction_model or not self.extraction_tokenizer:
            await self._load_extraction_model()
        
        all_entities = []
        
        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            try:
                # Tokenize batch
                inputs = self.extraction_tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_offsets_mapping=True
                )
                
                # Move to appropriate device
                device_inputs = {k: v.to(self.device) for k, v in inputs.items() if k != 'offset_mapping'}
                offset_mappings = inputs['offset_mapping']
                
                # Get model predictions for the batch
                with torch.no_grad():
                    outputs = self.extraction_model(**device_inputs)
                    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
                    predicted_token_classes = torch.argmax(predictions, dim=-1)
                
                # Convert predictions for each text in the batch
                batch_entities = []
                for j, text in enumerate(batch_texts):
                    entities = self._convert_ner_predictions_to_entities(
                        text,
                        predicted_token_classes[j].cpu().numpy(),
                        predictions[j].cpu().numpy(),
                        offset_mappings[j].cpu().numpy(),
                        inputs['input_ids'][j].cpu().numpy()
                    )
                    batch_entities.append(entities)
                
                all_entities.extend(batch_entities)
                
                logger.info(f"Processed batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}")
                
            except Exception as e:
                logger.error(f"Failed to process batch {i//batch_size + 1}: {e}")
                # Add empty lists for failed batch
                all_entities.extend([[] for _ in batch_texts])
        
        return all_entities

    def _convert_ner_predictions_to_entities(self, text: str, predicted_classes: np.ndarray, 
                                           predictions: np.ndarray, offset_mapping: np.ndarray,
                                           input_ids: np.ndarray) -> List[Dict]:
        """Convert NER model predictions to entity dictionaries."""
        entities = []
        
        # Get label names from model config
        id2label = self.extraction_model.config.id2label
        
        # Entity type mapping from NER model labels to Neo4j schema
        ner_to_neo4j_mapping = {
            'B-PER': 'PER', 'I-PER': 'PER',
            'B-ORG': 'ORG', 'I-ORG': 'ORG', 
            'B-NOR': 'NOR', 'I-NOR': 'NOR',
            'B-GPE': 'GPE', 'I-GPE': 'GPE',
            'B-LOC': 'LOC', 'I-LOC': 'LOC',
            'B-FAC': 'FAC', 'I-FAC': 'FAC',
            'B-LAW': 'LAW', 'I-LAW': 'LAW',
            'B-EVT': 'EVT', 'I-EVT': 'EVT',
            'B-DAT': 'DAT', 'I-DAT': 'DAT',
            'B-TIM': 'TIM', 'I-TIM': 'TIM',
            'B-CRD': 'CRD', 'I-CRD': 'CRD',
            'B-ORD': 'ORD', 'I-ORD': 'ORD',
            'B-QTY': 'QTY', 'I-QTY': 'QTY',
            'B-PRC': 'PRC', 'I-PRC': 'PRC',
            'B-MON': 'MON', 'I-MON': 'MON',
            'B-PRD': 'PRD', 'I-PRD': 'PRD',
            'B-REG': 'REG', 'I-REG': 'REG',
            'B-WOA': 'WOA', 'I-WOA': 'WOA',
            'B-LAN': 'LAN', 'I-LAN': 'LAN',
            'O': 'O'
        }
        
        current_entity = None
        current_tokens = []
        
        # Skip special tokens (CLS, SEP, PAD)
        special_tokens = {self.extraction_tokenizer.cls_token_id, 
                         self.extraction_tokenizer.sep_token_id, 
                         self.extraction_tokenizer.pad_token_id}
        
        for i, (predicted_class, prediction_scores, offset, input_id) in enumerate(
            zip(predicted_classes, predictions, offset_mapping, input_ids)
        ):
            # Skip special tokens
            if input_id in special_tokens:
                continue
                
            # Skip tokens with no offset mapping (usually special tokens)
            if offset[0] == 0 and offset[1] == 0 and i != 0:
                continue
            
            # Get label name
            label = id2label.get(predicted_class, 'O')
            
            # Get confidence score
            confidence = float(prediction_scores[predicted_class])
            
            # Skip low confidence predictions
            if confidence < 0.5:
                label = 'O'
            
            # Handle BIO tagging
            if label.startswith('B-'):
                # Begin new entity
                if current_entity:
                    # Finish previous entity
                    entities.append(current_entity)
                
                entity_type = ner_to_neo4j_mapping.get(label, 'O')
                if entity_type != 'O':
                    current_entity = {
                        'start': int(offset[0]),
                        'end': int(offset[1]),
                        'type': entity_type,
                        'confidence': confidence,
                        'tokens': [i]
                    }
                    current_tokens = [text[offset[0]:offset[1]]]
                else:
                    current_entity = None
                    current_tokens = []
                    
            elif label.startswith('I-'):
                # Continue current entity
                if current_entity and current_entity['type'] == ner_to_neo4j_mapping.get(label, 'O'):
                    current_entity['end'] = int(offset[1])
                    current_entity['confidence'] = (current_entity['confidence'] + confidence) / 2
                    current_entity['tokens'].append(i)
                    current_tokens.append(text[offset[0]:offset[1]])
                else:
                    # Start new entity if I- appears without B-
                    entity_type = ner_to_neo4j_mapping.get(label, 'O')
                    if entity_type != 'O':
                        current_entity = {
                            'start': int(offset[0]),
                            'end': int(offset[1]),
                            'type': entity_type,
                            'confidence': confidence,
                            'tokens': [i]
                        }
                        current_tokens = [text[offset[0]:offset[1]]]
                    else:
                        current_entity = None
                        current_tokens = []
                        
            else:  # 'O' or unknown
                # Finish current entity
                if current_entity:
                    entities.append(current_entity)
                    current_entity = None
                    current_tokens = []
        
        # Don't forget the last entity
        if current_entity:
            entities.append(current_entity)
        
        # Add text content and normalize entities
        final_entities = []
        for entity in entities:
            entity_text = text[entity['start']:entity['end']].strip()
            
            # Skip empty or very short entities
            if len(entity_text) < 2:
                continue
            
            # Add normalized text
            entity['text'] = entity_text
            entity['normalized_text'] = self._normalize_indonesian_entity_text(entity_text)
            
            final_entities.append(entity)
        
        return final_entities

    def _normalize_indonesian_entity_text(self, text: str) -> str:
        """Normalize Indonesian entity text for better matching."""
        # Convert to lowercase
        normalized = text.lower().strip()
        
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


    def get_model_info(self) -> dict:
        return {
            "embedding_model": settings.embedding_model,
            "extraction_model": settings.extraction_model,
            "device": self.device,
            "models_path": str(self.models_path),
            "embedding_model_loaded": self.embedding_model is not None,
            "extraction_model_loaded": self.extraction_model is not None,
        }

    async def cleanup(self):
        if hasattr(self.embedding_model, 'device'):
            self.embedding_model = None
        if self.extraction_model:
            self.extraction_model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


model_manager = ModelManager()