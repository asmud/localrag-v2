"""
Enhanced OCR module with multiple engines and Indonesian language support.
"""

import cv2
import numpy as np
import os
import re
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from PIL import Image, ImageEnhance, ImageFilter
from loguru import logger

from .config import settings

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False
    logger.warning("EasyOCR not available. Install with: pip install easyocr")

try:
    import paddleocr
    PADDLEOCR_AVAILABLE = True
except ImportError:
    PADDLEOCR_AVAILABLE = False
    logger.warning("PaddleOCR not available. Install with: pip install paddleocr")

try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    logger.warning("PyTesseract not available. Install with: pip install pytesseract")

from markitdown import MarkItDown

try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False
    logger.warning("PyMuPDF not available. Install with: pip install pymupdf")


def setup_easyocr_symlink():
    """
    Set up symlink from ~/.EasyOCR to our pre-downloaded models directory.
    This makes EasyOCR use our cached models instead of downloading new ones.
    """
    try:
        home_dir = Path.home()
        easyocr_cache = home_dir / ".EasyOCR"
        our_models_dir = Path(getattr(settings, 'ocr_models_cache_dir', './models/ocr')) / "easyocr"
        
        # Check if our models directory has valid models (including nested structure)
        if our_models_dir.exists() and any(our_models_dir.glob("**/*.pth")):
            # If ~/.EasyOCR already exists and is not a symlink, remove it
            if easyocr_cache.exists() and not easyocr_cache.is_symlink():
                if easyocr_cache.is_dir():
                    import shutil
                    shutil.rmtree(easyocr_cache)
                else:
                    easyocr_cache.unlink()
            
            # Create symlink if it doesn't exist
            if not easyocr_cache.exists():
                easyocr_cache.symlink_to(our_models_dir.resolve())
                logger.info(f"Created EasyOCR symlink: {easyocr_cache} -> {our_models_dir}")
                return True
            elif easyocr_cache.is_symlink():
                logger.info(f"EasyOCR symlink already exists: {easyocr_cache}")
                return True
        else:
            logger.warning(f"Pre-downloaded models not found or invalid at {our_models_dir}")
            return False
            
    except Exception as e:
        logger.warning(f"Failed to setup EasyOCR symlink: {e}")
        return False


def safe_paddleocr_init(timeout_seconds=60):
    """
    Safely initialize PaddleOCR with timeout and error handling.
    Returns (success, engine_or_error_message)
    """
    if not PADDLEOCR_AVAILABLE:
        return False, "PaddleOCR not available"
    
    def init_paddle():
        """Function to run in separate process for PaddleOCR initialization."""
        try:
            import paddleocr
            
            # Check configuration settings
            force_cpu = getattr(settings, 'ocr_force_cpu_mode', False)
            gpu_enabled = getattr(settings, 'ocr_gpu_enabled', True) and not force_cpu
            
            paddle_kwargs = {
                'use_angle_cls': True,
                'lang': 'en',
                'use_gpu': gpu_enabled,
                'show_log': False,  # Reduce verbose output
            }
            
            ocr_engine = paddleocr.PaddleOCR(**paddle_kwargs)
            return True, ocr_engine
            
        except Exception as e:
            return False, str(e)
    
    # Try initialization with timeout
    result = [None]
    exception_result = [None]
    
    def target():
        try:
            success, engine = init_paddle()
            result[0] = (success, engine)
        except Exception as e:
            exception_result[0] = str(e)
            result[0] = (False, str(e))
    
    thread = threading.Thread(target=target)
    thread.daemon = True
    thread.start()
    thread.join(timeout=timeout_seconds)
    
    if thread.is_alive():
        logger.warning(f"PaddleOCR initialization timeout after {timeout_seconds}s")
        return False, "Initialization timeout"
    
    if result[0] is None:
        return False, "Initialization failed without result"
    
    return result[0]


class PageContentType:
    """Enumeration for page content types."""
    TEXT = "text"
    IMAGE = "image" 
    MIXED = "mixed"


class PageInfo:
    """Information about a PDF page."""
    
    def __init__(self, page_num: int, content_type: str, has_text: bool, has_images: bool, 
                 text_ratio: float = 0.0, image_ratio: float = 0.0):
        self.page_num = page_num
        self.content_type = content_type
        self.has_text = has_text
        self.has_images = has_images
        self.text_ratio = text_ratio
        self.image_ratio = image_ratio


class OCRResult:
    """OCR result container with confidence metrics."""
    
    def __init__(self, text: str, confidence: float, engine: str, metadata: Dict = None):
        self.text = text
        self.confidence = confidence
        self.engine = engine
        self.metadata = metadata or {}
        
    def __str__(self):
        return f"OCRResult(text='{self.text[:50]}...', confidence={self.confidence:.3f}, engine='{self.engine}')"


class PDFPageAnalyzer:
    """Analyze PDF pages to determine content type and processing strategy."""
    
    def __init__(self):
        self.text_threshold = 0.1  # Minimum text ratio to consider page as text-based
        self.image_threshold = 0.3  # Minimum image ratio to consider page as image-based
    
    def analyze_pdf_pages(self, file_path: Union[str, Path]) -> List[PageInfo]:
        """Analyze all pages in a PDF and return page information."""
        if not PYMUPDF_AVAILABLE:
            logger.warning("PyMuPDF not available, cannot analyze PDF pages")
            return []
            
        try:
            doc = fitz.open(str(file_path))
            page_infos = []
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                page_info = self._analyze_page(page, page_num)
                page_infos.append(page_info)
                
            doc.close()
            return page_infos
            
        except Exception as e:
            logger.error(f"Failed to analyze PDF pages: {e}")
            return []
    
    def _analyze_page(self, page, page_num: int) -> PageInfo:
        """Analyze a single PDF page."""
        try:
            # Get text content
            text = page.get_text()
            has_text = bool(text.strip())
            
            # Get images
            image_list = page.get_images()
            has_images = len(image_list) > 0
            
            # Calculate ratios
            page_rect = page.rect
            page_area = page_rect.width * page_rect.height
            
            # Estimate text coverage (rough approximation)
            text_blocks = page.get_text("dict")["blocks"]
            text_area = 0
            for block in text_blocks:
                if "lines" in block:  # Text block
                    for line in block["lines"]:
                        for span in line["spans"]:
                            bbox = span["bbox"]
                            text_area += (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            
            text_ratio = text_area / page_area if page_area > 0 else 0
            
            # Estimate image coverage
            image_area = 0
            for img_info in image_list:
                bbox = page.get_image_bbox(img_info[7])  # Get image bbox
                if bbox:
                    image_area += (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            
            image_ratio = image_area / page_area if page_area > 0 else 0
            
            # Determine content type
            if text_ratio > self.text_threshold and image_ratio < self.image_threshold:
                content_type = PageContentType.TEXT
            elif image_ratio > self.image_threshold and text_ratio < self.text_threshold:
                content_type = PageContentType.IMAGE
            else:
                content_type = PageContentType.MIXED
                
            return PageInfo(
                page_num=page_num,
                content_type=content_type,
                has_text=has_text,
                has_images=has_images,
                text_ratio=text_ratio,
                image_ratio=image_ratio
            )
            
        except Exception as e:
            logger.warning(f"Failed to analyze page {page_num}: {e}")
            # Default to mixed content on analysis failure
            return PageInfo(
                page_num=page_num,
                content_type=PageContentType.MIXED,
                has_text=True,
                has_images=False
            )


class ImagePreprocessor:
    """Image preprocessing for improved OCR accuracy."""
    
    @staticmethod
    def enhance_image(image: Image.Image) -> Image.Image:
        """Apply image enhancements to improve OCR accuracy."""
        try:
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
                
            # Enhance contrast
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(1.2)
            
            # Enhance sharpness
            enhancer = ImageEnhance.Sharpness(image)
            image = enhancer.enhance(1.1)
            
            # Apply slight denoising
            image = image.filter(ImageFilter.MedianFilter(size=3))
            
            return image
            
        except Exception as e:
            logger.warning(f"Image enhancement failed: {e}")
            return image
    
    @staticmethod
    def check_dpi(image: Image.Image) -> Tuple[int, int]:
        """Check image DPI and recommend upscaling if needed."""
        try:
            dpi = image.info.get('dpi', (72, 72))
            if isinstance(dpi, (int, float)):
                dpi = (dpi, dpi)
            return dpi
        except:
            return (72, 72)  # Default DPI
    
    @staticmethod
    def upscale_if_needed(image: Image.Image, target_dpi: int = 300) -> Image.Image:
        """Upscale image if DPI is below target."""
        try:
            current_dpi = ImagePreprocessor.check_dpi(image)
            avg_dpi = sum(current_dpi) / 2
            
            if avg_dpi < target_dpi:
                scale_factor = target_dpi / avg_dpi
                new_size = (int(image.width * scale_factor), int(image.height * scale_factor))
                
                if new_size[0] <= settings.ocr_max_image_size and new_size[1] <= settings.ocr_max_image_size:
                    image = image.resize(new_size, Image.Resampling.LANCZOS)
                    logger.info(f"Upscaled image from {avg_dpi:.0f} DPI to {target_dpi} DPI")
                else:
                    logger.warning(f"Image too large to upscale: {new_size}")
                    
            return image
            
        except Exception as e:
            logger.warning(f"Image upscaling failed: {e}")
            return image
    
    @staticmethod
    def preprocess_image(image_path: Union[str, Path]) -> Image.Image:
        """Complete image preprocessing pipeline."""
        try:
            image = Image.open(image_path)
            
            if settings.ocr_enable_preprocessing:
                # Check and improve DPI
                image = ImagePreprocessor.upscale_if_needed(image, settings.ocr_dpi_threshold)
                
                # Apply enhancements
                image = ImagePreprocessor.enhance_image(image)
                
                logger.debug(f"Preprocessed image: {image.size}, mode: {image.mode}")
                
            return image
            
        except Exception as e:
            logger.error(f"Image preprocessing failed for {image_path}: {e}")
            raise


class IndonesianTextProcessor:
    """Indonesian-specific text processing and correction."""
    
    def __init__(self):
        # Common OCR errors in Indonesian text
        self.correction_patterns = {
            # Common character misrecognitions
            r'\b0(?=\w)': 'o',  # 0 -> o at word boundaries
            r'(?<=\w)0\b': 'o',  # 0 -> o at word boundaries  
            r'\b1(?=[aeiou])': 'i',  # 1 -> i before vowels
            r'(?<=[aeiou])1(?=[aeiou])': 'l',  # 1 -> l between vowels
            r'\brn(?=\w)': 'm',  # rn -> m at start of words
            r'(?<=\w)rn\b': 'm',  # rn -> m at end of words
            
            # Indonesian-specific patterns
            r'\bdan9\b': 'yang',  # Common misrecognition
            r'\b1tu\b': 'itu',
            r'\bd1\b': 'di',
            r'\bke(?=\d)': 'ke ',  # Add space after 'ke'
            r'\bpada(?=\d)': 'pada ',  # Add space after 'pada'
        }
        
        # Indonesian common words for validation
        self.common_words = {
            'dan', 'yang', 'dengan', 'untuk', 'dari', 'pada', 'dalam', 'atau', 'oleh',
            'kepada', 'tentang', 'sebagai', 'adalah', 'akan', 'dapat', 'harus', 'perlu',
            'pemerintah', 'daerah', 'kabupaten', 'kota', 'provinsi', 'kementerian', 'badan',
            'dinas', 'kantor', 'lembaga', 'direktur', 'kepala', 'sekretaris', 'bidang',
            'bagian', 'sub', 'unit', 'divisi', 'program', 'kegiatan', 'anggaran', 'dana'
        }
    
    def correct_text(self, text: str) -> str:
        """Apply Indonesian-specific text corrections."""
        if not settings.ocr_enable_postprocessing:
            return text
            
        try:
            corrected = text
            
            # Apply correction patterns
            for pattern, replacement in self.correction_patterns.items():
                corrected = re.sub(pattern, replacement, corrected, flags=re.IGNORECASE)
            
            # Fix spacing issues
            corrected = re.sub(r'\s+', ' ', corrected)  # Multiple spaces to single
            corrected = re.sub(r'([.!?])\s*([A-Z])', r'\1 \2', corrected)  # Space after punctuation
            
            return corrected.strip()
            
        except Exception as e:
            logger.warning(f"Text correction failed: {e}")
            return text
    
    def validate_quality(self, text: str) -> float:
        """Estimate text quality based on Indonesian language patterns."""
        if not text or len(text) < 10:
            return 0.0
            
        try:
            words = text.lower().split()
            if not words:
                return 0.0
                
            # Check for common Indonesian words
            common_word_count = sum(1 for word in words if word in self.common_words)
            common_word_ratio = common_word_count / len(words)
            
            # Check for suspicious patterns (too many numbers/special chars)
            suspicious_chars = sum(1 for char in text if not char.isalnum() and char not in ' .,!?-()[]{}:;"\'')
            suspicious_ratio = suspicious_chars / len(text) if text else 1.0
            
            # Check for single-letter spacing (corruption indicator)
            single_letter_pattern = len(re.findall(r'\b[A-Z]\s+[A-Z]\s+', text))
            single_letter_penalty = min(single_letter_pattern / 10, 0.5)
            
            # Calculate overall quality score
            quality_score = (
                common_word_ratio * 0.4 +           # Indonesian word presence
                (1 - suspicious_ratio) * 0.3 +      # Clean character ratio
                (1 - single_letter_penalty) * 0.3   # Lack of corruption patterns
            )
            
            return max(0.0, min(1.0, quality_score))
            
        except Exception as e:
            logger.warning(f"Quality validation failed: {e}")
            return 0.5  # Neutral score on error


class EnhancedOCREngine:
    """Enhanced OCR engine with multiple backends and Indonesian support."""
    
    def __init__(self):
        self.preprocessor = ImagePreprocessor()
        self.text_processor = IndonesianTextProcessor()
        self.markitdown = MarkItDown()
        self.pdf_analyzer = PDFPageAnalyzer()
        
        # Lazy loading for OCR engines
        self.engines = {}
        self._engines_initialized = False
        self._easyocr_reader = None
        self._paddleocr_engine = None
        
        # Always mark MarkItDown and Tesseract as available (no model download needed)
        self.engines['markitdown'] = self.markitdown
        if TESSERACT_AVAILABLE:
            self.engines['tesseract'] = 'available'
            logger.info("Tesseract marked as available (lazy loading)")
        logger.info("MarkItDown initialized successfully")
    
    @property
    def easyocr_reader(self):
        """Lazy loading property for EasyOCR reader."""
        if self._easyocr_reader is None and EASYOCR_AVAILABLE:
            try:
                logger.info("Initializing EasyOCR reader (lazy loading)...")
                
                # Set up symlink to use pre-downloaded models
                setup_easyocr_symlink()
                
                # Check configuration settings
                force_cpu = getattr(settings, 'ocr_force_cpu_mode', False)
                gpu_enabled = getattr(settings, 'ocr_gpu_enabled', True) and not force_cpu
                
                self._easyocr_reader = easyocr.Reader(
                    settings.ocr_languages,
                    gpu=gpu_enabled
                )
                self.engines['easyocr'] = self._easyocr_reader
                logger.info("EasyOCR initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize EasyOCR: {e}")
                self._easyocr_reader = False  # Mark as failed
        return self._easyocr_reader if self._easyocr_reader is not False else None
    
    @property
    def paddleocr_engine(self):
        """Lazy loading property for PaddleOCR engine with safe initialization."""
        if self._paddleocr_engine is None and PADDLEOCR_AVAILABLE:
            # Check if safe mode is enabled
            safe_mode = getattr(settings, 'ocr_safe_mode', True)
            
            if safe_mode:
                logger.info("Initializing PaddleOCR engine (safe mode with timeout)...")
                
                # Use safe initialization with timeout
                timeout = getattr(settings, 'ocr_initialization_timeout', 60)
                success, result = safe_paddleocr_init(timeout_seconds=timeout)
                
                if success:
                    self._paddleocr_engine = result
                    self.engines['paddleocr'] = self._paddleocr_engine
                    logger.info("PaddleOCR initialized successfully")
                else:
                    logger.error(f"PaddleOCR initialization failed: {result}")
                    logger.info("PaddleOCR will be disabled for this session")
                    self._paddleocr_engine = False  # Mark as failed
            else:
                logger.warning("OCR safe mode disabled - using direct PaddleOCR initialization")
                try:
                    import paddleocr
                    force_cpu = getattr(settings, 'ocr_force_cpu_mode', False)
                    gpu_enabled = getattr(settings, 'ocr_gpu_enabled', True) and not force_cpu
                    
                    self._paddleocr_engine = paddleocr.PaddleOCR(
                        use_angle_cls=True,
                        lang='en',
                        use_gpu=gpu_enabled,
                        show_log=False
                    )
                    self.engines['paddleocr'] = self._paddleocr_engine
                    logger.info("PaddleOCR initialized successfully (unsafe mode)")
                except Exception as e:
                    logger.error(f"PaddleOCR initialization failed: {e}")
                    self._paddleocr_engine = False
                
        return self._paddleocr_engine if self._paddleocr_engine is not False else None
        
    def get_available_engines(self):
        """Get list of available OCR engines without initializing them, respecting disabled engines."""
        available = []
        disabled_engines = getattr(settings, 'ocr_disable_engines', [])
        
        if 'markitdown' in self.engines and 'markitdown' not in disabled_engines:
            available.append('markitdown')
        if 'tesseract' in self.engines and 'tesseract' not in disabled_engines:
            available.append('tesseract')
        if EASYOCR_AVAILABLE and 'easyocr' not in disabled_engines:
            available.append('easyocr')
        if PADDLEOCR_AVAILABLE and 'paddleocr' not in disabled_engines:
            available.append('paddleocr')
        return available
    
    def get_working_engines(self):
        """Get list of OCR engines that are actually working (tested)."""
        working = []
        disabled_engines = getattr(settings, 'ocr_disable_engines', [])
        
        # MarkItDown is always working if available and not disabled
        if 'markitdown' in self.engines and 'markitdown' not in disabled_engines:
            working.append('markitdown')
            
        # Test Tesseract
        if 'tesseract' in self.engines and 'tesseract' not in disabled_engines:
            try:
                if TESSERACT_AVAILABLE:
                    import pytesseract
                    # Quick test to see if tesseract is working
                    pytesseract.get_tesseract_version()
                    working.append('tesseract')
            except Exception as e:
                logger.warning(f"Tesseract not working: {e}")
        
        # Test EasyOCR (lazy load) - only if not disabled
        if EASYOCR_AVAILABLE and 'easyocr' not in disabled_engines:
            try:
                reader = self.easyocr_reader
                if reader is not None:
                    working.append('easyocr')
            except Exception as e:
                logger.warning(f"EasyOCR not working: {e}")
                
        # Test PaddleOCR (lazy load with safe init) - only if not disabled
        if PADDLEOCR_AVAILABLE and 'paddleocr' not in disabled_engines:
            try:
                engine = self.paddleocr_engine
                if engine is not None:
                    working.append('paddleocr')
            except Exception as e:
                logger.warning(f"PaddleOCR not working: {e}")
                
        return working
    
    def get_safe_engine_list(self):
        """Get prioritized list of engines to try, with safe fallbacks."""
        working_engines = self.get_working_engines()
        
        # Primary engine preference
        primary = getattr(settings, 'ocr_engine', 'markitdown')
        fallback = getattr(settings, 'ocr_fallback_engine', 'tesseract')
        
        # Build prioritized list
        engines_to_try = []
        
        # Add primary if working
        if primary in working_engines:
            engines_to_try.append(primary)
            
        # Add fallback if working and different from primary
        if fallback != primary and fallback in working_engines:
            engines_to_try.append(fallback)
            
        # Add remaining working engines
        for engine in working_engines:
            if engine not in engines_to_try:
                engines_to_try.append(engine)
                
        # Always ensure MarkItDown is last resort if available
        if 'markitdown' not in engines_to_try and 'markitdown' in working_engines:
            engines_to_try.append('markitdown')
            
        return engines_to_try
        
    def extract_text_easyocr(self, image_path: Union[str, Path]) -> OCRResult:
        """Extract text using EasyOCR."""
        try:
            reader = self.easyocr_reader
            if reader is None:
                raise ValueError("EasyOCR not available")
            results = reader.readtext(str(image_path))
            
            # Combine all text with confidence scoring
            text_parts = []
            confidences = []
            
            for (bbox, text, confidence) in results:
                if confidence >= settings.ocr_confidence_threshold:
                    text_parts.append(text)
                    confidences.append(confidence)
                    
            combined_text = ' '.join(text_parts)
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
            
            # Apply Indonesian text processing
            corrected_text = self.text_processor.correct_text(combined_text)
            quality_score = self.text_processor.validate_quality(corrected_text)
            
            # Adjust confidence based on quality
            final_confidence = (avg_confidence + quality_score) / 2
            
            return OCRResult(
                text=corrected_text,
                confidence=final_confidence,
                engine='easyocr',
                metadata={
                    'raw_confidence': avg_confidence,
                    'quality_score': quality_score,
                    'detected_blocks': len(results)
                }
            )
            
        except Exception as e:
            logger.error(f"EasyOCR extraction failed: {e}")
            raise
            
    def extract_text_paddleocr(self, image_path: Union[str, Path]) -> OCRResult:
        """Extract text using PaddleOCR."""
        try:
            ocr = self.paddleocr_engine
            if ocr is None:
                raise ValueError("PaddleOCR not available")
            results = ocr.ocr(str(image_path), cls=True)
            
            # Combine all text
            text_parts = []
            confidences = []
            
            for line in results[0] if results[0] else []:
                if line and len(line) >= 2:
                    text = line[1][0]
                    confidence = line[1][1]
                    
                    if confidence >= settings.ocr_confidence_threshold:
                        text_parts.append(text)
                        confidences.append(confidence)
                        
            combined_text = ' '.join(text_parts)
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
            
            # Apply Indonesian text processing
            corrected_text = self.text_processor.correct_text(combined_text)
            quality_score = self.text_processor.validate_quality(corrected_text)
            
            # Adjust confidence based on quality
            final_confidence = (avg_confidence + quality_score) / 2
            
            return OCRResult(
                text=corrected_text,
                confidence=final_confidence,
                engine='paddleocr',
                metadata={
                    'raw_confidence': avg_confidence,
                    'quality_score': quality_score,
                    'detected_lines': len(text_parts)
                }
            )
            
        except Exception as e:
            logger.error(f"PaddleOCR extraction failed: {e}")
            raise
            
    def extract_text_tesseract(self, image_path: Union[str, Path]) -> OCRResult:
        """Extract text using Tesseract."""
        try:
            if not TESSERACT_AVAILABLE:
                raise ValueError("Tesseract not available")
                
            # Configure Tesseract for Indonesian
            config = '--oem 3 --psm 6 -l eng+ind'
            
            # Extract text with confidence
            image = Image.open(image_path)
            data = pytesseract.image_to_data(image, config=config, output_type=pytesseract.Output.DICT)
            
            # Filter by confidence and combine text
            text_parts = []
            confidences = []
            
            for i, confidence in enumerate(data['conf']):
                if int(confidence) >= (settings.ocr_confidence_threshold * 100):
                    text = data['text'][i].strip()
                    if text:
                        text_parts.append(text)
                        confidences.append(int(confidence) / 100.0)
                        
            combined_text = ' '.join(text_parts)
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
            
            # Apply Indonesian text processing
            corrected_text = self.text_processor.correct_text(combined_text)
            quality_score = self.text_processor.validate_quality(corrected_text)
            
            # Adjust confidence based on quality
            final_confidence = (avg_confidence + quality_score) / 2
            
            return OCRResult(
                text=corrected_text,
                confidence=final_confidence,
                engine='tesseract',
                metadata={
                    'raw_confidence': avg_confidence,
                    'quality_score': quality_score,
                    'detected_words': len(text_parts)
                }
            )
            
        except Exception as e:
            logger.error(f"Tesseract extraction failed: {e}")
            raise
            
    def extract_text_markitdown(self, file_path: Union[str, Path]) -> OCRResult:
        """Extract text using MarkItDown (current default)."""
        try:
            result = self.markitdown.convert(str(file_path))
            text_content = result.text_content
            
            # Apply Indonesian text processing
            corrected_text = self.text_processor.correct_text(text_content)
            quality_score = self.text_processor.validate_quality(corrected_text)
            
            return OCRResult(
                text=corrected_text,
                confidence=quality_score,  # Use quality as confidence estimate
                engine='markitdown',
                metadata={
                    'quality_score': quality_score,
                    'original_length': len(text_content),
                    'corrected_length': len(corrected_text)
                }
            )
            
        except Exception as e:
            logger.error(f"MarkItDown extraction failed: {e}")
            raise
    
    def extract_text_pdf_pages(self, file_path: Union[str, Path]) -> OCRResult:
        """Extract text from PDF using page-by-page analysis and smart engine selection."""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Check if it's a PDF file
        if file_path.suffix.lower() != '.pdf':
            logger.info(f"Not a PDF file, using standard extraction: {file_path.name}")
            return self.extract_text_with_fallback(file_path)
        
        # Analyze PDF pages
        logger.info(f"Starting page-by-page PDF analysis for {file_path.name}")
        page_infos = self.pdf_analyzer.analyze_pdf_pages(file_path)
        
        if not page_infos:
            logger.warning("PDF page analysis failed, falling back to standard extraction")
            return self.extract_text_with_fallback(file_path)
        
        # Process pages based on content type
        all_text_parts = []
        total_confidence = 0.0
        engines_used = set()
        page_results = []
        
        for page_info in page_infos:
            try:
                if page_info.content_type == PageContentType.TEXT:
                    # Use MarkItDown for text-based pages
                    result = self._extract_pdf_page_text(file_path, page_info.page_num)
                    logger.info(f"Page {page_info.page_num}: TEXT page processed with MarkItDown")
                    
                elif page_info.content_type == PageContentType.IMAGE:
                    # Use configured OCR engines for image-based pages
                    result = self._extract_pdf_page_ocr(file_path, page_info.page_num)
                    logger.info(f"Page {page_info.page_num}: IMAGE page processed with OCR")
                    
                else:  # MIXED content
                    # Try MarkItDown first, then OCR if confidence is low
                    result = self._extract_pdf_page_mixed(file_path, page_info.page_num)
                    logger.info(f"Page {page_info.page_num}: MIXED page processed")
                
                if result and result.text.strip():
                    all_text_parts.append(f"--- Page {page_info.page_num + 1} ---\n{result.text}")
                    total_confidence += result.confidence
                    engines_used.add(result.engine)
                    page_results.append({
                        'page': page_info.page_num,
                        'content_type': page_info.content_type,
                        'engine': result.engine,
                        'confidence': result.confidence,
                        'text_length': len(result.text)
                    })
                
            except Exception as e:
                logger.warning(f"Failed to process page {page_info.page_num}: {e}")
                continue
        
        if not all_text_parts:
            raise RuntimeError("No text could be extracted from any page")
        
        # Combine results
        combined_text = '\n\n'.join(all_text_parts)
        avg_confidence = total_confidence / len(page_results) if page_results else 0.0
        
        # Apply Indonesian text processing to combined text
        corrected_text = self.text_processor.correct_text(combined_text)
        quality_score = self.text_processor.validate_quality(corrected_text)
        
        # Adjust final confidence
        final_confidence = (avg_confidence + quality_score) / 2
        
        logger.info(f"PDF processing complete: {len(page_results)} pages, engines used: {engines_used}")
        
        return OCRResult(
            text=corrected_text,
            confidence=final_confidence,
            engine=f"page-by-page ({','.join(sorted(engines_used))})",
            metadata={
                'page_count': len(page_infos),
                'processed_pages': len(page_results),
                'engines_used': list(engines_used),
                'avg_confidence': avg_confidence,
                'quality_score': quality_score,
                'page_results': page_results
            }
        )
    
    def _extract_pdf_page_text(self, file_path: Path, page_num: int) -> OCRResult:
        """Extract text from a single PDF page using MarkItDown."""
        try:
            # For now, use MarkItDown on the whole document
            # In a more advanced implementation, we could extract single pages
            result = self.extract_text_markitdown(file_path)
            return result
        except Exception as e:
            logger.error(f"MarkItDown failed for page {page_num}: {e}")
            raise
    
    def _extract_pdf_page_ocr(self, file_path: Path, page_num: int) -> OCRResult:
        """Extract text from a single PDF page using configured OCR engines."""
        try:
            if not PYMUPDF_AVAILABLE:
                raise RuntimeError("PyMuPDF not available for page extraction")
            
            # Extract page as image
            doc = fitz.open(str(file_path))
            page = doc[page_num]
            
            # Render page as image
            mat = fitz.Matrix(2.0, 2.0)  # Scale factor for better quality
            pix = page.get_pixmap(matrix=mat)
            img_data = pix.tobytes("png")
            
            # Save temporarily
            temp_image_path = file_path.parent / f"temp_page_{page_num}.png"
            with open(temp_image_path, "wb") as f:
                f.write(img_data)
            
            doc.close()
            
            try:
                # Use configured OCR engines with fallback
                engines_to_try = self.get_safe_engine_list()
                # Remove MarkItDown from the list for image processing
                ocr_engines = [e for e in engines_to_try if e != 'markitdown']
                
                if not ocr_engines:
                    raise RuntimeError("No OCR engines available for image processing")
                
                last_error = None
                for engine in ocr_engines:
                    try:
                        if engine == 'easyocr':
                            result = self.extract_text_easyocr(temp_image_path)
                        elif engine == 'tesseract':
                            result = self.extract_text_tesseract(temp_image_path)
                        elif engine == 'paddleocr':
                            result = self.extract_text_paddleocr(temp_image_path)
                        else:
                            continue
                        
                        if result.confidence >= settings.ocr_confidence_threshold:
                            return result
                        else:
                            last_error = f"Low confidence: {result.confidence:.3f}"
                            
                    except Exception as e:
                        last_error = str(e)
                        continue
                
                raise RuntimeError(f"All OCR engines failed for page {page_num}. Last error: {last_error}")
                
            finally:
                # Clean up temp file
                if temp_image_path.exists():
                    temp_image_path.unlink()
                    
        except Exception as e:
            logger.error(f"OCR failed for page {page_num}: {e}")
            raise
    
    def _extract_pdf_page_mixed(self, file_path: Path, page_num: int) -> OCRResult:
        """Extract text from a mixed content page (try MarkItDown first, then OCR)."""
        try:
            # Try MarkItDown first
            text_result = self._extract_pdf_page_text(file_path, page_num)
            if text_result.confidence >= settings.ocr_confidence_threshold:
                return text_result
        except Exception:
            pass
        
        # Fall back to OCR
        try:
            return self._extract_pdf_page_ocr(file_path, page_num)
        except Exception as e:
            # If both fail, return the text result with lower confidence
            logger.warning(f"Mixed page {page_num}: OCR failed, using text extraction")
            try:
                return self._extract_pdf_page_text(file_path, page_num)
            except Exception:
                raise e

    def extract_text_with_fallback(self, file_path: Union[str, Path]) -> OCRResult:
        """Extract text using primary engine with intelligent fallback options."""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Determine if file needs image preprocessing
        image_extensions = {'.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif'}
        is_image = file_path.suffix.lower() in image_extensions
        
        # Get safe, working engines in priority order
        engines_to_try = self.get_safe_engine_list()
        
        if not engines_to_try:
            raise RuntimeError("No working OCR engines available")
        
        logger.info(f"Available OCR engines for {file_path.name}: {', '.join(engines_to_try)}")
        
        last_error = None
        
        for engine in engines_to_try:
            try:
                logger.info(f"Attempting OCR with {engine} engine for {file_path.name}")
                
                if engine == 'markitdown':
                    # MarkItDown handles all file types directly
                    result = self.extract_text_markitdown(file_path)
                    
                elif is_image and engine in ['easyocr', 'paddleocr', 'tesseract']:
                    # For images, preprocess first
                    if settings.ocr_enable_preprocessing:
                        processed_image = self.preprocessor.preprocess_image(file_path)
                        # Save temporarily for OCR engines that need file paths
                        temp_path = file_path.parent / f"temp_processed_{file_path.name}"
                        processed_image.save(temp_path)
                        
                        try:
                            if engine == 'easyocr':
                                result = self.extract_text_easyocr(temp_path)
                            elif engine == 'paddleocr':
                                result = self.extract_text_paddleocr(temp_path)
                            elif engine == 'tesseract':
                                result = self.extract_text_tesseract(temp_path)
                        finally:
                            # Clean up temp file
                            if temp_path.exists():
                                temp_path.unlink()
                    else:
                        if engine == 'easyocr':
                            result = self.extract_text_easyocr(file_path)
                        elif engine == 'paddleocr':
                            result = self.extract_text_paddleocr(file_path)
                        elif engine == 'tesseract':
                            result = self.extract_text_tesseract(file_path)
                else:
                    # For non-images or unsupported engines, fall back to MarkItDown
                    result = self.extract_text_markitdown(file_path)
                
                if result.confidence >= settings.ocr_confidence_threshold:
                    logger.info(f"OCR successful with {engine}: confidence={result.confidence:.3f}")
                    return result
                else:
                    logger.warning(f"OCR confidence too low with {engine}: {result.confidence:.3f}")
                    last_error = f"Low confidence: {result.confidence:.3f}"
                    
            except Exception as e:
                logger.warning(f"OCR failed with {engine}: {e}")
                last_error = str(e)
                continue
        
        # If all engines failed, raise the last error
        raise RuntimeError(f"All OCR engines failed. Last error: {last_error}")


# Global instance
enhanced_ocr_engine = EnhancedOCREngine()