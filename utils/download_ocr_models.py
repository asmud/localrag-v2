#!/usr/bin/env python3
"""
Utility script to pre-download OCR models for LocalRAG.
"""

import os
import sys
import tempfile
import time
from pathlib import Path

# Add the parent directory to sys.path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger

# Import check utilities
try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False

try:
    import paddleocr
    PADDLEOCR_AVAILABLE = True
except ImportError:
    PADDLEOCR_AVAILABLE = False

try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False


def validate_easyocr_models(languages=['en', 'id']):
    """Validate if EasyOCR models exist and are functional."""
    try:
        # Check if models directory exists
        models_dir = Path("models/ocr/easyocr")
        if not models_dir.exists():
            logger.info("EasyOCR models directory does not exist")
            return False
        
        # Check for expected model files (including nested structure)
        model_files = list(models_dir.glob("**/*.pth"))
        if not model_files:
            logger.info("No EasyOCR model files found")
            return False
        
        # Check if cache info exists
        cache_info = models_dir / ".model_cache_info"
        if cache_info.exists():
            logger.info("✅ EasyOCR models validation passed")
            return True
        
        # If models exist but no cache info, consider them valid but create cache info
        logger.info(f"EasyOCR models found ({len(model_files)} files), creating cache info")
        
        # Create cache info file since models exist
        cache_info.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_info, 'w') as f:
            f.write("EasyOCR models found during validation\n")
            f.write(f"Languages: {languages}\n")
            f.write(f"Timestamp: {time.time()}\n")
            f.write(f"Model files: {len(model_files)}\n")
        
        logger.info("✅ EasyOCR models validation passed")
        return True
        
    except Exception as e:
        logger.warning(f"EasyOCR model validation failed: {e}")
        return False


def download_easyocr_models(languages=['en', 'id'], gpu=False, force_download=False):
    """Download EasyOCR models for specified languages."""
    if not EASYOCR_AVAILABLE:
        logger.warning("EasyOCR not available. Install with: pip install easyocr")
        return False
    
    # Skip download if models already exist and are valid
    if not force_download and validate_easyocr_models(languages):
        logger.info("✅ EasyOCR models already exist and are valid, skipping download")
        return True
        
    try:
        logger.info("🔄 Downloading EasyOCR models...")
        logger.info(f"Languages: {languages}")
        logger.info(f"GPU enabled: {gpu}")
        
        # Initialize EasyOCR Reader - this will download models
        reader = easyocr.Reader(languages, gpu=gpu)
        
        # Copy models to our persistent directory
        models_dir = Path("models/ocr/easyocr")
        models_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy from ~/.EasyOCR to our models directory
        home_easyocr = Path.home() / ".EasyOCR"
        if home_easyocr.exists():
            import shutil
            for item in home_easyocr.rglob("*"):
                if item.is_file():
                    relative_path = item.relative_to(home_easyocr)
                    dest_path = models_dir / relative_path
                    dest_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(item, dest_path)
        
        # Create cache info file
        cache_info = models_dir / ".model_cache_info"
        with open(cache_info, 'w') as f:
            f.write("EasyOCR models downloaded\n")
            f.write(f"Languages: {languages}\n")
            f.write(f"Timestamp: {time.time()}\n")
        
        logger.info("✅ EasyOCR models downloaded successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to download EasyOCR models: {e}")
        return False


def validate_paddleocr_models(lang='en'):
    """Validate if PaddleOCR models exist and are functional."""
    try:
        # Check if models directory exists
        models_dir = Path("models/ocr/paddleocr")
        if not models_dir.exists():
            logger.info("PaddleOCR models directory does not exist")
            return False
        
        # Check if cache info exists
        cache_info = models_dir / ".model_cache_info"
        if cache_info.exists():
            logger.info("✅ PaddleOCR models validation passed")
            return True
        
        logger.info("PaddleOCR models directory exists but no cache info found")
        return False
        
    except Exception as e:
        logger.warning(f"PaddleOCR model validation failed: {e}")
        return False


def download_paddleocr_models(lang='en', force_download=False):
    """Download PaddleOCR models."""
    if not PADDLEOCR_AVAILABLE:
        logger.warning("PaddleOCR not available. Install with: pip install paddleocr")
        return False
    
    # Skip download if models already exist and are valid
    if not force_download and validate_paddleocr_models(lang):
        logger.info("✅ PaddleOCR models already exist and are valid, skipping download")
        return True
        
    try:
        logger.info("🔄 Downloading PaddleOCR models...")
        logger.info(f"Language: {lang}")
        
        # Initialize PaddleOCR - this will download models
        ocr = paddleocr.PaddleOCR(use_angle_cls=True, lang=lang)
        
        # Create models directory and cache info
        models_dir = Path("models/ocr/paddleocr")
        models_dir.mkdir(parents=True, exist_ok=True)
        
        # Create cache info file
        cache_info = models_dir / ".model_cache_info"
        with open(cache_info, 'w') as f:
            f.write("PaddleOCR models downloaded\n")
            f.write(f"Language: {lang}\n")
            f.write(f"Timestamp: {time.time()}\n")
        
        logger.info("✅ PaddleOCR models downloaded successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to download PaddleOCR models: {e}")
        return False


def check_tesseract_languages():
    """Check available Tesseract languages."""
    if not TESSERACT_AVAILABLE:
        logger.warning("Tesseract not available. Install with: apt-get install tesseract-ocr")
        return False
        
    try:
        # Get available languages
        langs = pytesseract.get_languages()
        logger.info(f"📋 Available Tesseract languages: {langs}")
        
        # Check for Indonesian
        if 'ind' in langs:
            logger.info("✅ Indonesian language pack available")
        else:
            logger.warning("⚠️  Indonesian language pack not found. Install with: apt-get install tesseract-ocr-ind")
            
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to check Tesseract languages: {e}")
        return False


def validate_all_models(languages=['en', 'id']):
    """Validate all OCR models exist and are functional."""
    logger.info("🔍 Validating OCR models")
    logger.info("=" * 60)
    
    results = {}
    
    # Validate EasyOCR models
    logger.info("\n🔍 EasyOCR Models")
    logger.info("-" * 30)
    results['easyocr'] = validate_easyocr_models(languages)
    
    # Validate PaddleOCR models
    logger.info("\n🔍 PaddleOCR Models")
    logger.info("-" * 30)
    results['paddleocr'] = validate_paddleocr_models('en')
    
    # Check Tesseract
    logger.info("\n📋 Tesseract Check")
    logger.info("-" * 30)
    results['tesseract'] = check_tesseract_languages()
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("📊 Validation Summary")
    logger.info("=" * 60)
    
    for engine, success in results.items():
        status = "✅ Valid" if success else "❌ Missing/Invalid"
        logger.info(f"{engine:12}: {status}")
    
    total_valid = sum(results.values())
    logger.info(f"\nOverall: {total_valid}/{len(results)} engines valid")
    
    return total_valid == len(results)


def download_all_models(languages=['en', 'id'], gpu=False, force_download=False):
    """Download all available OCR models."""
    logger.info("🚀 Starting OCR model download process")
    logger.info("=" * 60)
    
    results = {}
    
    # Download EasyOCR models
    logger.info("\n📥 EasyOCR Models")
    logger.info("-" * 30)
    results['easyocr'] = download_easyocr_models(languages, gpu, force_download)
    
    # Download PaddleOCR models  
    logger.info("\n📥 PaddleOCR Models")
    logger.info("-" * 30)
    results['paddleocr'] = download_paddleocr_models('en', force_download)
    
    # Check Tesseract
    logger.info("\n📋 Tesseract Check")
    logger.info("-" * 30)
    results['tesseract'] = check_tesseract_languages()
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("📊 Download Summary")
    logger.info("=" * 60)
    
    for engine, success in results.items():
        status = "✅ Success" if success else "❌ Failed"
        logger.info(f"{engine:12}: {status}")
    
    total_success = sum(results.values())
    logger.info(f"\nOverall: {total_success}/{len(results)} engines ready")
    
    if total_success > 0:
        logger.info("🎉 OCR models download completed!")
        logger.info("\n📝 Next steps:")
        logger.info("1. Rebuild your Docker container")
        logger.info("2. Models will be available without download delays")
        return True
    else:
        logger.error("💥 All model downloads failed")
        return False


def create_model_cache_info():
    """Create a file indicating models have been pre-downloaded."""
    try:
        import datetime
        
        models_dir = Path("models/ocr")
        models_dir.mkdir(parents=True, exist_ok=True)
        
        cache_info = models_dir / ".model_cache_info"
        with open(cache_info, 'w') as f:
            f.write("OCR models pre-downloaded\n")
            f.write("Timestamp: " + str(datetime.datetime.now()) + "\n")
            
        logger.info(f"✅ Created model cache info: {cache_info}")
        return True
        
    except Exception as e:
        logger.warning(f"Failed to create cache info: {e}")
        return False


def check_model_cache():
    """Check if models have been pre-downloaded."""
    cache_info = Path("models/ocr/.model_cache_info")
    if cache_info.exists():
        logger.info("✅ Model cache info found - models may be pre-downloaded")
        return True
    else:
        logger.info("ℹ️  No model cache info found")
        return False


def clean_model_cache():
    """Clean model cache to force re-download."""
    try:
        cache_dirs = [
            Path.home() / ".EasyOCR",
            Path.home() / ".paddleocr",
            Path("models/ocr")
        ]
        
        for cache_dir in cache_dirs:
            if cache_dir.exists():
                import shutil
                shutil.rmtree(cache_dir)
                logger.info(f"🗑️  Cleaned cache: {cache_dir}")
        
        logger.info("✅ Model cache cleaned")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to clean cache: {e}")
        return False


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Download OCR models for LocalRAG")
    parser.add_argument("--languages", nargs="+", default=["en", "id"], 
                       help="Languages to download (default: en id)")
    parser.add_argument("--gpu", action="store_true", 
                       help="Enable GPU support")
    parser.add_argument("--check-cache", action="store_true",
                       help="Check if models are already cached")
    parser.add_argument("--validate-models", action="store_true",
                       help="Validate existing models without downloading")
    parser.add_argument("--clean-cache", action="store_true",
                       help="Clean model cache to force re-download")
    parser.add_argument("--force-download", action="store_true",
                       help="Force download even if models exist")
    parser.add_argument("--easyocr-only", action="store_true",
                       help="Download only EasyOCR models")
    parser.add_argument("--paddleocr-only", action="store_true",
                       help="Download only PaddleOCR models")
    
    args = parser.parse_args()
    
    logger.info("🤖 LocalRAG OCR Model Downloader")
    logger.info("=" * 60)
    
    if args.check_cache:
        check_model_cache()
        return
    
    if args.validate_models:
        validate_all_models(args.languages)
        return
    
    if args.clean_cache:
        clean_model_cache()
        return
    
    if args.easyocr_only:
        logger.info("📥 Downloading EasyOCR models only...")
        download_easyocr_models(args.languages, args.gpu, args.force_download)
        return
    
    if args.paddleocr_only:
        logger.info("📥 Downloading PaddleOCR models only...")
        download_paddleocr_models('en', args.force_download)
        return
    
    # Download all models
    success = download_all_models(args.languages, args.gpu, args.force_download)
    
    if success:
        create_model_cache_info()
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()