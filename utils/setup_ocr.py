#!/usr/bin/env python3
"""
Setup script for enhanced OCR dependencies.
"""

import subprocess
import sys
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ {description} completed successfully")
            return True
        else:
            print(f"❌ {description} failed:")
            print(result.stderr)
            return False
    except Exception as e:
        print(f"❌ {description} failed with exception: {e}")
        return False

def install_ocr_dependencies():
    """Install OCR dependencies."""
    print("🚀 Setting up Enhanced OCR Dependencies")
    print("=" * 50)
    
    # Update pip first
    run_command(f"{sys.executable} -m pip install --upgrade pip", "Updating pip")
    
    # Install basic image processing dependencies
    basic_deps = [
        "pillow>=10.0.0",
        "opencv-python>=4.8.0",
        "pytesseract>=0.3.10"
    ]
    
    for dep in basic_deps:
        run_command(f"{sys.executable} -m pip install {dep}", f"Installing {dep}")
    
    # Install EasyOCR (recommended)
    print("\n📋 Installing EasyOCR (recommended for Indonesian documents)...")
    if run_command(f"{sys.executable} -m pip install easyocr>=1.7.0", "Installing EasyOCR"):
        print("✅ EasyOCR installed successfully")
    else:
        print("⚠️  EasyOCR installation failed - will use fallback engines")
    
    # Install PaddleOCR (optional)
    print("\n📋 Installing PaddleOCR (alternative engine)...")
    paddle_commands = [
        f"{sys.executable} -m pip install paddlepaddle>=2.6.0",
        f"{sys.executable} -m pip install paddleocr>=2.7.0"
    ]
    
    paddle_success = True
    for cmd in paddle_commands:
        if not run_command(cmd, "Installing PaddleOCR dependencies"):
            paddle_success = False
            break
    
    if paddle_success:
        print("✅ PaddleOCR installed successfully")
    else:
        print("⚠️  PaddleOCR installation failed - will use other engines")
    
    print("\n" + "=" * 50)
    print("🎉 OCR setup completed!")
    print("\nAvailable OCR engines:")
    print("  - MarkItDown (built-in, always available)")
    print("  - Tesseract (basic OCR, requires system installation)")
    if paddle_success:
        print("  - PaddleOCR ✅")
    print("  - EasyOCR ✅ (recommended)")
    
    print("\n📖 Usage:")
    print("  Set OCR_ENGINE=easyocr in your .env file for best Indonesian results")
    print("  Set OCR_FALLBACK_ENGINE=tesseract for reliability")

def check_system_dependencies():
    """Check for required system dependencies."""
    print("🔍 Checking system dependencies...")
    
    # Check for Tesseract
    try:
        result = subprocess.run(["tesseract", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Tesseract OCR found")
            return True
        else:
            print("⚠️  Tesseract OCR not found")
            print("   Install with: apt-get install tesseract-ocr tesseract-ocr-ind (Ubuntu)")
            print("   or: brew install tesseract tesseract-lang (macOS)")
            return False
    except FileNotFoundError:
        print("⚠️  Tesseract OCR not found in PATH")
        print("   Install Tesseract system package for better OCR support")
        return False

def main():
    """Main setup function."""
    print("🎯 Enhanced OCR Setup for LocalRAG")
    print("=" * 50)
    
    # Check system dependencies
    check_system_dependencies()
    
    print("\n")
    
    # Install Python dependencies
    install_ocr_dependencies()
    
    print("\n📝 Next steps:")
    print("1. Configure OCR settings in your .env file:")
    print("   OCR_ENGINE=easyocr")
    print("   OCR_FALLBACK_ENGINE=tesseract")
    print("   OCR_LANGUAGES=[\"en\", \"id\"]")
    print("   OCR_GPU_ENABLED=true")
    print("")
    print("2. Test the OCR setup with:")
    print("   python utils/test_ocr.py")
    print("")
    print("3. Rebuild your Docker container to include new dependencies")

if __name__ == "__main__":
    main()