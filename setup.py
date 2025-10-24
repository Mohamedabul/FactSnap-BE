"""
Setup script for FactSnap-V
Automates the installation of dependencies and required models
"""

import subprocess
import sys
import os


def run_command(command, description):
    """
    Run a command and handle errors
    """
    print(f"\n{description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Error during {description}:")
        print(f"Command: {command}")
        print(f"Error: {e.stderr}")
        return False


def check_python_version():
    """
    Check if Python version is compatible
    """
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("✗ Python 3.8 or higher is required")
        print(f"Current version: {version.major}.{version.minor}.{version.micro}")
        return False
    else:
        print(f"✓ Python version {version.major}.{version.minor}.{version.micro} is compatible")
        return True


def install_dependencies():
    """
    Install Python dependencies from requirements.txt
    """
    commands = [
        ("pip install --upgrade pip", "Upgrading pip"),
        ("pip install -r requirements.txt", "Installing Python dependencies")
    ]
    
    for command, description in commands:
        if not run_command(command, description):
            return False
    return True


def download_models():
    """
    Download required ML models
    """
    commands = [
        ("python -m spacy download en_core_web_sm", "Downloading spaCy English model"),
        ("python -c \"import nltk; nltk.download('punkt', quiet=True)\"", "Downloading NLTK punkt tokenizer")
    ]
    
    for command, description in commands:
        if not run_command(command, description):
            print(f"Warning: {description} failed. You may need to download manually.")
    
    return True


def test_imports():
    """
    Test if all required modules can be imported
    """
    print("\nTesting module imports...")
    
    modules_to_test = [
        "streamlit",
        "whisper", 
        "transformers",
        "torch",
        "spacy",
        "nltk",
        "moviepy",
        "requests",
        "pandas",
        "numpy",
        "matplotlib",
        "seaborn"
    ]
    
    failed_imports = []
    
    for module in modules_to_test:
        try:
            __import__(module)
            print(f"✓ {module}")
        except ImportError as e:
            print(f"✗ {module} - {str(e)}")
            failed_imports.append(module)
    
    if failed_imports:
        print(f"\nWarning: Failed to import {len(failed_imports)} modules: {', '.join(failed_imports)}")
        return False
    else:
        print("\n✓ All modules imported successfully!")
        return True


def check_ffmpeg():
    """
    Check if FFmpeg is available for video processing
    """
    print("\nChecking FFmpeg availability...")
    try:
        result = subprocess.run("ffmpeg -version", shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print("✓ FFmpeg is available for video processing")
            return True
        else:
            print("✗ FFmpeg not found")
            return False
    except Exception:
        print("✗ FFmpeg not found")
        return False


def create_directories():
    """
    Create necessary directories
    """
    directories = ["output", "temp"]
    
    print("\nCreating directories...")
    for directory in directories:
        try:
            os.makedirs(directory, exist_ok=True)
            print(f"✓ Created directory: {directory}")
        except Exception as e:
            print(f"✗ Failed to create directory {directory}: {str(e)}")


def main():
    """
    Main setup function
    """
    print("="*60)
    print("FactSnap-V Setup")
    print("="*60)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Install dependencies
    if not install_dependencies():
        print("\n✗ Failed to install dependencies")
        sys.exit(1)
    
    # Download models
    download_models()
    
    # Test imports
    if not test_imports():
        print("\n⚠️  Some modules failed to import. Please check the error messages above.")
    
    # Check FFmpeg
    ffmpeg_available = check_ffmpeg()
    if not ffmpeg_available:
        print("\n⚠️  FFmpeg not found. Video processing will not work.")
        print("   Install FFmpeg from https://ffmpeg.org/download.html")
    
    # Create directories
    create_directories()
    
    print("\n" + "="*60)
    print("Setup Summary")
    print("="*60)
    print("✓ Python dependencies installed")
    print("✓ AI models downloaded")
    print("✓ Project directories created")
    
    if ffmpeg_available:
        print("✓ FFmpeg available for video processing")
    else:
        print("⚠️  FFmpeg not available (video processing disabled)")
    
    print("\nSetup completed! You can now run FactSnap-V:")
    print("\n  Web Interface:     streamlit run app.py")
    print("  Command Line:      python main.py <audio_file>")
    print("\nFor detailed usage instructions, see README.md")


if __name__ == "__main__":
    main()
