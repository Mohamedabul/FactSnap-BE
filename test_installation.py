"""
Test script for FactSnap-V
Verifies that all components can be imported and initialized
"""

import sys
import traceback


def test_imports():
    """
    Test importing all required modules
    """
    print("Testing module imports...")
    
    modules = {
        'streamlit': 'Streamlit web framework',
        'whisper': 'OpenAI Whisper speech-to-text',
        'transformers': 'Hugging Face transformers',
        'torch': 'PyTorch deep learning framework',
        'spacy': 'spaCy NLP library',
        'nltk': 'NLTK natural language toolkit',
        'moviepy': 'MoviePy video processing',
        'requests': 'HTTP requests library',
        'pandas': 'Pandas data analysis',
        'numpy': 'NumPy numerical computing',
        'matplotlib': 'Matplotlib plotting',
        'seaborn': 'Seaborn statistical visualization'
    }
    
    failed_imports = []
    
    for module, description in modules.items():
        try:
            __import__(module)
            print(f"✓ {module:15} - {description}")
        except ImportError as e:
            print(f"✗ {module:15} - Failed: {str(e)}")
            failed_imports.append(module)
    
    return len(failed_imports) == 0


def test_component_initialization():
    """
    Test initializing individual FactSnap-V components
    """
    print("\nTesting component initialization...")
    
    components = [
        ('InputProcessor', 'input_processor', 'InputProcessor'),
        ('TextPreprocessor', 'text_preprocessor', 'TextPreprocessor'),
        ('EmotionDetector', 'emotion_detector', 'EmotionDetector'),
        ('BiasDetector', 'bias_detector', 'BiasDetector'),
        ('FactVerifier', 'fact_verifier', 'FactVerifier'),
    ]
    
    failed_components = []
    
    for name, module_name, class_name in components:
        try:
            module = __import__(module_name)
            component_class = getattr(module, class_name)
            
            # Try to initialize (but don't load heavy models for tests)
            if name in ['EmotionDetector', 'BiasDetector']:
                print(f"⚠️  {name:18} - Skipped (requires model download)")
            else:
                instance = component_class()
                print(f"✓ {name:18} - Initialized successfully")
                
                # Clean up if needed
                if hasattr(instance, 'cleanup'):
                    instance.cleanup()
                    
        except Exception as e:
            print(f"✗ {name:18} - Failed: {str(e)}")
            failed_components.append(name)
    
    return len(failed_components) == 0


def test_whisper_availability():
    """
    Test if Whisper models can be loaded
    """
    print("\nTesting Whisper availability...")
    
    try:
        import whisper
        
        # Try to load the smallest model
        print("  Loading Whisper 'tiny' model...")
        model = whisper.load_model("tiny")
        print("✓ Whisper model loaded successfully")
        
        # Test with dummy audio (silence)
        import numpy as np
        dummy_audio = np.zeros(16000)  # 1 second of silence at 16kHz
        
        result = model.transcribe(dummy_audio)
        print(f"✓ Whisper transcription test: '{result['text'].strip()}'")
        
        return True
        
    except Exception as e:
        print(f"✗ Whisper test failed: {str(e)}")
        return False


def test_spacy_model():
    """
    Test if spaCy English model is available
    """
    print("\nTesting spaCy English model...")
    
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
        
        # Test with sample text
        doc = nlp("This is a test sentence.")
        sentences = [sent.text for sent in doc.sents]
        
        print(f"✓ spaCy model loaded, sentences: {sentences}")
        return True
        
    except Exception as e:
        print(f"✗ spaCy model test failed: {str(e)}")
        print("  Run: python -m spacy download en_core_web_sm")
        return False


def test_nltk_data():
    """
    Test if NLTK punkt tokenizer is available
    """
    print("\nTesting NLTK punkt tokenizer...")
    
    try:
        import nltk
        from nltk.tokenize import sent_tokenize
        
        # Test tokenization
        text = "This is sentence one. This is sentence two!"
        sentences = sent_tokenize(text)
        
        print(f"✓ NLTK punkt tokenizer working, sentences: {sentences}")
        return True
        
    except Exception as e:
        print(f"✗ NLTK test failed: {str(e)}")
        print("  Run: python -c \"import nltk; nltk.download('punkt')\"")
        return False


def test_api_connectivity():
    """
    Test if Google Fact Check API is accessible
    """
    print("\nTesting Google Fact Check API connectivity...")
    
    try:
        import requests
        from config import GOOGLE_FACT_CHECK_API_KEY, GOOGLE_FACT_CHECK_API_URL
        
        # Simple test query
        params = {
            'key': GOOGLE_FACT_CHECK_API_KEY,
            'query': 'test',
            'languageCode': 'en'
        }
        
        response = requests.get(GOOGLE_FACT_CHECK_API_URL, params=params, timeout=5)
        
        if response.status_code == 200:
            print("✓ Google Fact Check API is accessible")
            return True
        else:
            print(f"⚠️  Google Fact Check API returned status {response.status_code}")
            return False
            
    except Exception as e:
        print(f"✗ API connectivity test failed: {str(e)}")
        return False


def main():
    """
    Run all tests
    """
    print("="*60)
    print("FactSnap-V Installation Test")
    print("="*60)
    
    tests = [
        ("Module Imports", test_imports),
        ("Component Initialization", test_component_initialization),
        ("Whisper Availability", test_whisper_availability),
        ("spaCy Model", test_spacy_model),
        ("NLTK Data", test_nltk_data),
        ("API Connectivity", test_api_connectivity)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"\n✗ {test_name} crashed: {str(e)}")
            print(traceback.format_exc())
            results[test_name] = False
    
    # Summary
    print("\n" + "="*60)
    print("Test Results Summary")
    print("="*60)
    
    passed = 0
    total = len(tests)
    
    for test_name, passed_test in results.items():
        status = "✓ PASSED" if passed_test else "✗ FAILED"
        print(f"{test_name:25} {status}")
        if passed_test:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! FactSnap-V is ready to use.")
        print("\nYou can now run:")
        print("  streamlit run app.py")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please check the installation.")
        print("\nFor help, see README.md or run:")
        print("  python setup.py")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
