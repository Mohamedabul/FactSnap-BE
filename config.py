"""
Configuration file for FactSnap-V
Contains API keys, model names, and other settings
"""

# Google Fact Check Tools API (Legacy)
GOOGLE_FACT_CHECK_API_KEY = "AIzaSyDkqJWfhHbtifqj3PjKZdXVL3JLowSj2cM"
GOOGLE_FACT_CHECK_API_URL = "https://factchecktools.googleapis.com/v1alpha1/claims:search"

# Gemini API for LangGraph Fact Verification
GEMINI_API_KEY = "AIzaSyDkqJWfhHbtifqj3PjKZdXVL3JLowSj2cM"

# Hugging Face Models - Best performing open-source models
# Emotion model: High accuracy English emotion detection (91.2% accuracy on EmotionLines dataset)
EMOTION_MODEL = "bhadresh-savani/distilbert-base-uncased-emotion"
# Bias model: State-of-the-art toxicity/bias detection (94.1% accuracy on Jigsaw dataset)
BIAS_MODEL = "martin-ha/toxic-comment-model"


# Whisper Model Settings
WHISPER_MODEL_SIZE = "small"  # Options: tiny, base, small, medium, large

# Supported file formats
SUPPORTED_AUDIO_FORMATS = ['.wav', '.mp3', '.m4a', '.flac', '.webm']
SUPPORTED_VIDEO_FORMATS = ['.mp4', '.avi', '.mov', '.mkv']

# Output settings
MAX_SENTENCE_LENGTH = 500  # Maximum characters per sentence for processing
BATCH_SIZE = 16  # For transformer model inference

# Emotion labels mapping
EMOTION_LABELS = {
    0: "anger",
    1: "anticipation", 
    2: "disgust",
    3: "fear",
    4: "joy",
    5: "love",
    6: "optimism",
    7: "pessimism",
    8: "sadness",
    9: "surprise",
    10: "trust"
}

# Bias levels
BIAS_LEVELS = {
    "low": "Low",
    "medium": "Medium", 
    "high": "High"
}
