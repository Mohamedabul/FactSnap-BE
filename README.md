# FactSnap-V: AI-Powered Audio Analysis Tool

FactSnap-V is a comprehensive AI-powered application that processes spoken content from audio or video files and extracts three key insights:

1. **Emotional Tone Detection** - Analyzes the speaker's emotional state
2. **Linguistic Bias Classification** - Identifies potential bias in speech
3. **Fact Verification** - Validates factual claims using external sources

## Features

### 🎙️ Input Processing
- Supports audio formats: WAV, MP3, M4A, FLAC
- Supports video formats: MP4, AVI, MOV, MKV
- Automatic audio extraction from video files using MoviePy

### 🗣️ Speech-to-Text
- Uses OpenAI Whisper for high-quality, local speech-to-text conversion
- Works completely offline (except fact verification)
- Supports multiple languages with auto-detection

### 😊 Emotion Detection
- Uses Hugging Face transformer models for emotion classification
- Detects emotions: joy, anger, fear, sadness, surprise, love, optimism, and more
- Provides confidence scores for each prediction

### ⚖️ Bias Detection
- Classifies linguistic bias as Low, Medium, or High
- Uses specialized transformer models for bias detection
- Highlights potentially problematic content

### 🔍 Fact Verification
- Integrates with Google Fact Check Tools API
- Validates factual claims against trusted sources
- Returns verification status: True, False, Mixed, or Not Verifiable

### 📊 Interactive Interface
- Built with Streamlit for easy web-based interaction
- Visual charts and graphs for analysis results
- Export results in CSV, JSON, and PDF formats
- Color-coded results for quick identification

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Internet connection (for initial model downloads and fact verification)

### Step 1: Clone or Download
```bash
git clone <repository-url>
cd FactSnap-V
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Download Required Models
```bash
# Download spaCy English model
python -m spacy download en_core_web_sm

# Download NLTK data
python -c "import nltk; nltk.download('punkt')"
```

### Step 4: Install FFmpeg (Optional)
For video processing, install FFmpeg:

**Windows:**
- Download from https://ffmpeg.org/download.html
- Add to PATH environment variable

**macOS:**
```bash
brew install ffmpeg
```

**Linux:**
```bash
sudo apt install ffmpeg
```

## Configuration

The application uses a Google Fact Check Tools API key for fact verification. The key is already configured in `config.py`:

```python
GOOGLE_FACT_CHECK_API_KEY = "AIzaSyCkcwhrM7kVX-rYH22WErrSn2aufkOsuZo"
```

You can modify other settings in `config.py`:
- Whisper model size (tiny, base, small, medium, large)
- Emotion and bias detection models
- Batch size and processing parameters

## Usage

### Web Interface (Recommended)
1. Start the Streamlit application:
```bash
streamlit run app.py
```

2. Open your browser and navigate to the displayed URL (typically http://localhost:8501)

3. Upload an audio or video file using the file uploader

4. Configure analysis options in the sidebar:
   - Enable/disable fact verification
   - Show/hide detailed results

5. Click "Start Analysis" and wait for processing to complete

6. View results in interactive charts and tables

7. Download results in your preferred format

### Command Line Interface
For batch processing or integration with other tools:

```bash
python main.py path/to/your/audio_file.wav
```

Example:
```bash
python main.py sample_speech.mp3
```

Results will be exported to the `output/` directory in multiple formats.

## Project Structure

```
FactSnap-V/
├── config.py              # Configuration settings
├── input_processor.py     # Audio/video input handling
├── speech_to_text.py      # Whisper speech-to-text
├── text_preprocessor.py   # Text segmentation and cleaning
├── emotion_detector.py    # Emotion classification
├── bias_detector.py       # Bias detection
├── fact_verifier.py       # Fact verification with Google API
├── main.py               # Main application orchestrator
├── app.py                # Streamlit web interface
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## Technical Details

### Models Used
- **Speech-to-Text**: OpenAI Whisper (configurable size)
- **Emotion Detection**: `cardiffnlp/twitter-roberta-base-emotion`
- **Bias Detection**: `unitary/toxic-bert`
- **Text Processing**: spaCy English model + NLTK

### Processing Pipeline
1. **Input Processing**: Convert video to audio if needed
2. **Speech-to-Text**: Extract transcript using Whisper
3. **Text Preprocessing**: Segment into sentences, clean text
4. **Emotion Analysis**: Classify emotions for each sentence
5. **Bias Analysis**: Detect linguistic bias levels
6. **Fact Verification**: Validate claims using Google API
7. **Results Compilation**: Generate reports and visualizations

### Privacy and Security
- All AI models run locally except fact verification
- No audio data is sent to external services (except Google Fact Check API for claims)
- Temporary files are automatically cleaned up
- API key is used only for fact verification queries

## Output Formats

### CSV Export
Detailed sentence-by-sentence analysis with columns:
- Sentence text and metadata
- Emotion classification and confidence
- Bias level and score
- Fact verification status and source

### JSON Export
Summary statistics including:
- Overall emotion distribution
- Bias level breakdown
- Fact verification coverage
- Analysis metadata

### Visual Reports
- Emotion distribution pie charts and bar graphs
- Bias level visualization
- Fact verification status charts
- Color-coded detailed results table

## Limitations

1. **Language Support**: Optimized for English, though Whisper supports multiple languages
2. **Fact Verification**: Limited by Google Fact Check Tools API coverage
3. **Processing Time**: Large files may take several minutes to process
4. **Model Accuracy**: Results depend on the quality of underlying AI models
5. **Internet Dependency**: Fact verification requires internet connection

## Troubleshooting

### Common Issues

**1. Model Download Errors**
```bash
# Manually download spaCy model
python -m spacy download en_core_web_sm
```

**2. FFmpeg Not Found**
- Install FFmpeg and ensure it's in your PATH
- Alternatively, use audio files directly

**3. Memory Issues**
- Reduce Whisper model size in `config.py`
- Process shorter audio files
- Increase available RAM

**4. API Rate Limits**
- Fact verification may be rate-limited
- Consider reducing the number of claims extracted

### Performance Optimization

1. **GPU Acceleration**: Install CUDA-compatible PyTorch for GPU processing
2. **Model Size**: Use smaller Whisper models for faster processing
3. **Batch Size**: Adjust batch sizes in `config.py` based on available memory

## Contributing

This project uses open-source technologies and follows modular design principles. Contributions are welcome for:

- Additional language support
- New emotion/bias detection models
- Performance optimizations
- Bug fixes and improvements

## License

This project is built using open-source libraries and models. Please refer to individual component licenses:

- OpenAI Whisper: MIT License
- Hugging Face Transformers: Apache 2.0
- Streamlit: Apache 2.0
- Other dependencies: See respective licenses

## Acknowledgments

- OpenAI for the Whisper speech-to-text model
- Hugging Face for transformer models and infrastructure
- Google for the Fact Check Tools API
- Cardiff NLP for emotion detection models
- The open-source community for supporting libraries

---

**Built with ❤️ using Python and Open Source AI**
