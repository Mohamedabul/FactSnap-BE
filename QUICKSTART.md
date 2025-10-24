# FactSnap-V Quick Start Guide

## 🚀 Get Started in 5 Minutes

### Step 1: Installation Complete ✅
The setup has been completed successfully! All components are installed and ready.

### Step 2: Start the Web Interface
Run this command to start the Streamlit web interface:

```bash
streamlit run app.py
```

The interface will open in your browser at: http://localhost:8501

### Step 3: Upload and Analyze
1. **Upload a file**: Choose an audio (.wav, .mp3) or video (.mp4) file
2. **Configure options**: Enable/disable fact verification in the sidebar
3. **Start analysis**: Click "🚀 Start Analysis" and wait for processing
4. **View results**: See emotion analysis, bias detection, and fact verification
5. **Export data**: Download results as CSV or JSON

## 📁 Sample Workflow

### For Audio Files
```bash
# Upload: speech.wav → FactSnap-V → Results
```

### For Video Files (Note: FFmpeg not installed)
Currently video processing is disabled. To enable:
1. Install FFmpeg from https://ffmpeg.org/download.html
2. Add FFmpeg to your system PATH
3. Restart the application

## 🔧 Command Line Usage
For batch processing:

```bash
python main.py path/to/audio_file.wav
```

Results will be saved in the `output/` directory.

## 📊 What You'll Get

### Emotion Analysis
- Joy, anger, fear, sadness, surprise detection
- Confidence scores for each emotion
- Visual charts and distribution graphs

### Bias Detection
- Low/Medium/High bias classification
- Sentence-level bias scoring
- Color-coded results for quick identification

### Fact Verification
- Claims extracted from speech
- Verification against trusted sources
- True/False/Mixed/Not Verifiable status

## 🎯 Key Features Working
✅ **Speech-to-Text**: OpenAI Whisper (local processing)  
✅ **Emotion Detection**: Hugging Face transformers  
✅ **Bias Classification**: AI-powered bias detection  
✅ **Fact Verification**: Google Fact Check Tools API  
✅ **Web Interface**: Interactive Streamlit dashboard  
✅ **Export Options**: CSV, JSON, and visual reports  

⚠️ **Video Processing**: Requires FFmpeg installation

## 🔍 Test Files
To test the application, you can:
1. Record a short audio message on your phone
2. Use any .wav or .mp3 file you have
3. Convert a video to audio using online tools

## 📈 Performance Tips
- **Small files**: Process faster (under 30 seconds)
- **Large files**: May take several minutes
- **Model loading**: First run takes longer as models download
- **GPU acceleration**: Install CUDA-compatible PyTorch for faster processing

## 🆘 Need Help?

### Common Issues
- **Model loading errors**: Restart the application
- **Memory issues**: Use smaller audio files or increase RAM
- **API rate limits**: Wait between fact verification requests

### Support Resources
- `README.md`: Detailed documentation
- `test_installation.py`: Verify installation
- `setup.py`: Reinstall dependencies

## 🎉 Ready to Analyze!
Your FactSnap-V installation is ready. Start the web interface with:

```bash
streamlit run app.py
```

Upload an audio file and explore the AI-powered insights!
