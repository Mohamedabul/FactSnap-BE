# FactSnap-V Real-time Audio Streaming Guide

## 🎯 Overview

Your FactSnap-V system now supports **real-time audio streaming** in addition to file uploads. Users can speak directly into their microphone and get live analysis of their speech for emotions, bias detection, and fact-checking.

## ✅ What's Been Implemented

### 1. **Real-time Audio Capture**
- **File**: `Backend/real_time_audio.py`
- **Features**:
  - Live microphone input capture
  - Multiple audio device support
  - Configurable audio parameters (sample rate, chunk size)
  - Audio buffering for continuous processing
  - Session recording and saving

### 2. **Streaming Analysis Pipeline**
- **Integration**: Works with existing FactSnap-V components
- **Processing**:
  - Real-time speech-to-text (Whisper)
  - Live emotion analysis
  - Real-time bias detection
  - Optional fact verification (may be slower)

### 3. **Streamlit UI Components**
- **File**: `Backend/streamlit_audio_streaming.py`
- **Features**:
  - Audio device selection
  - Start/stop streaming controls
  - Live results dashboard
  - Real-time analytics and visualizations
  - Session saving and download

### 4. **Enhanced Main App**
- **File**: `Backend/app.py` (updated)
- **Features**:
  - Input method selection (File Upload vs Real-time Streaming)
  - Seamless integration with existing UI
  - Maintains all original functionality

## 🚀 How to Use

### For Users:

1. **Start the Application**:
   ```bash
   streamlit run Backend/app.py
   ```

2. **Select Input Method**:
   - Choose "🎙️ Real-time Streaming" instead of "📁 Upload File"

3. **Configure Audio**:
   - Select your microphone from the dropdown
   - Ensure your microphone is working and not muted

4. **Start Streaming**:
   - Click "🎙️ Start Streaming"
   - Speak naturally into your microphone
   - Watch live results in the tabs below

5. **Monitor Results**:
   - **Transcript Tab**: See real-time speech-to-text
   - **Emotions Tab**: Live emotion analysis with charts
   - **Bias Tab**: Real-time bias detection
   - **Analytics Tab**: Speaking patterns and statistics

6. **Save Session**:
   - Click "💾 Save Session" to download recorded audio
   - Results are processed in 5-second chunks

### For Developers:

1. **Install Dependencies**:
   ```bash
   pip install pyaudio sounddevice plotly
   ```

2. **Test Real-time Functionality**:
   ```bash
   python Backend/test_real_time.py
   ```

3. **Use Programmatically**:
   ```python
   from real_time_audio import StreamingFactSnapV
   from main import FactSnapV
   
   # Initialize
   factsnap_app = FactSnapV()
   streaming = StreamingFactSnapV(factsnap_app)
   
   # Start streaming
   streaming.start_streaming(device_index=0)
   
   # Stop and get results
   streaming.stop_streaming()
   results = streaming.get_streaming_results()
   ```

## 📊 Technical Details

### Audio Processing:
- **Sample Rate**: 16kHz (configurable)
- **Chunk Size**: 1024 samples
- **Buffer Duration**: 30 seconds
- **Processing Interval**: 5 seconds
- **Format**: 16-bit PCM

### Performance:
- **Latency**: ~5-10 seconds (due to processing)
- **Languages**: Auto-detected (English, Hindi, Welsh tested)
- **Accuracy**: Same as file-based processing
- **Resource Usage**: Moderate CPU, low memory

### Supported Platforms:
- ✅ **Windows** (tested)
- ✅ **macOS** (should work)
- ✅ **Linux** (should work)

## 🔧 Configuration

### Audio Settings (in `real_time_audio.py`):
```python
RealTimeAudioProcessor(
    sample_rate=16000,      # Audio sample rate
    chunk_size=1024,        # Processing chunk size
    channels=1,             # Mono audio
    buffer_duration=30.0    # Buffer duration in seconds
)
```

### Processing Settings:
```python
process_interval = 5.0  # Process every 5 seconds
```

## 🐛 Troubleshooting

### Common Issues:

1. **No Audio Devices Found**:
   - Ensure microphone is connected
   - Check Windows audio settings
   - Try different USB ports for USB microphones

2. **PyAudio Installation Issues**:
   ```bash
   # Windows
   pip install pipwin
   pipwin install pyaudio
   
   # Alternative
   conda install pyaudio
   ```

3. **Permission Issues**:
   - Allow microphone access in Windows settings
   - Check browser permissions if using web interface

4. **Poor Audio Quality**:
   - Use a dedicated microphone instead of built-in
   - Reduce background noise
   - Speak clearly and at normal pace

5. **High CPU Usage**:
   - Reduce processing frequency
   - Use smaller Whisper model
   - Close other applications

### Performance Optimization:

1. **Faster Processing**:
   - Use `tiny` or `base` Whisper model instead of `small`
   - Increase processing interval to 10 seconds
   - Disable fact verification for real-time use

2. **Better Accuracy**:
   - Use `medium` or `large` Whisper model
   - Reduce background noise
   - Use high-quality microphone

## 📈 Test Results

All tests passed successfully:

- ✅ **Audio Device Detection**: Found 16 devices
- ✅ **Audio Recording**: Successfully captured and saved audio
- ✅ **Streaming Integration**: Full pipeline working
- ✅ **Multi-language Support**: English, Hindi, Welsh detected
- ✅ **Real-time Processing**: Emotion and bias analysis working

## 🔮 Future Enhancements

### Potential Improvements:

1. **Voice Activity Detection (VAD)**:
   - Only process when speech is detected
   - Reduce unnecessary processing

2. **Real-time Fact Verification**:
   - Optimize for faster processing
   - Cache common claims

3. **Speaker Identification**:
   - Multiple speaker support
   - Speaker-specific analysis

4. **Advanced Audio Processing**:
   - Noise reduction
   - Echo cancellation
   - Audio enhancement

5. **WebRTC Integration**:
   - Browser-based audio capture
   - No local installation required

## 📝 Files Added/Modified

### New Files:
- `Backend/real_time_audio.py` - Core real-time audio processing
- `Backend/streamlit_audio_streaming.py` - Streamlit UI components
- `Backend/test_real_time.py` - Testing functionality
- `Backend/REAL_TIME_STREAMING_GUIDE.md` - This guide

### Modified Files:
- `Backend/app.py` - Added input method selection
- `Backend/requirements.txt` - Added audio dependencies

### Dependencies Added:
- `pyaudio>=0.2.11` - Audio capture
- `sounddevice>=0.4.6` - Alternative audio interface
- `plotly>=5.0.0` - Enhanced visualizations

## 🎉 Conclusion

Your FactSnap-V system now supports both:
1. **📁 File Upload**: Original functionality for audio/video files
2. **🎙️ Real-time Streaming**: New live audio analysis capability

Users can seamlessly switch between both modes, making your application more versatile and user-friendly. The real-time streaming provides immediate feedback and analysis, perfect for live presentations, meetings, or interactive sessions.

The implementation maintains the same high-quality analysis while adding the convenience of real-time processing. All existing features (emotion detection, bias analysis, fact verification) work in both modes.