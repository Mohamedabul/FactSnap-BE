"""
Speech-to-text module for FactSnap-V
Uses OpenAI Whisper for local speech-to-text conversion
"""

import whisper
import os
import warnings
from config import WHISPER_MODEL_SIZE

# Suppress warnings
warnings.filterwarnings("ignore")


class SpeechToText:
    """
    Handles speech-to-text conversion using OpenAI Whisper
    """
    
    def __init__(self, model_size=WHISPER_MODEL_SIZE):
        """
        Initialize the Whisper model
        
        Args:
            model_size (str): Size of the Whisper model to use
        """
        self.model_size = model_size
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """
        Load the Whisper model for offline use
        """
        try:
            print(f"Loading Whisper model ({self.model_size})...")
            # Force CPU to avoid device/meta tensor issues on some Windows setups
            self.model = whisper.load_model(self.model_size, device='cpu')
            print("Whisper model loaded successfully!")
        except Exception as e:
            raise Exception(f"Error loading Whisper model: {str(e)}")
    
    def transcribe_audio(self, audio_path, language=None):
        """
        Transcribe audio file to text
        
        Args:
            audio_path (str): Path to the audio file
            language (str): Language code (optional, auto-detect if None)
            
        Returns:
            dict: Transcription result containing text and segments
        """
        # Convert to absolute path and normalize
        audio_path = os.path.abspath(audio_path)
        print(f"Attempting to transcribe: {audio_path}")
        
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        # Check file size
        file_size = os.path.getsize(audio_path)
        print(f"File size: {file_size:,} bytes")
        
        if file_size == 0:
            raise Exception("Audio file is empty")
        
        if self.model is None:
            raise Exception("Whisper model not loaded")
        
        try:
            print("Transcribing audio...")
            
            # Try to load and transcribe the audio file
            # Use raw string to avoid path issues
            result = self.model.transcribe(
                audio_path,
                language=language,
                task="transcribe",
                verbose=False,
                fp16=False  # Disable fp16 to avoid potential issues
            )
            
            print("Transcription completed!")
            
            # Return clean result
            return {
                'text': result['text'].strip(),
                'segments': result['segments'],
                'language': result['language']
            }
            
        except Exception as e:
            print(f"Transcription error details: {str(e)}")
            print(f"Audio file exists: {os.path.exists(audio_path)}")
            print(f"Audio file readable: {os.access(audio_path, os.R_OK)}")
            raise Exception(f"Error transcribing audio: {str(e)}")
    
    def get_transcript_with_timestamps(self, audio_path, language=None):
        """
        Get transcript with timestamp information
        
        Args:
            audio_path (str): Path to the audio file
            language (str): Language code (optional)
            
        Returns:
            list: List of segments with text and timestamps
        """
        result = self.transcribe_audio(audio_path, language)
        
        segments = []
        for segment in result['segments']:
            segments.append({
                'start': segment['start'],
                'end': segment['end'],
                'text': segment['text'].strip()
            })
        
        return segments
    
    def get_clean_transcript(self, audio_path, language=None):
        """
        Get clean transcript text without timestamps
        
        Args:
            audio_path (str): Path to the audio file
            language (str): Language code (optional)
            
        Returns:
            str: Clean transcript text
        """
        result = self.transcribe_audio(audio_path, language)
        return result['text']


def test_speech_to_text():
    """
    Test function for the SpeechToText class
    """
    try:
        # Initialize speech-to-text
        stt = SpeechToText()
        
        # Test with a sample audio file (if available)
        sample_audio = "sample.wav"
        
        if os.path.exists(sample_audio):
            print(f"Testing with audio file: {sample_audio}")
            
            # Get transcript
            transcript = stt.get_clean_transcript(sample_audio)
            print(f"Transcript: {transcript}")
            
            # Get segments with timestamps
            segments = stt.get_transcript_with_timestamps(sample_audio)
            print(f"Number of segments: {len(segments)}")
            
        else:
            print("No sample audio file found for testing")
            print("SpeechToText class initialized successfully!")
            
    except Exception as e:
        print(f"Error in test: {str(e)}")


if __name__ == "__main__":
    test_speech_to_text()
