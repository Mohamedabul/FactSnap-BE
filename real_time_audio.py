"""
Real-time audio streaming module for FactSnap-V
Handles live audio capture and streaming processing
"""

import pyaudio
import wave
import threading
import queue
import time
import numpy as np
from collections import deque
import tempfile
import os
from datetime import datetime


class RealTimeAudioProcessor:
    """
    Handles real-time audio capture and processing
    """
    
    def __init__(self, 
                 sample_rate=16000,
                 chunk_size=1024,
                 channels=1,
                 format=pyaudio.paInt16,
                 buffer_duration=30.0):
        """
        Initialize real-time audio processor
        
        Args:
            sample_rate (int): Audio sample rate
            chunk_size (int): Audio chunk size for processing
            channels (int): Number of audio channels
            format: PyAudio format
            buffer_duration (float): Duration of audio buffer in seconds
        """
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.channels = channels
        self.format = format
        self.buffer_duration = buffer_duration
        
        # Audio processing
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.is_recording = False
        
        # Threading and queues
        self.audio_queue = queue.Queue()
        self.processing_thread = None
        self.recording_thread = None
        
        # Audio buffer for continuous processing
        self.audio_buffer = deque(maxlen=int(sample_rate * buffer_duration))
        
        # Temporary file management
        self.temp_dir = tempfile.mkdtemp()
        
        # Callbacks
        self.on_audio_chunk = None
        self.on_transcript_update = None
        self.on_analysis_complete = None
    
    def get_available_devices(self):
        """
        Get list of available audio input devices
        
        Returns:
            list: List of available audio devices
        """
        devices = []
        for i in range(self.audio.get_device_count()):
            device_info = self.audio.get_device_info_by_index(i)
            if device_info['maxInputChannels'] > 0:
                devices.append({
                    'index': i,
                    'name': device_info['name'],
                    'channels': device_info['maxInputChannels'],
                    'sample_rate': device_info['defaultSampleRate']
                })
        return devices
    
    def start_recording(self, device_index=None):
        """
        Start real-time audio recording
        
        Args:
            device_index (int): Audio device index (None for default)
        """
        if self.is_recording:
            print("Already recording!")
            return
        
        try:
            # Open audio stream
            self.stream = self.audio.open(
                format=self.format,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                input_device_index=device_index,
                frames_per_buffer=self.chunk_size,
                stream_callback=self._audio_callback
            )
            
            self.is_recording = True
            
            # Start processing thread
            self.processing_thread = threading.Thread(target=self._processing_loop)
            self.processing_thread.daemon = True
            self.processing_thread.start()
            
            print("Real-time audio recording started!")
            
        except Exception as e:
            raise Exception(f"Error starting audio recording: {str(e)}")
    
    def stop_recording(self):
        """
        Stop real-time audio recording
        """
        if not self.is_recording:
            return
        
        self.is_recording = False
        
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None
        
        # Wait for processing thread to finish
        if self.processing_thread:
            self.processing_thread.join(timeout=5.0)
        
        print("Real-time audio recording stopped!")
    
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """
        PyAudio callback for incoming audio data
        """
        if self.is_recording:
            # Convert bytes to numpy array
            audio_data = np.frombuffer(in_data, dtype=np.int16)
            
            # Add to buffer
            self.audio_buffer.extend(audio_data)
            
            # Add to processing queue
            self.audio_queue.put(audio_data)
            
            # Call callback if set
            if self.on_audio_chunk:
                self.on_audio_chunk(audio_data)
        
        return (in_data, pyaudio.paContinue)
    
    def _processing_loop(self):
        """
        Main processing loop for real-time audio
        """
        accumulated_audio = []
        last_process_time = time.time()
        process_interval = 5.0  # Process every 5 seconds
        min_audio_length = 16000 * 2  # Minimum 2 seconds of audio
        
        while self.is_recording:
            try:
                # Get audio chunk with timeout
                audio_chunk = self.audio_queue.get(timeout=1.0)
                accumulated_audio.extend(audio_chunk)
                
                # Process accumulated audio periodically
                current_time = time.time()
                if current_time - last_process_time >= process_interval:
                    if len(accumulated_audio) >= min_audio_length:
                        # Check if there's significant audio activity
                        audio_array = np.array(accumulated_audio)
                        if self._has_significant_audio(audio_array):
                            print(f"Processing {len(accumulated_audio)} audio samples...")
                            self._process_audio_chunk(audio_array)
                        else:
                            print("Skipping processing - no significant audio detected")
                        accumulated_audio = []
                        last_process_time = current_time
                    else:
                        print(f"Skipping processing - insufficient audio ({len(accumulated_audio)} samples)")
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in processing loop: {str(e)}")
        
        # Process any remaining audio
        if len(accumulated_audio) >= min_audio_length:
            audio_array = np.array(accumulated_audio)
            if self._has_significant_audio(audio_array):
                self._process_audio_chunk(audio_array)
    
    def _has_significant_audio(self, audio_data, threshold=0.01):
        """
        Check if audio data contains significant audio activity
        
        Args:
            audio_data (np.array): Audio data to check
            threshold (float): RMS threshold for significant audio
            
        Returns:
            bool: True if significant audio is detected
        """
        if len(audio_data) == 0:
            return False
        
        # Calculate RMS (Root Mean Square) energy
        rms = np.sqrt(np.mean(audio_data.astype(float) ** 2))
        
        # Normalize to 0-1 range (assuming 16-bit audio)
        normalized_rms = rms / 32768.0
        
        return normalized_rms > threshold
    
    def _process_audio_chunk(self, audio_data):
        """
        Process a chunk of audio data
        
        Args:
            audio_data (np.array): Audio data to process
        """
        try:
            # Save audio chunk to temporary file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            temp_audio_path = os.path.join(self.temp_dir, f"chunk_{timestamp}.wav")
            
            # Save as WAV file
            with wave.open(temp_audio_path, 'wb') as wav_file:
                wav_file.setnchannels(self.channels)
                wav_file.setsampwidth(self.audio.get_sample_size(self.format))
                wav_file.setframerate(self.sample_rate)
                wav_file.writeframes(audio_data.tobytes())
            
            # Process with speech-to-text (if callback is set)
            if self.on_transcript_update:
                self.on_transcript_update(temp_audio_path)
            
        except Exception as e:
            print(f"Error processing audio chunk: {str(e)}")
    
    def get_current_buffer_audio(self):
        """
        Get current audio buffer as numpy array
        
        Returns:
            np.array: Current audio buffer
        """
        return np.array(list(self.audio_buffer))
    
    def save_current_buffer(self, output_path=None):
        """
        Save current audio buffer to file
        
        Args:
            output_path (str): Output file path
            
        Returns:
            str: Path to saved audio file
        """
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(self.temp_dir, f"recorded_audio_{timestamp}.wav")
        
        audio_data = self.get_current_buffer_audio()
        
        if len(audio_data) == 0:
            raise ValueError("No audio data in buffer")
        
        # Save as WAV file
        with wave.open(output_path, 'wb') as wav_file:
            wav_file.setnchannels(self.channels)
            wav_file.setsampwidth(self.audio.get_sample_size(self.format))
            wav_file.setframerate(self.sample_rate)
            wav_file.writeframes(audio_data.tobytes())
        
        return output_path
    
    def cleanup(self):
        """
        Clean up resources
        """
        self.stop_recording()
        
        if self.audio:
            self.audio.terminate()
        
        # Clean up temporary files
        try:
            import shutil
            shutil.rmtree(self.temp_dir)
        except Exception as e:
            print(f"Warning: Could not clean up temporary files: {str(e)}")


class StreamingFactSnapV:
    """
    Streaming version of FactSnap-V for real-time processing
    """
    
    def __init__(self, factsnap_app):
        """
        Initialize streaming processor
        
        Args:
            factsnap_app: Instance of FactSnapV main application
        """
        self.factsnap_app = factsnap_app
        self.audio_processor = RealTimeAudioProcessor()
        
        # Set up callbacks
        self.audio_processor.on_transcript_update = self._process_audio_chunk
        
        # Thread-safe results storage
        self.results_queue = queue.Queue()
        self.streaming_results = {
            'transcripts': [],
            'emotions': [],
            'bias': [],
            'facts': [],
            'timestamps': []
        }
        
        # Callback for UI updates
        self.on_results_update = None
    
    def start_streaming(self, device_index=None):
        """
        Start streaming audio processing
        
        Args:
            device_index (int): Audio device index
        """
        print("Starting streaming FactSnap-V...")
        self.audio_processor.start_recording(device_index)
    
    def stop_streaming(self):
        """
        Stop streaming audio processing
        """
        print("Stopping streaming FactSnap-V...")
        self.audio_processor.stop_recording()
    
    def _process_audio_chunk(self, audio_path):
        """
        Process audio chunk with FactSnap-V pipeline
        
        Args:
            audio_path (str): Path to audio chunk file
        """
        try:
            print(f"Processing audio chunk: {audio_path}")
            
            # Get transcript
            transcript = self.factsnap_app.speech_to_text.get_clean_transcript(audio_path)
            
            if transcript.strip():
                # Preprocess text
                sentences = self.factsnap_app.text_preprocessor.preprocess_transcript(transcript)
                
                if sentences:
                    # Analyze emotions
                    emotion_results = self.factsnap_app.emotion_detector.analyze_sentences(sentences)
                    
                    # Analyze bias
                    bias_results = self.factsnap_app.bias_detector.analyze_sentences(sentences)
                    
                    # Fact verification (optional, might be too slow for real-time)
                    # fact_results = self.factsnap_app.fact_verifier.verify_sentences_batch(sentences)
                    
                    # Store results in thread-safe queue
                    timestamp = datetime.now().isoformat()
                    result = {
                        'transcript': transcript,
                        'emotions': emotion_results,
                        'bias': bias_results,
                        'timestamp': timestamp,
                        'sentence_count': len(sentences)
                    }
                    
                    # Add to internal storage
                    self.streaming_results['transcripts'].append(transcript)
                    self.streaming_results['emotions'].append(emotion_results)
                    self.streaming_results['bias'].append(bias_results)
                    self.streaming_results['timestamps'].append(timestamp)
                    
                    # Add to thread-safe queue for UI updates
                    self.results_queue.put(result)
                    
                    print(f"Processed chunk: {len(sentences)} sentences")
            
        except Exception as e:
            print(f"Error processing audio chunk: {str(e)}")
    
    def get_new_results(self):
        """
        Get new results from the queue (thread-safe)
        
        Returns:
            list: List of new results since last call
        """
        new_results = []
        try:
            while True:
                result = self.results_queue.get_nowait()
                new_results.append(result)
        except queue.Empty:
            pass
        return new_results
    
    def get_streaming_results(self):
        """
        Get current streaming results
        
        Returns:
            dict: Current streaming results
        """
        return self.streaming_results.copy()
    
    def save_session(self, output_path=None):
        """
        Save current streaming session
        
        Args:
            output_path (str): Output file path
            
        Returns:
            str: Path to saved audio file
        """
        return self.audio_processor.save_current_buffer(output_path)
    
    def get_available_devices(self):
        """
        Get available audio devices
        
        Returns:
            list: Available audio devices
        """
        return self.audio_processor.get_available_devices()
    
    def cleanup(self):
        """
        Clean up resources
        """
        self.audio_processor.cleanup()


def test_real_time_audio():
    """
    Test function for real-time audio processing
    """
    try:
        # Initialize processor
        processor = RealTimeAudioProcessor()
        
        # Get available devices
        devices = processor.get_available_devices()
        print("Available audio devices:")
        for device in devices:
            print(f"  {device['index']}: {device['name']}")
        
        if devices:
            print(f"\nTesting with device: {devices[0]['name']}")
            
            # Set up callback
            def on_chunk(audio_data):
                print(f"Received audio chunk: {len(audio_data)} samples")
            
            processor.on_audio_chunk = on_chunk
            
            # Record for 5 seconds
            processor.start_recording(devices[0]['index'])
            time.sleep(5)
            processor.stop_recording()
            
            # Save buffer
            output_path = processor.save_current_buffer()
            print(f"Saved audio to: {output_path}")
        
        processor.cleanup()
        
    except Exception as e:
        print(f"Error in test: {str(e)}")


if __name__ == "__main__":
    test_real_time_audio()