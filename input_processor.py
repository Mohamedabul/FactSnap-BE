"""
Input module for FactSnap-V
Handles audio and video file processing and audio extraction
"""

import os
import tempfile
from pathlib import Path
import uuid
import moviepy.editor as mp
from config import SUPPORTED_AUDIO_FORMATS, SUPPORTED_VIDEO_FORMATS


class InputProcessor:
    """
    Processes input files (audio/video) and extracts audio for speech-to-text
    """
    
    def __init__(self):
        self.temp_dir = tempfile.mkdtemp()
    
    def is_supported_file(self, file_path):
        """
        Check if the file format is supported
        
        Args:
            file_path (str): Path to the input file
            
        Returns:
            tuple: (is_supported, file_type) where file_type is 'audio' or 'video'
        """
        file_extension = Path(file_path).suffix.lower()
        
        if file_extension in SUPPORTED_AUDIO_FORMATS:
            return True, 'audio'
        elif file_extension in SUPPORTED_VIDEO_FORMATS:
            return True, 'video'
        else:
            return False, None
    
    def _ensure_temp_dir(self):
        """
        Ensure the temporary directory exists (recreate if cleaned up)
        """
        if not getattr(self, 'temp_dir', None) or not os.path.isdir(self.temp_dir):
            self.temp_dir = tempfile.mkdtemp()

    def extract_audio_from_video(self, video_path, output_path=None):
        """
        Extract audio from video file using MoviePy
        
        Args:
            video_path (str): Path to the video file
            output_path (str): Path for the extracted audio file
            
        Returns:
            str: Path to the extracted audio file
        """
        try:
            # Make sure we have a valid temp directory for outputs
            self._ensure_temp_dir()

            if output_path is None:
                unique = uuid.uuid4().hex
                output_path = os.path.join(self.temp_dir, f"extracted_audio_{unique}.wav")
            else:
                # Ensure parent directory exists
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Load video file
            video = mp.VideoFileClip(video_path)
            
            # Extract audio
            audio = video.audio
            
            # Save audio as WAV file
            audio.write_audiofile(output_path, verbose=False, logger=None)
            
            # Clean up
            audio.close()
            video.close()
            
            return output_path
            
        except Exception as e:
            raise Exception(f"Error extracting audio from video: {str(e)}")
    
    def process_input_file(self, file_path):
        """
        Process input file and return path to audio file
        
        Args:
            file_path (str): Path to the input file
            
        Returns:
            str: Path to the audio file ready for speech-to-text
        """
        # Convert to absolute path
        file_path = os.path.abspath(file_path)
        print(f"Processing input file: {file_path}")
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Input file not found: {file_path}")
        
        # Check file size
        file_size = os.path.getsize(file_path)
        print(f"Input file size: {file_size:,} bytes")
        
        if file_size == 0:
            raise ValueError("Input file is empty")
        
        is_supported, file_type = self.is_supported_file(file_path)
        
        if not is_supported:
            supported_formats = SUPPORTED_AUDIO_FORMATS + SUPPORTED_VIDEO_FORMATS
            raise ValueError(f"Unsupported file format. Supported formats: {supported_formats}")
        
        if file_type == 'audio':
            # Audio file - return as is
            print(f"Audio file detected, using directly: {file_path}")
            return file_path
        elif file_type == 'video':
            # Video file - extract audio
            print(f"Video file detected, extracting audio...")
            return self.extract_audio_from_video(file_path)
    
    def cleanup(self):
        """
        Clean up temporary files and recreate a fresh temp directory for future use
        """
        try:
            import shutil
            if getattr(self, 'temp_dir', None) and os.path.isdir(self.temp_dir):
                shutil.rmtree(self.temp_dir)
        except Exception as e:
            print(f"Warning: Could not clean up temporary files: {str(e)}")
        finally:
            # Always recreate a fresh temp directory so subsequent requests work
            self.temp_dir = tempfile.mkdtemp()


def test_input_processor():
    """
    Test function for the InputProcessor class
    """
    processor = InputProcessor()
    
    # Test supported file detection
    print("Testing file format detection...")
    print(f"test.mp3: {processor.is_supported_file('test.mp3')}")
    print(f"test.mp4: {processor.is_supported_file('test.mp4')}")
    print(f"test.txt: {processor.is_supported_file('test.txt')}")
    
    processor.cleanup()


if __name__ == "__main__":
    test_input_processor()
