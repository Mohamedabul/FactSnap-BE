"""
Test script for real-time audio functionality
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from real_time_audio import RealTimeAudioProcessor, StreamingFactSnapV
from main import FactSnapV
import time


def test_audio_devices():
    """
    Test audio device detection
    """
    print("Testing audio device detection...")
    
    try:
        processor = RealTimeAudioProcessor()
        devices = processor.get_available_devices()
        
        print(f"Found {len(devices)} audio input devices:")
        for device in devices:
            print(f"  {device['index']}: {device['name']}")
            print(f"    Channels: {device['channels']}")
            print(f"    Sample Rate: {device['sample_rate']} Hz")
            print()
        
        processor.cleanup()
        return len(devices) > 0
        
    except Exception as e:
        print(f"Error testing audio devices: {str(e)}")
        return False


def test_audio_recording():
    """
    Test basic audio recording
    """
    print("Testing audio recording...")
    
    try:
        processor = RealTimeAudioProcessor()
        devices = processor.get_available_devices()
        
        if not devices:
            print("No audio devices available for testing")
            return False
        
        print(f"Using device: {devices[0]['name']}")
        
        # Set up callback
        chunk_count = 0
        def on_chunk(audio_data):
            nonlocal chunk_count
            chunk_count += 1
            print(f"Received chunk {chunk_count}: {len(audio_data)} samples")
        
        processor.on_audio_chunk = on_chunk
        
        # Record for 3 seconds
        print("Starting recording for 3 seconds...")
        processor.start_recording(devices[0]['index'])
        time.sleep(3)
        processor.stop_recording()
        
        # Save buffer
        output_path = processor.save_current_buffer("test_recording.wav")
        print(f"Saved recording to: {output_path}")
        
        processor.cleanup()
        
        # Check if file was created
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path)
            print(f"Recording file size: {file_size} bytes")
            return file_size > 0
        else:
            print("Recording file was not created")
            return False
        
    except Exception as e:
        print(f"Error testing audio recording: {str(e)}")
        return False


def test_streaming_integration():
    """
    Test streaming integration with FactSnap-V
    """
    print("Testing streaming integration...")
    
    try:
        # Initialize FactSnap-V
        print("Initializing FactSnap-V...")
        factsnap_app = FactSnapV()
        
        # Initialize streaming processor
        print("Initializing streaming processor...")
        streaming_processor = StreamingFactSnapV(factsnap_app)
        
        # Get available devices
        devices = streaming_processor.get_available_devices()
        
        if not devices:
            print("No audio devices available for testing")
            return False
        
        print(f"Using device: {devices[0]['name']}")
        
        # Set up callback
        results_received = []
        def on_results(results):
            results_received.append(results)
            print(f"Received results: {len(results['transcript'])} chars")
        
        streaming_processor.on_results_update = on_results
        
        # Stream for 5 seconds
        print("Starting streaming for 5 seconds...")
        streaming_processor.start_streaming(devices[0]['index'])
        time.sleep(5)
        streaming_processor.stop_streaming()
        
        # Check results
        print(f"Received {len(results_received)} result updates")
        
        # Save session
        output_path = streaming_processor.save_session("test_session.wav")
        print(f"Saved session to: {output_path}")
        
        streaming_processor.cleanup()
        
        return True
        
    except Exception as e:
        print(f"Error testing streaming integration: {str(e)}")
        return False


def main():
    """
    Run all tests
    """
    print("=" * 60)
    print("FactSnap-V Real-time Audio Testing")
    print("=" * 60)
    
    tests = [
        ("Audio Device Detection", test_audio_devices),
        ("Audio Recording", test_audio_recording),
        ("Streaming Integration", test_streaming_integration)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"Test failed with exception: {str(e)}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name}: {status}")
    
    all_passed = all(results.values())
    print(f"\nOverall: {'✅ ALL TESTS PASSED' if all_passed else '❌ SOME TESTS FAILED'}")
    
    if not all_passed:
        print("\nTroubleshooting:")
        print("- Ensure you have a working microphone connected")
        print("- Check that PyAudio is properly installed")
        print("- Try running: pip install pyaudio")
        print("- On Windows, you may need: pip install pipwin && pipwin install pyaudio")


if __name__ == "__main__":
    main()