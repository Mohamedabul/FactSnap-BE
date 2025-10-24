"""
Test audio detection and processing improvements
"""

import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from real_time_audio import RealTimeAudioProcessor


def test_audio_detection():
    """
    Test the audio detection functionality
    """
    print("Testing audio detection improvements...")
    
    processor = RealTimeAudioProcessor()
    
    # Test 1: Silent audio (should not trigger processing)
    print("\n--- Test 1: Silent Audio ---")
    silent_audio = np.zeros(16000)  # 1 second of silence
    has_audio = processor._has_significant_audio(silent_audio)
    print(f"Silent audio detected as significant: {has_audio} (should be False)")
    
    # Test 2: Low-level noise (should not trigger processing)
    print("\n--- Test 2: Low-level Noise ---")
    noise_audio = np.random.normal(0, 100, 16000)  # Low-level noise
    has_audio = processor._has_significant_audio(noise_audio)
    print(f"Low noise detected as significant: {has_audio} (should be False)")
    
    # Test 3: Significant audio (should trigger processing)
    print("\n--- Test 3: Significant Audio ---")
    # Simulate speech-like audio with higher amplitude
    significant_audio = np.random.normal(0, 5000, 16000)
    has_audio = processor._has_significant_audio(significant_audio)
    print(f"Significant audio detected: {has_audio} (should be True)")
    
    # Test 4: Very loud audio (should definitely trigger)
    print("\n--- Test 4: Very Loud Audio ---")
    loud_audio = np.random.normal(0, 15000, 16000)
    has_audio = processor._has_significant_audio(loud_audio)
    print(f"Loud audio detected: {has_audio} (should be True)")
    
    # Test different thresholds
    print("\n--- Test 5: Different Thresholds ---")
    test_audio = np.random.normal(0, 1000, 16000)
    
    for threshold in [0.001, 0.01, 0.1]:
        has_audio = processor._has_significant_audio(test_audio, threshold)
        rms = np.sqrt(np.mean(test_audio.astype(float) ** 2)) / 32768.0
        print(f"Threshold {threshold}: RMS={rms:.4f}, Detected={has_audio}")
    
    processor.cleanup()
    print("\nAudio detection tests completed!")


def test_minimum_audio_length():
    """
    Test minimum audio length requirements
    """
    print("\n--- Testing Minimum Audio Length ---")
    
    min_length = 16000 * 2  # 2 seconds at 16kHz
    
    # Test different audio lengths
    lengths = [8000, 16000, 32000, 48000]  # 0.5s, 1s, 2s, 3s
    
    for length in lengths:
        meets_requirement = length >= min_length
        duration = length / 16000
        print(f"Audio length: {duration}s ({length} samples) - Meets requirement: {meets_requirement}")


if __name__ == "__main__":
    test_audio_detection()
    test_minimum_audio_length()