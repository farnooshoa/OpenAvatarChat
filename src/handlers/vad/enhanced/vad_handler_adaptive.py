"""
Adaptive Voice Activity Detection (VAD) Handler
Addresses Issue #1: Incomplete speech capture
"""

import numpy as np
from collections import deque
from typing import Optional, List, Tuple
import time


class AdaptiveVADHandler:
    """
    Enhanced VAD with adaptive timeout and audio buffering
    to prevent premature speech cutoff
    """
    
    def __init__(self, config: dict):
        # Timeout configuration
        self.min_silence_duration = config.get('min_silence_duration_ms', 500)
        self.max_silence_duration = config.get('max_silence_duration_ms', 2000)
        self.sentence_end_silence = config.get('sentence_end_silence_ms', 800)
        self.phrase_pause_silence = config.get('phrase_pause_silence_ms', 300)
        
        # Speech detection thresholds
        self.speech_threshold = config.get('speech_threshold', 0.5)
        self.adaptive_timeout = config.get('adaptive_timeout', True)
        
        # Audio buffering
        self.sample_rate = config.get('sample_rate', 16000)
        self.chunk_duration_ms = config.get('chunk_duration_ms', 30)
        
        # Pre-speech buffer (capture audio before speech detection)
        pre_buffer_ms = config.get('pre_speech_buffer_ms', 500)
        pre_buffer_samples = int(pre_buffer_ms * self.sample_rate / 1000)
        self.pre_buffer = deque(maxlen=pre_buffer_samples)
        
        # Post-speech buffer (continue after silence detected)
        self.post_buffer_ms = config.get('post_speech_buffer_ms', 500)
        
        # State tracking
        self.is_speaking = False
        self.speech_start_time = None
        self.last_speech_time = None
        self.silence_start_time = None
        self.current_silence_threshold = self.min_silence_duration
        
        # Collected speech audio
        self.speech_audio = []
        self.post_silence_audio = []
        
        # Statistics for adaptive behavior
        self.recent_utterance_durations = deque(maxlen=5)
        self.recent_pause_durations = deque(maxlen=5)
        
    def update_silence_threshold(self, speech_duration_ms: float) -> float:
        """
        Adaptively adjust silence threshold based on speaking patterns
        Longer utterances get longer pause allowances
        """
        if not self.adaptive_timeout:
            return self.min_silence_duration
        
        # Long utterances (>2s) likely have natural sentence breaks
        if speech_duration_ms > 2000:
            threshold = self.sentence_end_silence
        # Medium utterances (0.5-2s) use moderate timeout
        elif speech_duration_ms > 500:
            threshold = (self.phrase_pause_silence + self.sentence_end_silence) / 2
        # Short utterances use minimum timeout
        else:
            threshold = self.phrase_pause_silence
        
        # Clamp to configured limits
        return max(self.min_silence_duration, 
                  min(threshold, self.max_silence_duration))
    
    def process_audio_chunk(self, 
                           audio_chunk: np.ndarray, 
                           speech_probability: float) -> Optional[np.ndarray]:
        """
        Process an audio chunk with speech probability from base VAD
        
        Args:
            audio_chunk: Audio samples for this chunk
            speech_probability: Probability of speech (0-1) from base VAD model
            
        Returns:
            Complete speech utterance if detected, None otherwise
        """
        current_time = time.time() * 1000  # Convert to ms
        is_speech = speech_probability >= self.speech_threshold
        
        # Always maintain pre-speech buffer
        if not self.is_speaking:
            self.pre_buffer.extend(audio_chunk)
        
        if is_speech:
            if not self.is_speaking:
                # Speech just started
                self._handle_speech_start(current_time)
                # Capture pre-buffer to get clean speech start
                self.speech_audio = list(self.pre_buffer)
            
            # Continue capturing speech
            self.speech_audio.extend(audio_chunk)
            self.last_speech_time = current_time
            self.silence_start_time = None
            self.post_silence_audio = []
            
        else:
            # Silence detected
            if self.is_speaking:
                if self.silence_start_time is None:
                    # Just transitioned to silence
                    self.silence_start_time = current_time
                
                # Collect post-speech audio (might be brief pause)
                self.post_silence_audio.extend(audio_chunk)
                
                # Calculate how long we've been silent
                silence_duration = current_time - self.silence_start_time
                
                # Check if silence exceeds threshold
                if silence_duration >= self.current_silence_threshold:
                    # Speech has ended
                    return self._handle_speech_end()
        
        return None
    
    def _handle_speech_start(self, current_time: float):
        """Handle the start of speech detection"""
        self.is_speaking = True
        self.speech_start_time = current_time
        self.last_speech_time = current_time
        self.silence_start_time = None
        self.speech_audio = []
        self.post_silence_audio = []
    
    def _handle_speech_end(self) -> np.ndarray:
        """Handle the end of speech detection and return captured audio"""
        # Calculate speech duration
        speech_duration = self.last_speech_time - self.speech_start_time
        
        # Update adaptive threshold for next utterance
        self.current_silence_threshold = self.update_silence_threshold(speech_duration)
        
        # Track statistics
        self.recent_utterance_durations.append(speech_duration)
        if self.silence_start_time:
            silence_duration = time.time() * 1000 - self.silence_start_time
            self.recent_pause_durations.append(silence_duration)
        
        # Combine speech + post-speech buffer
        final_audio = np.concatenate([
            np.array(self.speech_audio),
            np.array(self.post_silence_audio)
        ])
        
        # Reset state
        self.is_speaking = False
        self.speech_audio = []
        self.post_silence_audio = []
        
        return final_audio
    
    def force_end_speech(self) -> Optional[np.ndarray]:
        """Force end current speech capture (e.g., on timeout)"""
        if self.is_speaking and len(self.speech_audio) > 0:
            return self._handle_speech_end()
        return None
    
    def reset(self):
        """Reset VAD state"""
        self.is_speaking = False
        self.speech_start_time = None
        self.last_speech_time = None
        self.silence_start_time = None
        self.speech_audio = []
        self.post_silence_audio = []
        self.pre_buffer.clear()
    
    def get_statistics(self) -> dict:
        """Get current VAD statistics for monitoring"""
        return {
            'is_speaking': self.is_speaking,
            'current_silence_threshold_ms': self.current_silence_threshold,
            'avg_utterance_duration_ms': (
                np.mean(self.recent_utterance_durations) 
                if self.recent_utterance_durations else 0
            ),
            'avg_pause_duration_ms': (
                np.mean(self.recent_pause_durations)
                if self.recent_pause_durations else 0
            ),
            'buffer_size_samples': len(self.pre_buffer)
        }


# Example usage and testing
if __name__ == "__main__":
    # Configuration
    config = {
        'min_silence_duration_ms': 500,
        'max_silence_duration_ms': 2000,
        'sentence_end_silence_ms': 800,
        'phrase_pause_silence_ms': 300,
        'speech_threshold': 0.5,
        'adaptive_timeout': True,
        'sample_rate': 16000,
        'pre_speech_buffer_ms': 500,
        'post_speech_buffer_ms': 500,
    }
    
    vad = AdaptiveVADHandler(config)
    
    # Simulate audio stream with speech probabilities
    # This would come from SileroVAD in production
    print("Testing Adaptive VAD Handler...")
    
    # Simulate: silence -> speech -> pause -> speech -> silence
    test_pattern = [
        (0.1, 10),  # 10 chunks of silence
        (0.8, 30),  # 30 chunks of speech
        (0.2, 5),   # 5 chunks of quiet pause
        (0.9, 40),  # 40 chunks more speech
        (0.1, 20),  # 20 chunks of silence
    ]
    
    chunk_size = int(0.03 * 16000)  # 30ms chunks at 16kHz
    
    for speech_prob, num_chunks in test_pattern:
        for _ in range(num_chunks):
            # Generate dummy audio
            audio_chunk = np.random.randn(chunk_size) * 0.1
            
            # Process with VAD
            result = vad.process_audio_chunk(audio_chunk, speech_prob)
            
            if result is not None:
                duration_ms = len(result) / 16  # At 16kHz
                print(f"Speech captured! Duration: {duration_ms:.0f}ms")
                print(f"Statistics: {vad.get_statistics()}")
    
    print("Test complete!")