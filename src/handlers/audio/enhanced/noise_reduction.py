"""
Noise-Adaptive Audio Preprocessing
Addresses Issue #2: Environmental noise interference
"""

import numpy as np
from scipy import signal
from scipy.signal import butter, filtfilt
from typing import Optional, Tuple
from collections import deque


class AudioPreprocessor:
    """
    Audio preprocessing pipeline with noise reduction and filtering
    """
    
    def __init__(self, config: dict):
        self.sample_rate = config.get('sample_rate', 16000)
        
        # Feature flags
        self.enable_noise_reduction = config.get('enable_noise_reduction', True)
        self.enable_bandpass = config.get('enable_bandpass', True)
        self.enable_normalization = config.get('enable_normalization', True)
        
        # Bandpass filter parameters (speech frequencies: 300-3400 Hz)
        self.low_freq = config.get('low_freq', 300)
        self.high_freq = config.get('high_freq', 3400)
        self.filter_order = config.get('filter_order', 4)
        
        # Noise reduction parameters
        self.noise_reduce_strength = config.get('noise_reduce_strength', 0.5)
        self.noise_update_rate = config.get('noise_update_rate', 0.95)
        
        # Noise profile
        self.noise_profile = None
        self.noise_spectrum = None
        
        # Normalization
        self.target_level = config.get('target_level', 0.95)
        
    def apply_bandpass_filter(self, audio: np.ndarray) -> np.ndarray:
        """
        Apply bandpass filter to focus on speech frequencies
        Removes low-frequency rumble and high-frequency hiss
        """
        nyquist = self.sample_rate / 2
        low = self.low_freq / nyquist
        high = self.high_freq / nyquist
        
        # Design Butterworth bandpass filter
        b, a = butter(self.filter_order, [low, high], btype='band')
        
        # Apply filter (forward and backward to avoid phase shift)
        filtered = filtfilt(b, a, audio)
        
        return filtered
    
    def spectral_subtraction(self, 
                            audio: np.ndarray,
                            noise_spectrum: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Apply spectral subtraction for noise reduction
        
        Args:
            audio: Input audio signal
            noise_spectrum: Noise power spectrum (if None, estimate from signal)
            
        Returns:
            Noise-reduced audio
        """
        # Compute STFT
        nperseg = int(self.sample_rate * 0.025)  # 25ms windows
        noverlap = int(nperseg * 0.75)  # 75% overlap
        
        f, t, Zxx = signal.stft(audio, 
                               fs=self.sample_rate,
                               nperseg=nperseg,
                               noverlap=noverlap)
        
        # Magnitude and phase
        magnitude = np.abs(Zxx)
        phase = np.angle(Zxx)
        
        # Estimate or use provided noise spectrum
        if noise_spectrum is None:
            # Estimate from first few frames (assumed to be silence)
            noise_frames = min(10, magnitude.shape[1])
            noise_mag = np.mean(magnitude[:, :noise_frames], axis=1, keepdims=True)
        else:
            noise_mag = noise_spectrum.reshape(-1, 1)
        
        # Spectral subtraction with over-subtraction factor
        alpha = self.noise_reduce_strength * 2  # Over-subtraction factor
        beta = 0.01  # Spectral floor
        
        # Subtract noise
        clean_magnitude = magnitude - alpha * noise_mag
        
        # Apply spectral floor to prevent musical noise
        clean_magnitude = np.maximum(clean_magnitude, beta * magnitude)
        
        # Reconstruct signal
        clean_Zxx = clean_magnitude * np.exp(1j * phase)
        
        # Inverse STFT
        _, reconstructed = signal.istft(clean_Zxx,
                                       fs=self.sample_rate,
                                       nperseg=nperseg,
                                       noverlap=noverlap)
        
        # Trim to original length
        reconstructed = reconstructed[:len(audio)]
        
        return reconstructed
    
    def wiener_filter(self, 
                     audio: np.ndarray,
                     noise_power: float) -> np.ndarray:
        """
        Apply Wiener filter for noise reduction
        More sophisticated than spectral subtraction
        """
        # Compute power spectral density
        f, psd = signal.welch(audio, self.sample_rate, nperseg=512)
        
        # Estimate signal power
        signal_power = psd - noise_power
        signal_power = np.maximum(signal_power, 0)  # Ensure non-negative
        
        # Wiener gain
        gain = signal_power / (signal_power + noise_power)
        
        # Apply in frequency domain
        fft_audio = np.fft.rfft(audio)
        fft_filtered = fft_audio * gain[:len(fft_audio)]
        filtered = np.fft.irfft(fft_filtered, n=len(audio))
        
        return filtered
    
    def normalize_audio(self, audio: np.ndarray) -> np.ndarray:
        """
        Normalize audio to target level
        Helps maintain consistent volume across utterances
        """
        max_val = np.abs(audio).max()
        
        if max_val > 0:
            normalized = audio / max_val * self.target_level
            return normalized
        
        return audio
    
    def process(self, 
                audio: np.ndarray,
                noise_profile: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Apply full preprocessing pipeline
        
        Args:
            audio: Input audio signal
            noise_profile: Optional noise profile for reduction
            
        Returns:
            Preprocessed audio
        """
        processed = audio.copy()
        
        # Step 1: Bandpass filter (remove frequencies outside speech range)
        if self.enable_bandpass:
            processed = self.apply_bandpass_filter(processed)
        
        # Step 2: Noise reduction
        if self.enable_noise_reduction:
            if noise_profile is not None:
                # Use provided noise profile
                processed = self.spectral_subtraction(processed, noise_profile)
            else:
                # Estimate noise from signal
                processed = self.spectral_subtraction(processed)
        
        # Step 3: Normalize
        if self.enable_normalization:
            processed = self.normalize_audio(processed)
        
        return processed


class NoiseProfiler:
    """
    Manages noise profiling and adaptation
    """
    
    def __init__(self, config: dict):
        self.sample_rate = config.get('sample_rate', 16000)
        self.profile_duration_ms = config.get('profile_duration_ms', 3000)
        self.update_rate = config.get('noise_update_rate', 0.95)
        
        self.noise_profile = None
        self.is_calibrated = False
        
    def calibrate(self, audio: np.ndarray) -> np.ndarray:
        """
        Initial noise calibration from silence recording
        
        Args:
            audio: Audio recording during silence (noise only)
            
        Returns:
            Noise power spectrum
        """
        # Compute power spectrum
        f, psd = signal.welch(audio, self.sample_rate, nperseg=512)
        
        self.noise_profile = psd
        self.is_calibrated = True
        
        return self.noise_profile
    
    def update_noise_profile(self, audio: np.ndarray, is_silence: bool):
        """
        Continuously update noise profile during silence periods
        
        Args:
            audio: Current audio chunk
            is_silence: Whether this chunk is silence (no speech)
        """
        if not is_silence or not self.is_calibrated:
            return
        
        # Compute current power spectrum
        f, psd = signal.welch(audio, self.sample_rate, nperseg=512)
        
        # Exponential moving average
        self.noise_profile = (
            self.update_rate * self.noise_profile +
            (1 - self.update_rate) * psd
        )
    
    def get_noise_profile(self) -> Optional[np.ndarray]:
        """Get current noise profile"""
        return self.noise_profile if self.is_calibrated else None
    
    def estimate_snr(self, audio: np.ndarray) -> float:
        """
        Estimate Signal-to-Noise Ratio in dB
        
        Args:
            audio: Audio signal (assumed to contain speech)
            
        Returns:
            SNR in decibels
        """
        if not self.is_calibrated:
            return float('inf')
        
        # Signal power
        signal_power = np.mean(audio ** 2)
        
        # Noise power (from profile)
        noise_power = np.mean(self.noise_profile)
        
        # SNR in dB
        if noise_power > 0:
            snr_db = 10 * np.log10(signal_power / noise_power)
            return snr_db
        
        return float('inf')


# Example usage and testing
if __name__ == "__main__":
    print("Testing Audio Preprocessor...")
    
    # Configuration
    config = {
        'sample_rate': 16000,
        'enable_noise_reduction': True,
        'enable_bandpass': True,
        'enable_normalization': True,
        'low_freq': 300,
        'high_freq': 3400,
        'noise_reduce_strength': 0.5,
    }
    
    preprocessor = AudioPreprocessor(config)
    profiler = NoiseProfiler(config)
    
    # Generate test signal: clean speech
    duration = 2  # seconds
    t = np.linspace(0, duration, int(16000 * duration))
    clean_speech = np.sin(2 * np.pi * 500 * t)  # 500 Hz tone (speech-like)
    
    # Add noise
    noise = np.random.randn(len(clean_speech)) * 0.3
    noisy_speech = clean_speech + noise
    
    # Calibrate noise profile from noise sample
    print("Calibrating noise profile...")
    profiler.calibrate(noise[:int(16000 * 0.5)])  # 0.5s of noise
    
    # Estimate SNR
    snr_before = profiler.estimate_snr(noisy_speech)
    print(f"SNR before processing: {snr_before:.1f} dB")
    
    # Process audio
    print("Applying preprocessing...")
    processed = preprocessor.process(noisy_speech, profiler.get_noise_profile())
    
    # Estimate SNR after processing
    snr_after = profiler.estimate_snr(processed)
    print(f"SNR after processing: {snr_after:.1f} dB")
    print(f"SNR improvement: {snr_after - snr_before:.1f} dB")
    
    print("Test complete!")
