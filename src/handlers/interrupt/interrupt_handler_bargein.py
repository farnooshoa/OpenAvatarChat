"""
Barge-In Interrupt Handler
Addresses Issue #3: Inability to interrupt the chatbot
"""

import asyncio
import numpy as np
from typing import Optional, Callable
from enum import Enum
import time
from queue import Queue, Empty


class ConversationState(Enum):
    """States of the conversation system"""
    IDLE = "idle"
    LISTENING = "listening"
    PROCESSING = "processing"
    RESPONDING = "responding"
    INTERRUPTED = "interrupted"


class BargeInHandler:
    """
    Handles user interruption (barge-in) during chatbot response
    Monitors audio input while TTS is playing
    """
    
    def __init__(self, vad_handler, config: dict):
        self.vad = vad_handler
        
        # Configuration
        self.enable_barge_in = config.get('enable_barge_in', True)
        self.interrupt_threshold = config.get('interrupt_threshold', 0.7)
        self.interrupt_min_duration_ms = config.get('interrupt_min_duration_ms', 200)
        self.interrupt_cooldown_ms = config.get('interrupt_cooldown_ms', 500)
        
        # State
        self.is_monitoring = False
        self.interrupt_detected = False
        self.last_interrupt_time = 0
        
        # Callbacks
        self.on_interrupt_callback: Optional[Callable] = None
        
    async def start_monitoring(self, audio_stream) -> bool:
        """
        Monitor audio stream for user speech during bot response
        
        Args:
            audio_stream: Async audio input stream
            
        Returns:
            True if interrupt detected, False if monitoring stopped normally
        """
        if not self.enable_barge_in:
            return False
        
        self.is_monitoring = True
        self.interrupt_detected = False
        
        consecutive_speech_ms = 0
        
        print("[Barge-In] Monitoring started")
        
        while self.is_monitoring:
            try:
                # Read audio chunk (non-blocking with timeout)
                audio_chunk = await asyncio.wait_for(
                    audio_stream.read_async(),
                    timeout=0.1
                )
                
                # Check for speech using VAD
                speech_prob = self.vad.get_speech_probability(audio_chunk)
                is_speech = speech_prob >= self.interrupt_threshold
                
                if is_speech:
                    consecutive_speech_ms += 30  # Assuming 30ms chunks
                    
                    # Check if speech duration exceeds threshold
                    if consecutive_speech_ms >= self.interrupt_min_duration_ms:
                        # Check cooldown period
                        current_time = time.time() * 1000
                        if (current_time - self.last_interrupt_time) > self.interrupt_cooldown_ms:
                            print(f"[Barge-In] Interrupt detected! "
                                  f"Speech duration: {consecutive_speech_ms}ms")
                            
                            self.interrupt_detected = True
                            self.last_interrupt_time = current_time
                            
                            # Trigger callback if set
                            if self.on_interrupt_callback:
                                await self.on_interrupt_callback()
                            
                            return True
                else:
                    consecutive_speech_ms = 0
                
            except asyncio.TimeoutError:
                # No audio available, continue monitoring
                continue
            except Exception as e:
                print(f"[Barge-In] Error during monitoring: {e}")
                break
        
        print("[Barge-In] Monitoring stopped")
        return False
    
    def stop_monitoring(self):
        """Stop interrupt monitoring"""
        self.is_monitoring = False
    
    def set_interrupt_callback(self, callback: Callable):
        """Set callback to be called when interrupt is detected"""
        self.on_interrupt_callback = callback


class CancellableTTS:
    """
    TTS handler with interruption support
    Generates and plays audio in chunks for quick cancellation
    """
    
    def __init__(self, base_tts, config: dict):
        self.tts = base_tts
        
        # Configuration
        self.enable_interruption = config.get('enable_interruption', True)
        self.streaming_mode = config.get('streaming_mode', True)
        self.chunk_size_ms = config.get('chunk_size_ms', 1000)
        
        # State
        self.audio_queue = Queue()
        self.is_playing = False
        self.interrupt_requested = False
        
        # For tracking playback progress
        self.total_generated_chunks = 0
        self.played_chunks = 0
    
    async def generate_and_queue_audio(self, text: str):
        """
        Generate TTS audio in chunks and queue for playback
        Allows early interruption during generation
        
        Args:
            text: Text to synthesize
        """
        # Split text into sentences for streaming
        sentences = self._split_into_sentences(text)
        
        self.total_generated_chunks = 0
        
        for i, sentence in enumerate(sentences):
            if self.interrupt_requested:
                print(f"[TTS] Generation interrupted at sentence {i}/{len(sentences)}")
                break
            
            # Generate audio for sentence
            try:
                audio_chunk = await self.tts.generate_async(sentence)
                
                if not self.interrupt_requested:
                    self.audio_queue.put(audio_chunk)
                    self.total_generated_chunks += 1
            except Exception as e:
                print(f"[TTS] Error generating audio: {e}")
                break
    
    async def play_audio_stream(self, audio_output):
        """
        Play queued audio chunks with interruption support
        
        Args:
            audio_output: Audio output stream
        """
        self.is_playing = True
        self.interrupt_requested = False
        self.played_chunks = 0
        
        print("[TTS] Playback started")
        
        while self.is_playing:
            try:
                # Get next audio chunk (non-blocking with timeout)
                audio_chunk = self.audio_queue.get(timeout=0.1)
                
                # Check for interrupt before playing
                if self.interrupt_requested:
                    print(f"[TTS] Playback interrupted at chunk {self.played_chunks}")
                    break
                
                # Play chunk
                await audio_output.write_async(audio_chunk)
                self.played_chunks += 1
                
            except Empty:
                # No more audio in queue
                if self.audio_queue.empty():
                    print("[TTS] All audio played")
                    break
            except Exception as e:
                print(f"[TTS] Error during playback: {e}")
                break
        
        self.is_playing = False
        self._clear_queue()
    
    def interrupt(self):
        """
        Stop current generation and playback immediately
        """
        print("[TTS] Interrupt requested")
        self.interrupt_requested = True
        self.is_playing = False
        self._clear_queue()
    
    def _clear_queue(self):
        """Clear remaining audio from queue"""
        cleared = 0
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
                cleared += 1
            except Empty:
                break
        
        if cleared > 0:
            print(f"[TTS] Cleared {cleared} queued audio chunks")
    
    def _split_into_sentences(self, text: str) -> list:
        """
        Split text into sentences for streaming TTS
        
        Args:
            text: Input text
            
        Returns:
            List of sentences
        """
        import re
        
        # Split on sentence boundaries
        sentences = re.split(r'([.!?]+\s+)', text)
        
        # Recombine sentence with its punctuation
        result = []
        for i in range(0, len(sentences) - 1, 2):
            sentence = sentences[i] + (sentences[i + 1] if i + 1 < len(sentences) else '')
            sentence = sentence.strip()
            if sentence:
                result.append(sentence)
        
        # Add last sentence if exists
        if len(sentences) > 0 and sentences[-1].strip():
            result.append(sentences[-1].strip())
        
        return result if result else [text]
    
    def get_playback_progress(self) -> float:
        """
        Get current playback progress (0.0 to 1.0)
        """
        if self.total_generated_chunks == 0:
            return 0.0
        
        return self.played_chunks / self.total_generated_chunks


class InterruptController:
    """
    Central controller for managing conversation with interrupt support
    Coordinates VAD, ASR, LLM, TTS, and barge-in detection
    """
    
    def __init__(self, vad, asr, llm, tts, config: dict):
        self.vad = vad
        self.asr = asr
        self.llm = llm
        
        # Create cancellable TTS wrapper
        self.tts = CancellableTTS(tts, config)
        
        # Create barge-in handler
        self.barge_in = BargeInHandler(vad, config)
        
        # Set interrupt callback
        self.barge_in.set_interrupt_callback(self._on_interrupt_detected)
        
        # State
        self.state = ConversationState.IDLE
        self.current_response_task = None
        
        # Configuration
        self.max_response_time_ms = config.get('max_response_time_ms', 30000)
        
    async def _on_interrupt_detected(self):
        """Callback when user interrupt is detected"""
        print("[Controller] Interrupt detected - stopping TTS")
        self.tts.interrupt()
        self.state = ConversationState.INTERRUPTED
    
    async def handle_conversation_turn(self, audio_input, audio_output):
        """
        Handle one conversation turn with interrupt support
        
        Args:
            audio_input: Audio input stream
            audio_output: Audio output stream
            
        Returns:
            True if conversation should continue, False to end
        """
        # State: Listening for user input
        print("\n[Controller] Listening for user input...")
        self.state = ConversationState.LISTENING
        
        user_audio = await self._listen_for_speech(audio_input)
        
        if user_audio is None:
            return True  # Continue listening
        
        # State: Processing user input
        print("[Controller] Processing user input...")
        self.state = ConversationState.PROCESSING
        
        user_text = await self.asr.transcribe_async(user_audio)
        print(f"[Controller] User said: {user_text}")
        
        llm_response = await self.llm.generate_async(user_text)
        print(f"[Controller] LLM response: {llm_response[:100]}...")
        
        # State: Responding with TTS
        print("[Controller] Generating and playing response...")
        self.state = ConversationState.RESPONDING
        
        # Create concurrent tasks: TTS generation/playback + interrupt monitoring
        generation_task = asyncio.create_task(
            self.tts.generate_and_queue_audio(llm_response)
        )
        
        playback_task = asyncio.create_task(
            self.tts.play_audio_stream(audio_output)
        )
        
        monitor_task = asyncio.create_task(
            self.barge_in.start_monitoring(audio_input)
        )
        
        # Wait for either completion or interrupt
        done, pending = await asyncio.wait(
            [playback_task, monitor_task],
            return_when=asyncio.FIRST_COMPLETED
        )
        
        # Check if user interrupted
        interrupted = False
        if monitor_task in done:
            interrupted = await monitor_task
        
        if interrupted:
            print("[Controller] User interrupted the response")
            # Cancel remaining tasks
            generation_task.cancel()
            playback_task.cancel()
            
            # Stop monitoring
            self.barge_in.stop_monitoring()
        else:
            # Playback completed normally
            monitor_task.cancel()
            await generation_task  # Wait for generation to finish
        
        # Clean up
        for task in pending:
            task.cancel()
        
        self.state = ConversationState.IDLE
        return True
    
    async def _listen_for_speech(self, audio_input):
        """
        Listen for complete user utterance using VAD
        
        Args:
            audio_input: Audio input stream
            
        Returns:
            Complete speech audio or None
        """
        # This would integrate with the AdaptiveVADHandler
        # For now, simplified placeholder
        speech_audio = await self.vad.wait_for_speech_async(audio_input)
        return speech_audio


# Example usage and testing
if __name__ == "__main__":
    print("Testing Interrupt Handler...")
    
    # Mock components for testing
    class MockVAD:
        def get_speech_probability(self, audio):
            return 0.8 if np.mean(np.abs(audio)) > 0.1 else 0.2
    
    class MockTTS:
        async def generate_async(self, text):
            await asyncio.sleep(0.1)  # Simulate generation time
            return np.random.randn(16000)  # 1 second of audio
    
    class MockAudioStream:
        def __init__(self):
            self.chunks = []
            
        async def read_async(self):
            await asyncio.sleep(0.03)  # 30ms chunks
            # Simulate speech after 10 chunks
            if len(self.chunks) > 10:
                return np.random.randn(480) * 0.5  # Loud audio (speech)
            else:
                return np.random.randn(480) * 0.01  # Quiet audio (silence)
    
    # Configuration
    config = {
        'enable_barge_in': True,
        'interrupt_threshold': 0.7,
        'interrupt_min_duration_ms': 200,
        'interrupt_cooldown_ms': 500,
        'enable_interruption': True,
        'streaming_mode': True,
    }
    
    async def test_barge_in():
        vad = MockVAD()
        tts = CancellableTTS(MockTTS(), config)
        barge_in = BargeInHandler(vad, config)
        
        # Set interrupt callback
        interrupt_detected = asyncio.Event()
        barge_in.set_interrupt_callback(lambda: interrupt_detected.set())
        
        # Start monitoring
        audio_stream = MockAudioStream()
        monitor_task = asyncio.create_task(barge_in.start_monitoring(audio_stream))
        
        # Wait for interrupt or timeout
        try:
            result = await asyncio.wait_for(monitor_task, timeout=2.0)
            print(f"Interrupt detected: {result}")
        except asyncio.TimeoutError:
            print("No interrupt detected within timeout")
            barge_in.stop_monitoring()
    
    # Run test
    asyncio.run(test_barge_in())
    print("Test complete!")