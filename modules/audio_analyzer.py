import pyaudio
import wave
import numpy as np
import threading
from collections import deque
import time
from typing import Dict, List, Optional, Any
from scipy.signal import welch
from .utils import FeatureModule, AnalyzerConfig

class AudioProcessor:
    """Handles low-level audio processing operations"""
    def __init__(self):
        self.audio = pyaudio.PyAudio()
        self.chunk_size = 1024
        self.sample_rate = 44100
        self.format = pyaudio.paFloat32
        self.channels = 1
        
        # Analysis parameters
        self.min_frequency = 50
        self.max_frequency = 4000
        self.volume_threshold = 0.01
        self.silence_threshold = 0.1
        
    def __del__(self):
        """Cleanup audio resources"""
        self.audio.terminate()

class AudioAnalyzer(FeatureModule):
    """Main audio analysis class"""
    def __init__(self, config: AnalyzerConfig):
        super().__init__(config)
        self.processor = AudioProcessor()
        
        # Analysis buffers
        self.audio_buffer = deque(maxlen=50)
        self.energy_buffer = deque(maxlen=50)
        self.pitch_buffer = deque(maxlen=50)
        
        # State tracking
        self.is_recording = False
        self.speech_detected = False
        self.pause_duration = 0
        self.speaking_duration = 0
        self.last_process_time = None
        
        # Speech metrics
        self.speech_stats = {
            'volume_level': 0,
            'speech_rate': 0,
            'pitch_variation': 0,
            'speaking_duration': 0,
            'pause_count': 0,
            'energy_level': 0
        }
        
        # Initialize recording thread
        self.recording_thread = None
        self.stream = None
        
    def start_recording(self):
        """Start audio recording"""
        if self.is_recording:
            return
            
        self.is_recording = True
        self.last_process_time = time.time()
        
        try:
            # Open audio stream
            self.stream = self.processor.audio.open(
                format=self.processor.format,
                channels=self.processor.channels,
                rate=self.processor.sample_rate,
                input=True,
                frames_per_buffer=self.processor.chunk_size,
                stream_callback=self._audio_callback
            )
            
            print("Audio recording started")
            
        except Exception as e:
            print(f"Error starting audio recording: {str(e)}")
            self.is_recording = False
            if self.stream:
                self.stream.close()
            self.stream = None
    
    def stop_recording(self, output_path: Optional[str] = None):
        """Stop audio recording"""
        if not self.is_recording:
            return
            
        self.is_recording = False
        
        if self.stream:
            try:
                self.stream.stop_stream()
                self.stream.close()
                print("Audio recording stopped")
            except Exception as e:
                print(f"Error stopping audio stream: {str(e)}")
        
        if output_path and self.audio_buffer:
            self._save_audio(output_path)
    
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """Process incoming audio data"""
        if self.is_recording:
            try:
                # Store raw audio data
                self.audio_buffer.append(in_data)
                
                # Process audio for analysis
                audio_data = np.frombuffer(in_data, dtype=np.float32)
                self._process_audio(audio_data)
                
                return (in_data, pyaudio.paContinue)
            except Exception as e:
                print(f"Error in audio callback: {str(e)}")
                return (in_data, pyaudio.paContinue)
        return (None, pyaudio.paComplete)
    
    def _process_audio(self, audio_data: np.ndarray):
        """Process audio data with improved speech rate calculation"""
        try:
            # Calculate volume level
            volume = float(np.mean(np.abs(audio_data))) if len(audio_data) > 0 else 0.0
            self.energy_buffer.append(volume)
            self.speech_stats['volume_level'] = volume
            
            current_time = time.time()
            if self.last_process_time is None:
                self.last_process_time = current_time
                return
            
            time_diff = current_time - self.last_process_time
            
            # Improved speech detection threshold
            is_speaking = volume > self.processor.volume_threshold * 1.2  # Slightly higher threshold
            
            if is_speaking:
                if not self.speech_detected:
                    # New speech segment started
                    self.speech_stats['pause_count'] += 1
                self.speech_detected = True
                self.speaking_duration += time_diff
                
                # Update speech rate based on energy variations
                energy_variations = np.diff([e for e in self.energy_buffer])
                zero_crossings = np.sum(np.diff(np.signbit(energy_variations)))
                
                # Estimate syllables from energy variations
                estimated_syllables = max(1, zero_crossings // 2)
                # Estimate words (assuming average 1.5 syllables per word)
                estimated_words = estimated_syllables / 1.5
                
                # Calculate words per minute
                if self.speaking_duration > 0:
                    wpm = (estimated_words * 60) / self.speaking_duration
                    # Smooth the speech rate
                    if 'speech_rate' not in self.speech_stats:
                        self.speech_stats['speech_rate'] = wpm
                    else:
                        alpha = 0.3  # Smoothing factor
                        self.speech_stats['speech_rate'] = (alpha * wpm + 
                            (1 - alpha) * self.speech_stats['speech_rate'])
            else:
                if self.speech_detected:
                    print(f"Speech ended. Duration: {self.speaking_duration:.1f}s")
                self.speech_detected = False
                self.pause_duration += time_diff
            
            # Round speech rate to whole number
            if 'speech_rate' in self.speech_stats:
                self.speech_stats['speech_rate'] = int(round(self.speech_stats['speech_rate']))
            
            self.last_process_time = current_time
            
        except Exception as e:
            print(f"Error processing audio: {str(e)}")
    
    def _calculate_energy(self, audio_data: np.ndarray) -> float:
        """Calculate audio energy level with safe division"""
        try:
            if len(audio_data) == 0:
                return 0.0
            return float(np.sum(audio_data ** 2)) / len(audio_data)
        except Exception as e:
            print(f"Error calculating energy: {str(e)}")
            return 0.0

    def get_speech_feedback(self) -> str:
        """Generate feedback based on speech analysis with safe value checking"""
        feedback = []
        
        try:
            # Volume feedback
            vol = self.speech_stats.get('volume_level', 0)
            if vol < 0.01:
                feedback.append("Speak louder")
            elif vol > 0.5:
                feedback.append("Lower your voice")
            elif vol > 0.02:
                feedback.append("Good volume")
            
            # Speech rate feedback only if speaking
            wpm = self.speech_stats.get('speech_rate', 0)
            if self.speaking_duration > 1 and self.speech_detected:
                if wpm > 160:
                    feedback.append("Speaking too fast")
                elif wpm < 80:
                    feedback.append("Speaking too slow")
                else:
                    feedback.append("Good pace")
            
            # Pitch variation feedback
            pitch_var = self.speech_stats.get('pitch_variation', 0)
            if self.speech_detected and len(self.pitch_buffer) > 0:
                if pitch_var < 10:
                    feedback.append("Add more vocal variety")
                elif pitch_var > 50:
                    feedback.append("Maintain more consistent tone")
            
            # Pause analysis only after sufficient speech
            if self.speaking_duration > 5:
                try:
                    pause_ratio = self.pause_duration / max(self.speaking_duration, 0.1)
                    if pause_ratio > 0.3:
                        feedback.append("Too many pauses")
                    elif pause_ratio < 0.1:
                        feedback.append("Consider adding pauses")
                except:
                    pass  # Skip pause analysis if calculation fails
                    
        except Exception as e:
            print(f"Error generating speech feedback: {str(e)}")
            return "Audio analysis active..."
        
        return " | ".join(feedback) if feedback else "Start speaking..."
    

    def _analyze_pitch(self, audio_data: np.ndarray) -> Optional[float]:
        """Analyze pitch using autocorrelation"""
        try:
            # Apply window function
            windowed = audio_data * np.hanning(len(audio_data))
            
            # Calculate autocorrelation
            correlation = np.correlate(windowed, windowed, mode='full')
            correlation = correlation[len(correlation)//2:]
            
            # Find peaks
            peaks = []
            for i in range(1, len(correlation)-1):
                if correlation[i] > correlation[i-1] and correlation[i] > correlation[i+1]:
                    peaks.append(i)
            
            if peaks:
                # Convert first peak to frequency
                fundamental_freq = self.processor.sample_rate / peaks[0]
                return fundamental_freq
                
        except Exception as e:
            print(f"Error in pitch analysis: {str(e)}")
            
        return None
    

    def get_speech_stats(self) -> Dict[str, Any]:
        """Get current speech statistics"""
        return self.speech_stats.copy()
    
    def _save_audio(self, output_path: str):
        """Save recorded audio to file"""
        try:
            print(f"Saving audio to {output_path}")
            
            # Convert float32 to int16 for WAV file
            audio_data = np.concatenate([np.frombuffer(frame, dtype=np.float32) 
                                       for frame in self.audio_buffer])
            audio_data = (audio_data * 32767).astype(np.int16)
            
            with wave.open(output_path, 'wb') as wf:
                wf.setnchannels(self.processor.channels)
                wf.setsampwidth(2)  # 2 bytes for int16
                wf.setframerate(self.processor.sample_rate)
                wf.writeframes(audio_data.tobytes())
                
            print("Audio saved successfully")
            
        except Exception as e:
            print(f"Error saving audio file: {str(e)}")
    
    def process(self, frame: Optional[np.ndarray], audio_data: np.ndarray, 
                metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Process audio data and return metrics (implements FeatureModule interface)"""
        if audio_data is not None:
            self._process_audio(audio_data)
        return self.get_speech_stats()
    
    def get_feedback(self, priority: str = "all") -> List[str]:
        """Get current feedback messages (implements FeatureModule interface)"""
        feedback = self.get_speech_feedback()
        return feedback.split(" | ") if feedback else []
    
    def __del__(self):
        """Cleanup resources"""
        self.stop_recording()
        if hasattr(self, 'processor'):
            del self.processor