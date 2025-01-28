import pyaudio
import wave
import numpy as np
import threading
from collections import deque
import time

class AudioAnalyzer:
    def __init__(self):
        self.audio = pyaudio.PyAudio()
        self.frames = []
        self.is_recording = False
        self.stream = None
        
        # Audio settings
        self.channels = 1  # Changed to mono for simplicity
        self.rate = 44100
        self.format = pyaudio.paFloat32
        self.chunk_size = 1024
        
        # Analysis buffers
        self.energy_buffer = deque(maxlen=50)
        self.volume_threshold = 0.01
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
            'pause_count': 0
        }
        
        print("AudioAnalyzer initialized")
        
    def start_recording(self):
        if self.is_recording:
            return
            
        print("Starting audio recording...")
        self.frames = []
        self.is_recording = True
        self.last_process_time = time.time()
        
        try:
            # Open audio stream for recording
            self.stream = self.audio.open(
                format=self.format,
                channels=self.channels,
                rate=self.rate,
                input=True,
                frames_per_buffer=self.chunk_size,
                stream_callback=self._audio_callback
            )
            
            print("Audio stream opened successfully")
            
        except Exception as e:
            print(f"Error starting audio recording: {e}")
            self.is_recording = False
            if self.stream:
                self.stream.close()
            self.stream = None

    def _audio_callback(self, in_data, frame_count, time_info, status):
        if self.is_recording:
            try:
                # Store raw audio data
                self.frames.append(in_data)
                
                # Process audio for analysis
                audio_data = np.frombuffer(in_data, dtype=np.float32)
                self._process_audio(audio_data)
                
                return (in_data, pyaudio.paContinue)
            except Exception as e:
                print(f"Error in audio callback: {e}")
                return (in_data, pyaudio.paContinue)
        return (None, pyaudio.paComplete)

    def _process_audio(self, audio_data):
        try:
            # Calculate volume level
            volume = np.abs(audio_data).mean()
            self.energy_buffer.append(volume)
            self.speech_stats['volume_level'] = volume
            
            # Detect speech activity
            is_speaking = volume > self.volume_threshold
            current_time = time.time()
            
            if self.last_process_time is None:
                self.last_process_time = current_time
                return
            
            time_diff = current_time - self.last_process_time
            
            if is_speaking:
                if not self.speech_detected:
                    self.speech_stats['pause_count'] += 1
                    print(f"Speech detected! Volume: {volume:.4f}")
                self.speech_detected = True
                self.speaking_duration += time_diff
            else:
                if self.speech_detected:
                    print(f"Speech ended. Duration: {self.speaking_duration:.2f}s")
                self.speech_detected = False
                self.pause_duration += time_diff
            
            # Update speech rate
            if self.speaking_duration > 0:
                # Estimate words based on speaking duration
                estimated_words = (self.speaking_duration * 2)  # Assume 2 words per second
                self.speech_stats['speech_rate'] = (estimated_words * 60) / self.speaking_duration
            
            self.speech_stats['speaking_duration'] = self.speaking_duration
            self.last_process_time = current_time
            
        except Exception as e:
            print(f"Error processing audio: {e}")

    def get_speech_feedback(self):
        """Generate feedback based on speech analysis."""
        feedback = []
        
        # Volume feedback
        vol = self.speech_stats['volume_level']
        if vol < 0.01:
            feedback.append("Speak louder")
        elif vol > 0.5:
            feedback.append("Lower your voice")
        elif vol > 0.02:
            feedback.append("Good volume")
        
        # Speech rate feedback
        wpm = self.speech_stats['speech_rate']
        if self.speaking_duration > 1:
            if wpm > 160:
                feedback.append("Speaking too fast")
            elif wpm < 80:
                feedback.append("Speaking too slow")
            else:
                feedback.append("Good pace")
        
        # Pause analysis
        if self.speaking_duration > 5:
            if self.speech_stats['pause_count'] > 5:
                feedback.append("Too many pauses")
        
        return " | ".join(feedback) if feedback else "Start speaking..."

    def get_speech_stats(self):
        """Get current speech statistics."""
        return self.speech_stats

    def stop_recording(self, output_path=None):
        print("Stopping audio recording...")
        if not self.is_recording:
            return
            
        self.is_recording = False
        
        if self.stream:
            try:
                self.stream.stop_stream()
                self.stream.close()
                print("Audio stream closed")
            except Exception as e:
                print(f"Error closing audio stream: {e}")
        
        if output_path and self.frames:
            self.save_audio(output_path)
    
    def save_audio(self, output_path):
        try:
            print(f"Saving audio to {output_path}")
            # Convert float32 to int16 for WAV file
            audio_data = np.concatenate([np.frombuffer(frame, dtype=np.float32) 
                                       for frame in self.frames])
            audio_data = (audio_data * 32767).astype(np.int16)
            
            with wave.open(output_path, 'wb') as wf:
                wf.setnchannels(self.channels)
                wf.setsampwidth(2)  # 2 bytes for int16
                wf.setframerate(self.rate)
                wf.writeframes(audio_data.tobytes())
            print("Audio saved successfully")
        except Exception as e:
            print(f"Error saving audio file: {e}")
    
    def __del__(self):
        try:
            self.audio.terminate()
            print("Audio system terminated")
        except:
            pass