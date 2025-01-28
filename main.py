import cv2
import numpy as np
import time
import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import threading
import queue
import mediapipe as mp
from collections import deque
import pyaudio
import wave
from scipy.signal import welch
import os
import subprocess

from modules.utils import AnalyzerConfig, FeatureModule

# Import base analyzer components
from modules.video_analyzer import VideoAnalyzer
from modules.audio_analyzer import AudioAnalyzer
from modules.presentation_coach import PresentationCoach, CoachingDisplay

# Import real-time feedback modules
from modules.feedback_modules import (
    TimerModule,
    ConfidenceTracker,
    VoicePitchAnalyzer,
    RealTimeFeedbackManager
)

# Import behavioral analysis modules
from modules.behavioral_modules import (
    GestureAnalyzer,
    EyeContactTracker,
    MovementAnalyzer
)

# Import content analysis modules
from modules.content_modules import (
    SpeechAnalyzer,
    ContentAnalyzer
)

# Import emotional intelligence modules
from modules.emotion_modules import (
    EmotionalIntelligenceAnalyzer
)

# Import environment monitoring modules
from modules.environment_modules import (
    EnvironmentMonitor,
    BackgroundAnalyzer,
    AudioEnvironmentAnalyzer
)

# Import accessibility modules
from modules.accessibility_modules import (
    AccessibilityFeatures,
    CaptionGenerator,
    ColorEnhancer,
    GestureRecognizer,
    TextToSpeech
)

# Import common utilities
from modules.utils import AnalyzerConfig, FeatureModule

@dataclass
class PresentationConfig:
    """Global configuration for the presentation analyzer"""
    enable_real_time_feedback: bool = True
    enable_behavioral_analysis: bool = True
    enable_content_analysis: bool = True
    enable_emotional_intelligence: bool = True
    enable_environment_monitoring: bool = True
    enable_accessibility: bool = True
    recording_mode: str = "presentation"  # presentation, interview, pitch
    feedback_level: str = "detailed"  # basic, detailed, expert
    
    # Analysis settings
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    
    # Recording settings
    video_output_format: str = "mp4"
    audio_output_format: str = "wav"
    frame_rate: int = 30
    
    # UI settings
    show_metrics: bool = True
    show_feedback: bool = True
    show_alerts: bool = True

class IntegratedPresentationAnalyzer:
    def __init__(self, config_path: Optional[str] = "config.json"):
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Initialize queues for inter-module communication
        self.feedback_queue = queue.Queue()
        self.metrics_queue = queue.Queue()
        
        # Initialize module managers
        print("Initializing modules...")
        self.initialize_modules()  # Now passes config to all modules
        
        # UI state
        self.show_metrics = True
        self.paused = False
        self.recording = False
    
        print("Initialization complete")

    def _load_config(self, config_path: str) -> PresentationConfig:
        """Load configuration from file or use defaults"""
        try:
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
                return PresentationConfig(**config_dict)
        except FileNotFoundError:
            return PresentationConfig()

    def initialize_modules(self):
        """Initialize all analysis modules"""
        # Create default config for modules
        default_config = AnalyzerConfig()
        
        # Core analyzers
        self.video_analyzer = VideoAnalyzer(default_config)
        self.audio_analyzer = AudioAnalyzer(default_config)
        self.audio_analyzer.start_recording() 
        self.coach = PresentationCoach()
        self.display = CoachingDisplay()
        
        # Real-time feedback modules
        if self.config.enable_real_time_feedback:
            self.feedback_manager = RealTimeFeedbackManager()
            self.timer = TimerModule(default_config)
            self.confidence_tracker = ConfidenceTracker(default_config)
            self.voice_analyzer = VoicePitchAnalyzer(default_config)
        
        # Behavioral analysis modules
        if self.config.enable_behavioral_analysis:
            self.eye_tracker = EyeContactTracker(default_config)
            self.movement_analyzer = MovementAnalyzer(default_config)
            self.gesture_analyzer = GestureAnalyzer(default_config)
        
        # Content analysis modules
        if self.config.enable_content_analysis:
            self.speech_analyzer = SpeechAnalyzer(default_config)
            self.content_analyzer = ContentAnalyzer(default_config)
        
        # Emotional intelligence modules
        if self.config.enable_emotional_intelligence:
            self.emotion_analyzer = EmotionalIntelligenceAnalyzer(default_config)
        
        # Environment monitoring
        if self.config.enable_environment_monitoring:
            self.environment_monitor = EnvironmentMonitor(default_config)
            self.background_analyzer = BackgroundAnalyzer(default_config)
            self.audio_environment = AudioEnvironmentAnalyzer(default_config)
        
        # Accessibility features
        if self.config.enable_accessibility:
            self.accessibility = AccessibilityFeatures(default_config)
            self.caption_generator = CaptionGenerator(default_config)
            self.color_enhancer = ColorEnhancer(default_config)
            self.text_to_speech = TextToSpeech(default_config)
            self.gesture_recognizer = GestureRecognizer(default_config)

    def toggle_module(self, module_name: str) -> bool:
        """Toggle specific module on/off"""
        module_map = {
            'feedback': 'enable_real_time_feedback',
            'behavioral': 'enable_behavioral_analysis',
            'content': 'enable_content_analysis',
            'emotional': 'enable_emotional_intelligence',
            'environment': 'enable_environment_monitoring',
            'accessibility': 'enable_accessibility'
        }
        
        if module_name in module_map:
            attr_name = module_map[module_name]
            current_state = getattr(self.config, attr_name)
            setattr(self.config, attr_name, not current_state)
            self._reinitialize_module(module_name)
            return not current_state
        return False

    def _reinitialize_module(self, module_name: str):
        """Reinitialize specific module after toggle"""
        if module_name == 'feedback':
            if self.config.enable_real_time_feedback:
                self.feedback_manager = RealTimeFeedbackManager()
                self.timer = TimerModule(AnalyzerConfig())
                self.confidence_tracker = ConfidenceTracker(AnalyzerConfig())
            else:
                self.feedback_manager = None
                self.timer = None
                self.confidence_tracker = None
        # Add similar reinitializations for other modules...

    def process_frame(self, frame, audio_data) -> tuple:
        """Process a single frame through all enabled modules
        
        Args:
            frame: Video frame to process
            audio_data: Audio data if available
            
        Returns:
            tuple: (processed frame, metrics dictionary)
        """
        if not isinstance(frame, np.ndarray):
            print(f"Error: frame must be a numpy array, got {type(frame)}")
            return frame, {}
            
        if self.paused:
            return frame, {}
                
        metrics = {}
        enhanced_frame = frame.copy()
        
        try:
            # Core analysis
            video_metrics, video_feedback, annotated_frame = self.video_analyzer.analyze_frame(frame)
            metrics.update(video_metrics)
            enhanced_frame = annotated_frame if isinstance(annotated_frame, np.ndarray) else enhanced_frame
            
            if audio_data is not None:
                audio_metrics = self.audio_analyzer.process(frame, audio_data, metrics)
                metrics.update(audio_metrics)
            
            # Process through enabled modules
            if self.config.enable_real_time_feedback:
                feedback_metrics, feedback = self.feedback_manager.process_frame(
                    enhanced_frame, audio_data, metrics)
                metrics.update(feedback_metrics)
            
            if self.config.enable_behavioral_analysis:
                behavioral_metrics = self._process_behavioral(enhanced_frame, audio_data, metrics)
                metrics.update(behavioral_metrics)
            
            if self.config.enable_content_analysis:
                content_metrics = self._process_content(enhanced_frame, audio_data, metrics)
                metrics.update(content_metrics)
            
            if self.config.enable_emotional_intelligence:
                emotion_metrics = self.emotion_analyzer.process(enhanced_frame, audio_data, metrics)
                metrics.update(emotion_metrics)
            
            if self.config.enable_environment_monitoring:
                env_metrics = self._process_environment(enhanced_frame, audio_data, metrics)
                metrics.update(env_metrics)
            
            # Apply accessibility features last
            if self.config.enable_accessibility:
                processed_frame, acc_metrics = self._process_accessibility(
                    enhanced_frame, audio_data, metrics)
                if isinstance(processed_frame, np.ndarray):
                    enhanced_frame = processed_frame
                metrics.update(acc_metrics)
            
            # Update recording if active
            if self.recording:
                self.recorded_frames.append(enhanced_frame)
                
        except Exception as e:
            print(f"Error processing frame: {str(e)}")
            return frame, metrics  # Return original frame on error
            
        return enhanced_frame, metrics

    def _process_behavioral(self, frame, audio_data, metrics):
        """Process behavioral analysis modules"""
        behavioral_metrics = {}
        
        eye_metrics = self.eye_tracker.process(frame, audio_data, metrics)
        movement_metrics = self.movement_analyzer.process(frame, audio_data, metrics)
        gesture_metrics = self.gesture_analyzer.process(frame, audio_data, metrics)
        
        behavioral_metrics.update(eye_metrics)
        behavioral_metrics.update(movement_metrics)
        behavioral_metrics.update(gesture_metrics)
        
        return behavioral_metrics

    def _process_content(self, frame, audio_data, metrics):
        """Process content analysis modules"""
        content_metrics = {}
        
        speech_metrics = self.speech_analyzer.process(frame, audio_data, metrics)
        content_analysis = self.content_analyzer.process(frame, audio_data, metrics)
        
        content_metrics.update(speech_metrics)
        content_metrics.update(content_analysis)
        
        return content_metrics

    def _process_environment(self, frame, audio_data, metrics):
        """Process environment monitoring modules"""
        env_metrics = {}
        
        env_analysis = self.environment_monitor.process(frame, audio_data, metrics)
        background = self.background_analyzer.process(frame)
        audio_env = self.audio_environment.process(audio_data) if audio_data is not None else {}
        
        env_metrics.update(env_analysis)
        env_metrics.update(background)
        env_metrics.update(audio_env)
        
        return env_metrics

    def _process_accessibility(self, frame, audio_data, metrics):
        """Process accessibility features"""
        acc_metrics = {}
        
        # Generate captions if speech detected
        if audio_data is not None and 'transcribed_text' in metrics:
            captions = self.caption_generator.generate(metrics['transcribed_text'])
            acc_metrics['captions'] = captions
        
        # Enhance frame for accessibility
        enhanced_frame = self.color_enhancer.process(frame)
        
        # Process gestures for sign language
        if 'gesture_detected' in metrics and metrics['gesture_detected']:
            sign_language = self.gesture_recognizer.process(frame)
            acc_metrics['sign_language'] = sign_language
        
        return enhanced_frame, acc_metrics

    def start_analysis(self):
        """Start the presentation analysis"""
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise RuntimeError("Could not access webcam")
            
        print("\nPresentation Analysis Started")
        print("Controls:")
        print("'q': Quit")
        print("'p': Pause/Resume")
        print("'m': Toggle metrics display")
        print("'s': Save screenshot")
        print("'r': Toggle recording")
        print("'h': Show help")
        print("1-6: Toggle individual modules")
        print("=====================================")
        
        self._check_modules()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Get latest audio data from buffer
            audio_data = None
            if hasattr(self, 'audio_analyzer') and self.audio_analyzer.audio_buffer:
                try:
                    latest_audio = self.audio_analyzer.audio_buffer[-1]
                    audio_data = np.frombuffer(latest_audio, dtype=np.float32)
                except Exception as e:
                    print(f"Error getting audio data: {str(e)}")
                    
                                    
            # Process frame
            processed_frame, metrics = self.process_frame(frame, audio_data)  
            
            # Draw UI
            display_frame = self.draw_ui(processed_frame, metrics)
            
            # Display frame
            cv2.imshow('Presentation Analysis', display_frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('p'):
                self.paused = not self.paused
                status = "PAUSED" if self.paused else "RESUMED"
                print(f"Analysis {status}")
            elif key == ord('m'):
                self.show_metrics = not self.show_metrics
                print(f"Metrics display {'enabled' if self.show_metrics else 'disabled'}")
            elif key == ord('s'):
                self._save_screenshot(display_frame)
            elif key == ord('r'):
                self._toggle_recording()
            elif key == ord('h'):
                self._show_help()
            elif ord('1') <= key <= ord('6'):
                module_idx = key - ord('1')
                self._toggle_module_by_index(module_idx)
                
        # Cleanup
        if self.recording:
            self._stop_recording()
        cap.release()
        cv2.destroyAllWindows()

    def draw_ui(self, frame: np.ndarray, metrics: dict) -> np.ndarray:
        """Draw UI elements on frame
        
        Args:
            frame: Video frame to draw on
            metrics: Metrics dictionary
            
        Returns:
            np.ndarray: Frame with UI elements
        """
        if not isinstance(frame, np.ndarray):
            print(f"Error: frame must be a numpy array, got {type(frame)}")
            # Return a blank frame of default size
            return np.zeros((480, 640, 3), dtype=np.uint8)
            
        if not self.show_metrics:
            return frame
                
        h, w = frame.shape[:2]
        
        # Create semi-transparent overlay
        overlay = frame.copy()
        
        # Draw module-specific visualizations
        if hasattr(self, 'feedback_manager') and self.config.enable_real_time_feedback:
            overlay = self.feedback_manager.draw_feedback(overlay, metrics)
            
        if hasattr(self, 'behavioral_feedback') and self.config.enable_behavioral_analysis:
            overlay = self._draw_behavioral_feedback(overlay, metrics)
            
        if hasattr(self, 'emotion_analyzer') and self.config.enable_emotional_intelligence:
            overlay = self._draw_emotional_feedback(overlay, metrics)
            
        if hasattr(self, 'environment_monitor') and self.config.enable_environment_monitoring:
            overlay = self._draw_environment_status(overlay, metrics)
            
        # Add general metrics panel
        overlay = self._draw_metrics_panel(overlay, metrics)
        
        # Blend overlay with original frame
        return cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)

    def _draw_metrics_panel(self, frame: np.ndarray, metrics: dict) -> np.ndarray:
        """Draw metrics panel with improved layout and organization"""
        h, w = frame.shape[:2]
        
        # Panel configuration
        panel_width = 550  # Increased width for better readability
        panel_start = w - panel_width
        panel_opacity = 0.85  # Increased opacity for better contrast
        
        # Create semi-transparent overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (panel_start, 0), (w, h), (0, 0, 0), -1)
        
        # Drawing settings
        font = cv2.FONT_HERSHEY_SIMPLEX
        title_scale = 1.9
        text_scale = 1.7
        font_thickness = 2
        line_spacing = 50
        margin = 20
        
        y_offset = margin
        
        # Helper function to add metric text with improved formatting
        def add_metric_text(label, value, color=(255, 255, 255)):
            nonlocal y_offset
            text = f"{label}: {value}"
            cv2.putText(overlay, text, 
                    (panel_start + margin, y_offset),
                    font, text_scale, color, font_thickness)
            y_offset += line_spacing
        
        # Draw title with background
        title = "Presentation Metrics"
        title_bg_height = 50
        cv2.rectangle(overlay, 
                    (panel_start, 0),
                    (w, title_bg_height),
                    (40, 40, 40), -1)  # Dark background for title
        cv2.putText(overlay, title,
                (panel_start + margin, y_offset + 10),
                font, title_scale,
                (255, 223, 0),  # Gold color for title
                font_thickness)
        y_offset += title_bg_height + margin

        # Section headers style
        def add_section_header(text):
            nonlocal y_offset
            y_offset += 10  # Extra spacing before section
            cv2.putText(overlay, text,
                    (panel_start + margin, y_offset),
                    font, text_scale,
                    (0, 191, 255),  # Deep sky blue for headers
                    font_thickness)
            y_offset += line_spacing
        
        # 1. Audio Metrics Section
        add_section_header("Speech")
        if 'volume_level' in metrics:
            volume = metrics['volume_level']
            color = (0, 255, 0) if 0.1 <= volume <= 0.8 else (0, 165, 255)
            add_metric_text("Volume", f"{volume:.2f}", color)
        
        if 'speech_rate' in metrics:
            rate = metrics['speech_rate']
            color = (0, 255, 0) if 120 <= rate <= 160 else (0, 165, 255)
            add_metric_text("Speech Rate", f"{rate:.0f} wpm", color)

        if 'pitch_variation' in metrics:
            pitch_var = metrics['pitch_variation']
            color = (0, 255, 0) if 10 <= pitch_var <= 50 else (0, 165, 255)
            add_metric_text("Pitch Variation", f"{pitch_var:.1f}", color)

        # 2. Behavioral Metrics Section
        add_section_header("Behavior")
        if 'posture' in metrics:
            posture = metrics['posture']
            color = (0, 255, 0) if posture == "Good Posture" else (0, 165, 255)
            add_metric_text("Posture", posture, color)
        
        if 'eye_contact_score' in metrics:
            score = metrics['eye_contact_score']
            color = (0, 255, 0) if score >= 0.7 else (0, 165, 255)
            add_metric_text("Eye Contact", f"{score:.2f}", color)

        if 'gesture_score' in metrics:
            score = metrics['gesture_score']
            color = (0, 255, 0) if score >= 0.7 else (0, 165, 255)
            add_metric_text("Gestures", f"{score:.2f}", color)

        # 3. Emotional Metrics Section
        add_section_header("Expression")
        if 'dominant_emotion' in metrics:
            emotion = metrics['dominant_emotion']
            confidence = metrics.get('emotion_confidence', 0)
            add_metric_text("Emotion", f"{emotion} ({confidence:.1%})")

        # 4. Environment Metrics (if available)
        if any(key in metrics for key in ['lighting_quality', 'background_quality']):
            add_section_header("Environment")
            if 'lighting_quality' in metrics:
                quality = metrics['lighting_quality']
                color = (0, 255, 0) if quality >= 0.7 else (0, 165, 255)
                add_metric_text("Lighting", f"{quality:.2f}", color)
            
            if 'background_quality' in metrics:
                quality = metrics['background_quality']
                color = (0, 255, 0) if quality >= 0.7 else (0, 165, 255)
                add_metric_text("Background", f"{quality:.2f}", color)

        # Apply the overlay with proper opacity
        return cv2.addWeighted(frame, 1 - panel_opacity, overlay, panel_opacity, 0)

    def _draw_behavioral_feedback(self, frame: np.ndarray, metrics: dict) -> np.ndarray:
        """Draw comprehensive behavioral feedback overlay"""
        h, w = frame.shape[:2]
        padding = 10
        
        # Create semi-transparent overlay for feedback area
        feedback_height = 120
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h-feedback_height), (w//2, h), 
                    (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
        
        # Starting position for feedback
        y_offset = h - feedback_height + padding
        x_start = padding
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        
        # Draw title
        cv2.putText(frame, "Behavioral Feedback", 
                    (x_start, y_offset), font, 0.7,
                    (255, 255, 255), 2)
        y_offset += 25
        
        # Draw feedback items
        def add_feedback(text, score=None):
            nonlocal y_offset
            if score is not None:
                color = (0, 255, 0) if score >= 0.7 else (0, 165, 255)
                text = f"{text}: {score:.2f}"
            else:
                color = (255, 255, 255)
                
            cv2.putText(frame, text, (x_start, y_offset),
                    font, font_scale, color, 1)
            y_offset += 20
        
        # Eye contact feedback
        if 'eye_contact_score' in metrics:
            add_feedback("Eye Contact", metrics['eye_contact_score'])
            
        # Gesture feedback
        if 'gesture_score' in metrics:
            add_feedback("Gesture Quality", metrics['gesture_score'])
            
        # Posture feedback
        if 'posture' in metrics:
            posture_score = 1.0 if metrics['posture'] == "Good Posture" else 0.5
            add_feedback("Posture", posture_score)
            
        # Movement feedback
        if 'movement_level' in metrics:
            movement_score = 1.0 - metrics['movement_level']  # Invert so less movement = higher score
            add_feedback("Stability", movement_score)
        
        return frame


    def _draw_emotional_feedback(self, frame: np.ndarray, metrics: dict) -> np.ndarray:
        """Draw emotional feedback overlay"""
        if 'dominant_emotion' in metrics:
            emotion = metrics['dominant_emotion']
            confidence = metrics.get('emotion_confidence', 0)
            
            text = f"Emotion: {emotion} ({confidence:.2f})"
            cv2.putText(frame, text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 255), 2)
        
        return frame

    def _draw_environment_status(self, frame: np.ndarray, metrics: dict) -> np.ndarray:
        """Draw environment status overlay"""
        h, w = frame.shape[:2]
        
        if 'lighting_quality' in metrics:
            quality = metrics['lighting_quality']
            text = f"Lighting: {quality:.2f}"
            cv2.putText(frame, text, (10, h-30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 255), 2)
        
        return frame

    def _draw_gauge(self, frame: np.ndarray, label: str, value: float, 
                    position: tuple, width: int = 150, height: int = 20):
        """Draw a gauge indicator"""
        x, y = position
        
        # Draw background
        cv2.rectangle(frame, (x, y), (x + width, y + height),
                    (255, 255, 255), 1)
        
        # Draw filled portion
        fill_width = int(width * value)
        color = (0, 255, 0) if value >= 0.7 else (0, 165, 255)
        cv2.rectangle(frame, (x, y), (x + fill_width, y + height),
                    color, -1)
        
        # Draw label
        cv2.putText(frame, f"{label}: {value:.2f}", (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        

    def _check_modules(self):
        """Verify all modules are properly initialized and started"""
        try:
            if hasattr(self, 'audio_analyzer'):
                if not self.audio_analyzer.is_recording:
                    print("Starting audio analyzer...")
                    self.audio_analyzer.start_recording()
            
            # Initialize feedback manager if needed
            if self.config.enable_real_time_feedback and not hasattr(self, 'feedback_manager'):
                print("Initializing feedback manager...")
                self.feedback_manager = RealTimeFeedbackManager()
            
            # Verify all required modules are present
            required_modules = {
                'video_analyzer': 'Core video analysis',
                'audio_analyzer': 'Core audio analysis',
                'feedback_manager': 'Real-time feedback',
                'eye_tracker': 'Eye tracking',
                'movement_analyzer': 'Movement analysis',
                'gesture_analyzer': 'Gesture analysis'
            }
            
            missing_modules = []
            for module, description in required_modules.items():
                if not hasattr(self, module):
                    missing_modules.append(f"{description} ({module})")
            
            if missing_modules:
                print("Warning: Some modules are missing:")
                for module in missing_modules:
                    print(f" - {module}")
                    
        except Exception as e:
            print(f"Error checking modules: {str(e)}")



def main():
    try:
        analyzer = IntegratedPresentationAnalyzer()
        analyzer.start_analysis()
    except Exception as e:
        print(f"Error: {str(e)}")
        raise

if __name__ == "__main__":
    main()