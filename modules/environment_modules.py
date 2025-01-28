import cv2
import numpy as np
from collections import deque
import time 
from typing import Dict, List, Optional, Tuple
from .utils import FeatureModule, AnalyzerConfig

class EnvironmentMonitor(FeatureModule):
    def __init__(self, config: AnalyzerConfig):
        super().__init__(config)
        self.light_analyzer = LightingAnalyzer(config)
        self.background_analyzer = BackgroundAnalyzer(config)
        self.audio_environment = AudioEnvironmentAnalyzer(config)
        self.history = deque(maxlen=30)
        
    def process(self, frame, audio_data, metrics: dict) -> dict:
        # Analyze lighting
        lighting_metrics = self.light_analyzer.process(frame, audio_data, metrics)
        
        # Analyze background
        background_metrics = self.background_analyzer.process(frame, audio_data, metrics)
        
        # Analyze audio environment
        audio_metrics = self.audio_environment.process(frame, audio_data, metrics)
        
        # Combine metrics
        combined_metrics = {
            **lighting_metrics,
            **background_metrics,
            **audio_metrics,
            'environment_score': self._calculate_environment_score(
                lighting_metrics,
                background_metrics,
                audio_metrics
            )
        }
        
        self.history.append(combined_metrics)
        
        return combined_metrics
    
    def _calculate_environment_score(self, lighting: dict, background: dict, 
                                   audio: dict) -> float:
        """Calculate overall environment quality score"""
        weights = {
            'lighting': 0.4,
            'background': 0.3,
            'audio': 0.3
        }
        
        scores = {
            'lighting': lighting.get('lighting_quality', 0.0),
            'background': background.get('background_quality', 0.0),
            'audio': audio.get('audio_quality', 0.0)
        }
        
        total_score = sum(weights[k] * scores[k] for k in weights)
        return max(0.0, min(1.0, total_score))

class LightingAnalyzer(FeatureModule):
    def __init__(self, config: AnalyzerConfig):
        super().__init__(config)
        self.light_history = deque(maxlen=30)  # Store last 30 frames of lighting data
    
    def process(self, frame, audio_data, metrics: dict) -> dict:
        if frame is None:
            return {}
            
        # Convert to grayscale for light analysis
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Analyze lighting conditions
        brightness = np.mean(gray)
        contrast = np.std(gray)
        uniformity = self._calculate_lighting_uniformity(gray)
        
        # Calculate lighting quality
        quality = self._calculate_lighting_quality(brightness, contrast, uniformity)
        
        metrics = {
            'brightness_level': brightness,
            'contrast_level': contrast,
            'lighting_uniformity': uniformity,
            'lighting_quality': quality,
            'lighting_feedback': self._generate_feedback(brightness, contrast, uniformity)
        }
        
        self.light_history.append(metrics)
        return metrics
        
    def _calculate_lighting_uniformity(self, gray: np.ndarray) -> float:
        """Calculate how uniform the lighting is across the frame
        
        Args:
            gray: Grayscale image
        
        Returns:
            float: Uniformity score between 0 and 1
        """
        # Divide image into regions and compare brightness
        h, w = gray.shape
        regions = [
            gray[:h//2, :w//2],     # top-left
            gray[:h//2, w//2:],     # top-right
            gray[h//2:, :w//2],     # bottom-left
            gray[h//2:, w//2:]      # bottom-right
        ]
        
        means = [np.mean(region) for region in regions]
        mean_brightness = np.mean(means)
        
        if mean_brightness == 0:
            return 0.0
            
        std_brightness = np.std(means)
        uniformity = 1.0 - (std_brightness / mean_brightness)
        
        return max(0.0, min(1.0, uniformity))

    def _calculate_lighting_quality(self, brightness: float, contrast: float, 
                                  uniformity: float) -> float:
        """Calculate overall lighting quality score"""
        # Ideal ranges
        ideal_brightness = (100, 200)  # out of 255
        ideal_contrast = (40, 80)
        min_uniformity = 0.7
        
        # Calculate individual scores
        brightness_score = 1.0 - min(abs(brightness - np.mean(ideal_brightness)) / 255, 1.0)
        contrast_score = 1.0 - min(abs(contrast - np.mean(ideal_contrast)) / 255, 1.0)
        uniformity_score = uniformity if uniformity >= min_uniformity else uniformity/min_uniformity
        
        # Weighted combination
        weights = {'brightness': 0.4, 'contrast': 0.3, 'uniformity': 0.3}
        quality = (weights['brightness'] * brightness_score +
                  weights['contrast'] * contrast_score +
                  weights['uniformity'] * uniformity_score)
                  
        return max(0.0, min(1.0, quality))
        
    
    def _generate_feedback(self, brightness: float, contrast: float, 
                          uniformity: float) -> List[str]:
        """Generate feedback about lighting conditions"""
        feedback = []
        
        if brightness < 100:
            feedback.append("Increase lighting brightness")
        elif brightness > 200:
            feedback.append("Reduce lighting brightness")
            
        if contrast < 40:
            feedback.append("Increase lighting contrast")
        elif contrast > 80:
            feedback.append("Reduce lighting contrast")
            
        if uniformity < 0.7:
            feedback.append("Improve lighting uniformity")
            
        return feedback

class BackgroundAnalyzer(FeatureModule):
    def __init__(self, config: AnalyzerConfig):
        super().__init__(config)
        self.background_history = deque(maxlen=30)
        self.background_model = None
        
    def process(self, frame, audio_data=None, metrics=None) -> dict:
        """Analyze background and generate metrics
        
        Args:
            frame: Video frame to analyze
            audio_data: Optional audio data (not used for background analysis)
            metrics: Optional metrics dictionary (not used for background analysis)
            
        Returns:
            dict: Background analysis metrics
        """
        if frame is None:
            return {}
                
        # Initialize background model if needed
        if self.background_model is None:
            self.background_model = cv2.createBackgroundSubtractorMOG2()
                
        try:
            # Analyze background
            foreground_mask = self.background_model.apply(frame)
            
            # Calculate metrics
            clutter = self._analyze_clutter(frame)
            movement = self._analyze_movement(foreground_mask)
            distractions = self._detect_distractions(frame)
            
            metrics = {
                'background_clutter': clutter,
                'background_movement': movement,
                'distraction_level': len(distractions),
                'background_quality': self._calculate_background_quality(clutter, movement, len(distractions)),
                'background_feedback': self._generate_feedback(clutter, movement, distractions)
            }
            
            self.background_history.append(metrics)
            return metrics
            
        except Exception as e:
            print(f"Error analyzing background: {str(e)}")
            return {
                'background_clutter': 0,
                'background_movement': 0,
                'distraction_level': 0,
                'background_quality': 0.5,
                'background_feedback': []
            }
        
    def _analyze_clutter(self, frame: np.ndarray) -> float:
        """Analyze visual clutter in the background"""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate edge density
        edges = cv2.Canny(gray, 100, 200)
        edge_density = np.mean(edges > 0)
        
        return min(1.0, edge_density * 5)  # Scale up for better sensitivity
        
    def _analyze_movement(self, foreground_mask: np.ndarray) -> float:
        """Analyze movement in the background"""
        return np.mean(foreground_mask > 0)
        
    def _detect_distractions(self, frame: np.ndarray) -> List[str]:
        """Detect potential distractions in the background"""
        distractions = []
        
        # Implement distraction detection (e.g., moving objects, bright spots)
        # This is a placeholder for actual implementation
        
        return distractions
        
    def _calculate_background_quality(self, clutter: float, movement: float, 
                                    num_distractions: int) -> float:
        """Calculate overall background quality score"""
        # Weight factors
        weights = {'clutter': 0.4, 'movement': 0.4, 'distractions': 0.2}
        
        # Calculate individual scores (lower is better)
        clutter_score = 1.0 - clutter
        movement_score = 1.0 - movement
        distraction_score = 1.0 - min(1.0, num_distractions / 5)
        
        # Combine scores
        quality = (weights['clutter'] * clutter_score +
                  weights['movement'] * movement_score +
                  weights['distractions'] * distraction_score)
                  
        return max(0.0, min(1.0, quality))
        
    def _generate_feedback(self, clutter: float, movement: float, 
                          distractions: List[str]) -> List[str]:
        """Generate feedback about background conditions"""
        feedback = []
        
        if clutter > 0.3:
            feedback.append("Reduce background clutter")
            
        if movement > 0.2:
            feedback.append("Reduce background movement")
            
        if distractions:
            feedback.append(f"Remove distracting elements: {', '.join(distractions)}")
            
        return feedback

class AudioEnvironmentAnalyzer(FeatureModule):
    def __init__(self, config: AnalyzerConfig):
        super().__init__(config)
        self.noise_history = deque(maxlen=30)
        self.echo_detector = EchoDetector()
        
    def process(self, frame, audio_data, metrics: dict) -> dict:
        if audio_data is None:
            return {}
            
        # Analyze audio environment
        noise_level = self._calculate_noise_level(audio_data)
        echo = self.echo_detector.detect_echo(audio_data)
        room_acoustics = self._analyze_room_acoustics(audio_data)
        
        metrics = {
            'noise_level': noise_level,
            'echo_level': echo,
            'room_acoustics': room_acoustics,
            'audio_quality': self._calculate_audio_quality(noise_level, echo, room_acoustics),
            'audio_feedback': self._generate_feedback(noise_level, echo, room_acoustics)
        }
        
        self.noise_history.append(metrics)
        return metrics
        
    def _calculate_noise_level(self, audio_data: np.ndarray) -> float:
        """Calculate background noise level"""
        return np.std(audio_data)
        
    def _analyze_room_acoustics(self, audio_data: np.ndarray) -> float:
        """Analyze room acoustic properties"""
        # Placeholder for acoustic analysis
        return 0.8
        
    def _calculate_audio_quality(self, noise: float, echo: float, 
                               acoustics: float) -> float:
        """Calculate overall audio environment quality"""
        weights = {'noise': 0.4, 'echo': 0.3, 'acoustics': 0.3}
        
        # Convert metrics to scores (higher is better)
        noise_score = 1.0 - min(1.0, noise * 5)
        echo_score = 1.0 - echo
        acoustics_score = acoustics
        
        # Combine scores
        quality = (weights['noise'] * noise_score +
                  weights['echo'] * echo_score +
                  weights['acoustics'] * acoustics_score)
                  
        return max(0.0, min(1.0, quality))
        
    def _generate_feedback(self, noise: float, echo: float, 
                          acoustics: float) -> List[str]:
        """Generate feedback about audio environment"""
        feedback = []
        
        if noise > 0.2:
            feedback.append("Reduce background noise")
            
        if echo > 0.3:
            feedback.append("Reduce room echo")
            
        if acoustics < 0.6:
            feedback.append("Improve room acoustics")
            
        return feedback

class EchoDetector:
    def __init__(self):
        self.buffer_size = 1024
        self.echo_threshold = 0.3
        
    def detect_echo(self, audio_data: np.ndarray) -> float:
        """Detect echo in audio signal"""
        # Placeholder for echo detection
        return 0.1

class EnvironmentFeedbackManager:
    def __init__(self, config: AnalyzerConfig):
        self.monitor = EnvironmentMonitor(config)
        self.feedback_history = deque(maxlen=50)
        
    def process_frame(self, frame, audio_data, metrics: dict) -> tuple:
        """Process frame and generate environment feedback"""
        # Analyze environment
        env_metrics = self.monitor.process(frame, audio_data, metrics)
        
        # Generate feedback
        feedback = self._generate_feedback(env_metrics)
        
        # Update history
        self.feedback_history.append((env_metrics, feedback))
        
        return env_metrics, feedback
        
    def _generate_feedback(self, metrics: dict) -> List[str]:
        """Generate environment-related feedback"""
        feedback = []
        
        # Lighting feedback
        if metrics.get('lighting_quality', 1.0) < 0.6:
            feedback.extend(metrics.get('lighting_feedback', []))
            
        # Background feedback
        if metrics.get('background_quality', 1.0) < 0.6:
            feedback.extend(metrics.get('background_feedback', []))
            
        # Audio environment feedback
        if metrics.get('audio_quality', 1.0) < 0.6:
            feedback.extend(metrics.get('audio_feedback', []))
            
        return feedback

