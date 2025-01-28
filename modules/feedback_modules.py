import cv2
import numpy as np
from collections import deque
from typing import Dict, List, Optional, Tuple
import time
from .utils import FeatureModule, AnalyzerConfig


class TimerModule(FeatureModule):
    def __init__(self, config: AnalyzerConfig):
        super().__init__(config)
        self.start_time = None
        self.planned_duration = 300  # 5 minutes default
        self.sections = {
            'introduction': 0.2,    # 20% of total time
            'main_content': 0.6,    # 60% of total time
            'conclusion': 0.2       # 20% of total time
        }
        
    def set_duration(self, minutes: int):
        self.planned_duration = minutes * 60
        
    def process(self, frame, audio_data, metrics: dict) -> dict:
        if not self.start_time:
            self.start_time = time.time()
            
        elapsed = time.time() - self.start_time
        remaining = max(0, self.planned_duration - elapsed)
        
        current_section = self._get_current_section(elapsed)
        
        return {
            'elapsed_time': elapsed,
            'remaining_time': remaining,
            'current_section': current_section,
            'time_warning': self._get_time_warning(elapsed)
        }
    
    def _get_current_section(self, elapsed: float) -> str:
        progress = elapsed / self.planned_duration
        if progress < self.sections['introduction']:
            return 'introduction'
        elif progress < (self.sections['introduction'] + self.sections['main_content']):
            return 'main_content'
        else:
            return 'conclusion'
            
    def _get_time_warning(self, elapsed: float) -> Optional[str]:
        progress = elapsed / self.planned_duration
        if progress > 0.9:
            return "WARNING: Less than 10% of time remaining"
        elif progress > 0.8:
            return "Note: Approach conclusion"
        return None

class ConfidenceTracker(FeatureModule):
    def __init__(self, config: AnalyzerConfig):
        super().__init__(config)
        self.confidence_history = []
        self.factors = {
            'posture': 0.2,
            'voice_stability': 0.2,
            'eye_contact': 0.2,
            'gesture_naturalness': 0.2,
            'speech_clarity': 0.2
        }
        
    def process(self, frame, audio_data, metrics: dict) -> dict:
        confidence_score = self._calculate_confidence(metrics)
        self.confidence_history.append(confidence_score)
        
        if len(self.confidence_history) > 100:
            self.confidence_history.pop(0)
            
        return {
            'confidence_score': confidence_score,
            'confidence_trend': self._analyze_trend()
        }
        
    def _calculate_confidence(self, metrics: dict) -> float:
        score = 0
        for factor, weight in self.factors.items():
            if factor in metrics:
                score += metrics[factor] * weight
        return min(max(score, 0), 100)
        
    def _analyze_trend(self) -> str:
        if len(self.confidence_history) < 10:
            return "Insufficient data"
            
        recent = np.mean(self.confidence_history[-10:])
        older = np.mean(self.confidence_history[:-10])
        
        if recent > older + 5:
            return "Improving"
        elif recent < older - 5:
            return "Declining"
        return "Stable"

class VoicePitchAnalyzer(FeatureModule):
    def __init__(self, config: AnalyzerConfig):
        super().__init__(config)
        self.pitch_history = []
        self.monotone_threshold = 10
        self.variation_target = (20, 40)  # Ideal pitch variation range
        
    def process(self, frame, audio_data, metrics: dict) -> dict:
        if not audio_data:
            return {}
            
        pitch = self._extract_pitch(audio_data)
        if pitch:
            self.pitch_history.append(pitch)
            
        if len(self.pitch_history) > 100:
            self.pitch_history.pop(0)
            
        variation = np.std(self.pitch_history) if self.pitch_history else 0
        
        return {
            'pitch_variation': variation,
            'monotone_warning': variation < self.monotone_threshold,
            'pitch_feedback': self._generate_pitch_feedback(variation)
        }
        
    def _extract_pitch(self, audio_data: np.ndarray) -> Optional[float]:
        # Implement pitch extraction using librosa or similar
        # This is a placeholder
        return None
        
    def _generate_pitch_feedback(self, variation: float) -> str:
        if variation < self.monotone_threshold:
            return "Try varying your tone more to maintain engagement"
        elif variation > self.variation_target[1]:
            return "Consider maintaining a more consistent tone"
        return "Good vocal variation"

class RealTimeFeedbackManager:
    def __init__(self):
        self.display_config = {
            'font': cv2.FONT_HERSHEY_SIMPLEX,
            'font_scale': 1.8,  # Increased font size
            'font_thickness': 3,
            'margin': 20,  # Increased margin
            'line_spacing': 60,  # Increased line spacing
            'panel_width': 750,  # Fixed panel width
            'panel_opacity': 0.95,  # Increased opacity for better readability
            'colors': {
                'success': (0, 255, 0),    # Green
                'warning': (0, 165, 255),  # Orange
                'error': (0, 0, 255),      # Red
                'info': (255, 255, 255),   # White
                'title': (255, 223, 0)     # Gold for title
            }
        }
        self.feedback_history = deque(maxlen=5)

    def process_frame(self, frame, audio_data, metrics: dict) -> tuple:
        """Generate comprehensive feedback based on all available metrics"""
        feedback_items = []
        feedback_metrics = {}

        # 1. Audio/Speech Feedback
        if 'volume_level' in metrics:
            vol = metrics['volume_level']
            if vol < 0.01:
                feedback_items.append({'text': "Speak louder", 'type': 'warning'})
            elif vol > 0.5:
                feedback_items.append({'text': "Lower your voice", 'type': 'warning'})
            elif vol > 0.02:
                feedback_items.append({'text': "Good volume", 'type': 'success'})

        if 'speech_rate' in metrics:
            rate = metrics['speech_rate']
            if rate > 160:
                feedback_items.append({'text': "Speaking too fast", 'type': 'warning'})
            elif rate < 80 and rate > 0:
                feedback_items.append({'text': "Speaking too slow", 'type': 'warning'})
            elif rate > 0:
                feedback_items.append({'text': "Good speaking pace", 'type': 'success'})

        if 'pitch_variation' in metrics:
            pitch_var = metrics['pitch_variation']
            if pitch_var < 10:
                feedback_items.append({'text': "Add more vocal variety", 'type': 'warning'})
            elif pitch_var > 50:
                feedback_items.append({'text': "Maintain consistent tone", 'type': 'warning'})

        # 2. Posture and Movement Feedback
        if 'posture' in metrics:
            posture = metrics['posture']
            if isinstance(posture, str):
                if posture == "Good Posture":
                    feedback_items.append({'text': "Good posture", 'type': 'success'})
                else:
                    feedback_items.append({'text': f"Fix posture: {posture}", 'type': 'warning'})

        if 'movement_level' in metrics:
            movement = metrics['movement_level']
            if movement > 0.7:
                feedback_items.append({'text': "Too much movement", 'type': 'warning'})
            elif movement < 0.1:
                feedback_items.append({'text': "Try to be more dynamic", 'type': 'warning'})

        # 3. Eye Contact and Engagement
        if 'eye_contact_score' in metrics:
            eye_score = metrics['eye_contact_score']
            if eye_score < 0.5:
                feedback_items.append({'text': "Maintain eye contact", 'type': 'warning'})
            elif eye_score > 0.8:
                feedback_items.append({'text': "Good eye contact", 'type': 'success'})

        # 4. Expression and Emotion
        if 'dominant_emotion' in metrics:
            emotion = metrics['dominant_emotion']
            confidence = metrics.get('emotion_confidence', 0)
            if confidence > 0.7:
                feedback_items.append({'text': f"Expression: {emotion}", 'type': 'info'})

        # 5. Environment Feedback
        if 'lighting_quality' in metrics:
            quality = metrics['lighting_quality']
            if quality < 0.5:
                feedback_items.append({'text': "Improve lighting", 'type': 'warning'})

        if 'background_quality' in metrics:
            bg_quality = metrics['background_quality']
            if bg_quality < 0.5:
                feedback_items.append({'text': "Check background", 'type': 'warning'})

        # 6. Gesture Feedback
        if 'gesture_score' in metrics:
            gesture = metrics['gesture_score']
            if gesture < 0.3:
                feedback_items.append({'text': "Use more gestures", 'type': 'warning'})
            elif gesture > 0.8:
                feedback_items.append({'text': "Good gesture use", 'type': 'success'})

        # Update feedback history
        if feedback_items:
            self.feedback_history.append(feedback_items)

        return feedback_metrics, feedback_items

    def draw_feedback(self, frame: np.ndarray, metrics: dict) -> np.ndarray:
        """Draw feedback with improved layout and readability"""
        h, w = frame.shape[:2]
        
        # Create semi-transparent overlay for feedback panel
        overlay = frame.copy()
        
        # Calculate panel dimensions
        panel_width = min(self.display_config['panel_width'], w - 40)  # Leave margin
        panel_height = h // 2  # Use half of screen height
        panel_x = 20  # Left margin
        panel_y = h - panel_height - 20  # Bottom margin
        
        # Draw main panel background
        cv2.rectangle(overlay, 
                     (panel_x, panel_y),
                     (panel_x + panel_width, panel_y + panel_height),
                     (0, 0, 0), -1)
        
        # Add title bar
        title_height = 50
        cv2.rectangle(overlay,
                     (panel_x, panel_y),
                     (panel_x + panel_width, panel_y + title_height),
                     (40, 40, 40), -1)  # Darker background for title
        
        # Draw title
        cv2.putText(overlay, "Real-Time Feedback",
                   (panel_x + self.display_config['margin'], 
                    panel_y + title_height - 15),
                   self.display_config['font'],
                   self.display_config['font_scale'],
                   self.display_config['colors']['title'],
                   self.display_config['font_thickness'])
        
        # Initialize starting position for feedback items
        y_offset = panel_y + title_height + self.display_config['margin']
        x_offset = panel_x + self.display_config['margin']
        
        try:
            # Get feedback items
            _, feedback_items = self.process_frame(frame, None, metrics)
            
            # Group feedback by type
            grouped_feedback = {
                'success': [],
                'warning': [],
                'error': [],
                'info': []
            }
            
            for item in feedback_items:
                grouped_feedback[item['type']].append(item['text'])
            
            # Draw feedback items with text wrapping
            max_width = panel_width - (2 * self.display_config['margin'])
            
            for feedback_type, messages in grouped_feedback.items():
                if messages:
                    color = self.display_config['colors'][feedback_type]
                    for message in messages:
                        # Check if message needs wrapping
                        text_size = cv2.getTextSize(message, 
                                                  self.display_config['font'],
                                                  self.display_config['font_scale'] - 0.1,
                                                  self.display_config['font_thickness'])[0]
                        
                        if text_size[0] > max_width:
                            # Split message into multiple lines
                            words = message.split()
                            lines = []
                            current_line = []
                            current_width = 0
                            
                            for word in words:
                                word_size = cv2.getTextSize(word + " ",
                                                          self.display_config['font'],
                                                          self.display_config['font_scale'] - 0.1,
                                                          self.display_config['font_thickness'])[0]
                                if current_width + word_size[0] <= max_width:
                                    current_line.append(word)
                                    current_width += word_size[0]
                                else:
                                    lines.append(" ".join(current_line))
                                    current_line = [word]
                                    current_width = word_size[0]
                                    
                            if current_line:
                                lines.append(" ".join(current_line))
                                
                            # Draw wrapped text
                            for line in lines:
                                if y_offset + self.display_config['line_spacing'] <= panel_y + panel_height:
                                    cv2.putText(overlay, line,
                                              (x_offset, y_offset),
                                              self.display_config['font'],
                                              self.display_config['font_scale'] - 0.1,
                                              color,
                                              self.display_config['font_thickness'])
                                    y_offset += self.display_config['line_spacing']
                        else:
                            # Draw single line
                            if y_offset + self.display_config['line_spacing'] <= panel_y + panel_height:
                                cv2.putText(overlay, message,
                                          (x_offset, y_offset),
                                          self.display_config['font'],
                                          self.display_config['font_scale'] - 0.1,
                                          color,
                                          self.display_config['font_thickness'])
                                y_offset += self.display_config['line_spacing']
                    
                    # Add spacing between groups
                    y_offset += self.display_config['line_spacing'] // 2
                    
        except Exception as e:
            print(f"Error drawing feedback: {str(e)}")
        
        # Blend overlay with original frame
        result = cv2.addWeighted(frame, 1 - self.display_config['panel_opacity'],
                               overlay, self.display_config['panel_opacity'], 0)
        
        return result