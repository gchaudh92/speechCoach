import cv2
import numpy as np
from collections import deque
import mediapipe as mp
import time 
from typing import Dict, List, Optional
from .utils import FeatureModule, AnalyzerConfig

class EmotionalIntelligenceAnalyzer(FeatureModule):
    def __init__(self, config: AnalyzerConfig):
        super().__init__(config)
        self.face_emotion = FacialEmotionDetector(config)
        self.voice_emotion = VoiceEmotionAnalyzer(config)
        self.emotion_history = deque(maxlen=30)
        
    def process(self, frame, audio_data, metrics: dict) -> dict:
        # Analyze facial emotions
        face_emotions = self.face_emotion.process(frame, audio_data, metrics)
        
        # Analyze voice emotions
        voice_emotions = self.voice_emotion.process(frame, audio_data, metrics)
        
        # Combine analyses
        combined_emotions = self._combine_emotions(face_emotions, voice_emotions)
        
        # Update history
        self.emotion_history.append(combined_emotions)
        
        return {
            **combined_emotions,
            'emotional_congruence': self._calculate_congruence(face_emotions, voice_emotions),
            'emotional_stability': self._calculate_stability(),
            'engagement_score': self._calculate_engagement(combined_emotions)
        }
    
    def _combine_emotions(self, face_emotions: dict, voice_emotions: dict) -> dict:
        """Combine facial and vocal emotional analyses"""
        if not face_emotions and not voice_emotions:
            return {'dominant_emotion': 'neutral', 'emotion_confidence': 0.0}
            
        # Weight the emotions (can be adjusted based on confidence)
        face_weight = 0.6
        voice_weight = 0.4
        
        combined = {}
        
        # Get dominant emotions and confidences
        face_emotion = face_emotions.get('dominant_emotion', 'neutral')
        face_conf = face_emotions.get('emotion_confidence', 0.0)
        
        voice_emotion = voice_emotions.get('dominant_emotion', 'neutral')
        voice_conf = voice_emotions.get('emotion_confidence', 0.0)
        
        # If emotions match, use the higher confidence
        if face_emotion == voice_emotion:
            return {
                'dominant_emotion': face_emotion,
                'emotion_confidence': max(face_conf, voice_conf)
            }
        
        # If emotions differ, use weighted combination
        weighted_face = face_conf * face_weight
        weighted_voice = voice_conf * voice_weight
        
        if weighted_face >= weighted_voice:
            return {
                'dominant_emotion': face_emotion,
                'emotion_confidence': weighted_face
            }
        else:
            return {
                'dominant_emotion': voice_emotion,
                'emotion_confidence': weighted_voice
            }
    
    def _calculate_congruence(self, face_emotions: dict, voice_emotions: dict) -> float:
        """Calculate emotional congruence between face and voice"""
        if not face_emotions or not voice_emotions:
            return 1.0
            
        face_emotion = face_emotions.get('dominant_emotion', 'neutral')
        voice_emotion = voice_emotions.get('dominant_emotion', 'neutral')
        
        if face_emotion == voice_emotion:
            return 1.0
            
        # Define emotion similarity matrix
        emotion_similarity = {
            'happy': {'excited': 0.8, 'neutral': 0.5},
            'sad': {'neutral': 0.5, 'anxious': 0.6},
            'angry': {'frustrated': 0.8, 'anxious': 0.6},
            'neutral': {'calm': 0.8, 'focused': 0.7}
        }
        
        # Get similarity score
        similarity = emotion_similarity.get(face_emotion, {}).get(voice_emotion, 0.0)
        return similarity
    
    def _calculate_stability(self) -> float:
        """Calculate emotional stability over time"""
        if not self.emotion_history:
            return 1.0
            
        # Count emotion changes
        changes = 0
        prev_emotion = None
        
        for emotion_data in self.emotion_history:
            current_emotion = emotion_data.get('dominant_emotion')
            if prev_emotion and current_emotion != prev_emotion:
                changes += 1
            prev_emotion = current_emotion
            
        stability = 1.0 - (changes / len(self.emotion_history))
        return max(0.0, min(1.0, stability))
    
    def _calculate_engagement(self, emotions: dict) -> float:
        """Calculate engagement score based on emotional state"""
        engagement_weights = {
            'happy': 1.0,
            'excited': 1.0,
            'focused': 0.9,
            'neutral': 0.7,
            'calm': 0.6,
            'anxious': 0.4,
            'sad': 0.3,
            'angry': 0.2
        }
        
        emotion = emotions.get('dominant_emotion', 'neutral')
        confidence = emotions.get('emotion_confidence', 0.5)
        
        base_score = engagement_weights.get(emotion, 0.5)
        return base_score * confidence

class FacialEmotionDetector(FeatureModule):
    def __init__(self, config: AnalyzerConfig):
        super().__init__(config)
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
    def process(self, frame, audio_data, metrics: dict) -> dict:
        if frame is None:
            return {}
            
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        if not results.multi_face_landmarks:
            return {'dominant_emotion': 'neutral', 'emotion_confidence': 0.0}
            
        landmarks = results.multi_face_landmarks[0]
        return self._analyze_facial_expression(landmarks)
    
    def _analyze_facial_expression(self, landmarks) -> dict:
        """Analyze facial expressions with improved emotion detection"""
        try:
            # Extract key facial features
            eyebrows = self._get_eyebrow_position(landmarks)
            mouth = self._get_mouth_shape(landmarks)
            eyes = self._get_eye_state(landmarks)
            
            # Initialize emotion scores
            emotions = {
                'happy': 0.0,
                'sad': 0.0,
                'neutral': 0.5,  # Default state
                'surprised': 0.0,
                'concerned': 0.0
            }
            
            # Update emotion scores based on features
            # Happy indicators
            if mouth['state'] == 'smile':
                emotions['happy'] += 0.6
            if eyebrows['average_height'] > 0:
                emotions['happy'] += 0.2
                
            # Sad indicators
            if mouth['state'] == 'frown':
                emotions['sad'] += 0.6
            if eyebrows['average_height'] < -0.1:
                emotions['sad'] += 0.2
                
            # Surprised indicators
            if eyebrows['average_height'] > 0.2:
                emotions['surprised'] += 0.5
            if mouth['state'] == 'open':
                emotions['surprised'] += 0.3
                
            # Concerned indicators
            if eyebrows['asymmetry'] > 0.1:
                emotions['concerned'] += 0.4
            if mouth['corner_angle'] < -0.1:
                emotions['concerned'] += 0.4
                
            # Find dominant emotion
            dominant_emotion = max(emotions.items(), key=lambda x: x[1])
            confidence = dominant_emotion[1]
            
            # If no strong emotion detected, default to neutral
            if confidence < 0.4:
                dominant_emotion = ('neutral', 0.5)
            
            return {
                'dominant_emotion': dominant_emotion[0],
                'emotion_confidence': dominant_emotion[1],
                'emotion_scores': emotions
            }
            
        except Exception as e:
            print(f"Error analyzing facial expression: {str(e)}")
            return {
                'dominant_emotion': 'neutral',
                'emotion_confidence': 0.5,
                'emotion_scores': {'neutral': 1.0}
            }
        

    def _get_mouth_shape(self, landmarks) -> dict:
        """Analyze mouth shape from facial landmarks
        
        Args:
            landmarks: Face landmarks from MediaPipe
            
        Returns:
            dict: Mouth shape metrics
        """
        # MediaPipe indices for mouth landmarks
        MOUTH_TOP = 13      # Upper lip
        MOUTH_BOTTOM = 14   # Lower lip
        MOUTH_LEFT = 61     # Left corner
        MOUTH_RIGHT = 291   # Right corner
        
        try:
            # Get mouth points
            top = landmarks.landmark[MOUTH_TOP]
            bottom = landmarks.landmark[MOUTH_BOTTOM]
            left = landmarks.landmark[MOUTH_LEFT]
            right = landmarks.landmark[MOUTH_RIGHT]
            
            # Calculate mouth metrics
            height = abs(top.y - bottom.y)
            width = abs(left.x - right.x)
            
            # Calculate mouth aspect ratio
            aspect_ratio = height / max(width, 0.001)
            
            # Calculate mouth corners angle (for smile detection)
            mouth_angle = (right.y - left.y) / max(abs(right.x - left.x), 0.001)
            
            # Determine mouth state
            if aspect_ratio > 0.5:
                state = "open"
            elif mouth_angle < -0.1:
                state = "smile"
            elif mouth_angle > 0.1:
                state = "frown"
            else:
                state = "neutral"
                
            return {
                'aspect_ratio': aspect_ratio,
                'opening': height,
                'width': width,
                'corner_angle': mouth_angle,
                'state': state
            }
            
        except Exception as e:
            print(f"Error analyzing mouth shape: {str(e)}")
            return {
                'aspect_ratio': 0.0,
                'opening': 0.0,
                'width': 0.0,
                'corner_angle': 0.0,
                'state': 'unknown'
            }
        
    def _get_eyebrow_position(self, landmarks) -> dict:
        """Get eyebrow positions relative to neutral position
        
        Args:
            landmarks: Face landmarks from MediaPipe
            
        Returns:
            dict: Eyebrow positions and metrics
        """
        # MediaPipe indices for eyebrows
        LEFT_EYEBROW = [70, 63, 105, 66, 107]  # Central points of left eyebrow
        RIGHT_EYEBROW = [336, 296, 334, 293, 300]  # Central points of right eyebrow
        LEFT_EYE = [33, 133]  # Points for left eye reference
        RIGHT_EYE = [362, 263]  # Points for right eye reference
        
        try:
            # Get average eyebrow heights
            left_brow_y = np.mean([landmarks.landmark[i].y for i in LEFT_EYEBROW])
            right_brow_y = np.mean([landmarks.landmark[i].y for i in RIGHT_EYEBROW])
            
            # Get eye reference points for relative position
            left_eye_y = np.mean([landmarks.landmark[i].y for i in LEFT_EYE])
            right_eye_y = np.mean([landmarks.landmark[i].y for i in RIGHT_EYE])
            
            # Calculate relative positions (distance from eyes)
            left_position = left_eye_y - left_brow_y
            right_position = right_eye_y - right_brow_y
            
            # Calculate asymmetry
            asymmetry = abs(left_position - right_position)
            
            return {
                'left_height': left_position,
                'right_height': right_position,
                'asymmetry': asymmetry,
                'average_height': (left_position + right_position) / 2
            }
            
        except Exception as e:
            print(f"Error getting eyebrow position: {str(e)}")
            return {
                'left_height': 0.1,  # Default values
                'right_height': 0.1,
                'asymmetry': 0,
                'average_height': 0.1
            }
        
    def _get_eye_state(self, landmarks) -> dict:
        """Analyze eye state from facial landmarks
        
        Args:
            landmarks: Face landmarks from MediaPipe
            
        Returns:
            dict: Eye state information
        """
        # MediaPipe indices for eyes
        LEFT_EYE_TOP = 159
        LEFT_EYE_BOTTOM = 145
        LEFT_EYE_LEFT = 33
        LEFT_EYE_RIGHT = 133
        
        RIGHT_EYE_TOP = 386
        RIGHT_EYE_BOTTOM = 374
        RIGHT_EYE_LEFT = 362
        RIGHT_EYE_RIGHT = 263
        
        try:
            # Calculate eye aspect ratios
            def get_eye_ratio(top, bottom, left, right):
                eye_height = abs(landmarks.landmark[top].y - landmarks.landmark[bottom].y)
                eye_width = abs(landmarks.landmark[left].x - landmarks.landmark[right].x)
                return eye_height / max(eye_width, 0.001)
            
            left_ratio = get_eye_ratio(LEFT_EYE_TOP, LEFT_EYE_BOTTOM, 
                                    LEFT_EYE_LEFT, LEFT_EYE_RIGHT)
            right_ratio = get_eye_ratio(RIGHT_EYE_TOP, RIGHT_EYE_BOTTOM, 
                                    RIGHT_EYE_LEFT, RIGHT_EYE_RIGHT)
            
            # Determine eye states
            def get_state(ratio):
                if ratio < 0.15:
                    return "closed"
                elif ratio < 0.25:
                    return "squinting"
                else:
                    return "open"
                    
            left_state = get_state(left_ratio)
            right_state = get_state(right_ratio)
            
            # Calculate average openness
            avg_ratio = (left_ratio + right_ratio) / 2
            
            return {
                'left_eye': {
                    'ratio': left_ratio,
                    'state': left_state
                },
                'right_eye': {
                    'ratio': right_ratio,
                    'state': right_state
                },
                'average_ratio': avg_ratio,
                'blink_detected': avg_ratio < 0.15,
                'overall_state': 'closed' if avg_ratio < 0.15 else 'open'
            }
            
        except Exception as e:
            print(f"Error analyzing eye state: {str(e)}")
            return {
                'left_eye': {'ratio': 0.3, 'state': 'unknown'},
                'right_eye': {'ratio': 0.3, 'state': 'unknown'},
                'average_ratio': 0.3,
                'blink_detected': False,
                'overall_state': 'unknown'
            }

    def _check_happy(self, eyebrow_pos: dict, mouth_shape: dict) -> float:
        """Check for happy expression
        
        Args:
            eyebrow_pos: Eyebrow position metrics
            mouth_shape: Mouth shape metrics
            
        Returns:
            float: Confidence score for happy expression (0-1)
        """
        score = 0.0
        try:
            # Check for smile
            if mouth_shape['state'] == 'smile':
                score += 0.6
                
            # Slightly raised eyebrows often accompany genuine smiles
            if eyebrow_pos['average_height'] > 0.1:
                score += 0.2
                
            # Symmetric expression is more likely genuine
            if eyebrow_pos['asymmetry'] < 0.1:
                score += 0.2
                
            return min(1.0, score)
        except:
            return 0.0

    def _check_sad(self, eyebrow_pos: dict, mouth_shape: dict) -> float:
        """Check for sad expression"""
        score = 0.0
        try:
            # Downturned mouth
            if mouth_shape['state'] == 'frown':
                score += 0.6
                
            # Inner eyebrows often raised in sadness
            if eyebrow_pos['average_height'] > 0.15:
                score += 0.4
                
            return min(1.0, score)
        except:
            return 0.0

    def _check_angry(self, eyebrow_pos: dict, eye_state: dict) -> float:
        """Check for angry expression"""
        score = 0.0
        try:
            # Lowered/furrowed brows
            if eyebrow_pos['average_height'] < -0.1:
                score += 0.5
                
            # Squinting eyes often accompany anger
            if eye_state['overall_state'] == 'squinting':
                score += 0.3
                
            # Asymmetric brows can indicate anger
            if eyebrow_pos['asymmetry'] > 0.1:
                score += 0.2
                
            return min(1.0, score)
        except:
            return 0.0

    def _check_surprised(self, eyebrow_pos: dict, mouth_shape: dict) -> float:
        """Check for surprised expression"""
        score = 0.0
        try:
            # Raised eyebrows
            if eyebrow_pos['average_height'] > 0.2:
                score += 0.4
                
            # Open mouth
            if mouth_shape['state'] == 'open':
                score += 0.4
                
            # Symmetric expression
            if eyebrow_pos['asymmetry'] < 0.1:
                score += 0.2
                
            return min(1.0, score)
        except:
            return 0.0

    def _check_neutral(self, eyebrow_pos: dict, mouth_shape: dict) -> float:
        """Check for neutral expression"""
        score = 0.5  # Start at middle
        try:
            # Neutral mouth
            if mouth_shape['state'] == 'neutral':
                score += 0.2
                
            # Normal eyebrow position
            if -0.05 < eyebrow_pos['average_height'] < 0.05:
                score += 0.2
                
            # Symmetric expression
            if eyebrow_pos['asymmetry'] < 0.05:
                score += 0.1
                
            return min(1.0, score)
        except:
            return 0.5  

class VoiceEmotionAnalyzer(FeatureModule):
    def __init__(self, config: AnalyzerConfig):
        super().__init__(config)
        self.voice_history = deque(maxlen=30)
        
    def process(self, frame, audio_data, metrics: dict) -> dict:
        if audio_data is None:
            return {}
            
        # Extract voice features
        pitch = self._extract_pitch(audio_data)
        energy = self._calculate_energy(audio_data)
        rhythm = self._analyze_rhythm(audio_data)
        
        # Analyze emotional content
        emotions = self._classify_emotions(pitch, energy, rhythm)
        
        # Update history
        self.voice_history.append(emotions)
        
        return {
            'dominant_emotion': max(emotions.items(), key=lambda x: x[1])[0],
            'emotion_confidence': max(emotions.values()),
            'emotion_scores': emotions
        }
    
    def _extract_pitch(self, audio_data: np.ndarray) -> float:
        """Extract pitch from audio data"""
        # Implement pitch extraction
        return 0.0
    
    def _calculate_energy(self, audio_data: np.ndarray) -> float:
        """Calculate energy of audio signal"""
        return np.mean(np.abs(audio_data))
    
    def _analyze_rhythm(self, audio_data: np.ndarray) -> float:
        """Analyze speech rhythm"""
        # Implement rhythm analysis
        return 0.0
    
    def _classify_emotions(self, pitch: float, energy: float, rhythm: float) -> dict:
        """Classify emotions based on voice features"""
        emotions = {
            'happy': 0.0,
            'sad': 0.0,
            'angry': 0.0,
            'neutral': 0.5
        }
        
        # Implement emotion classification based on voice features
        return emotions

class EmotionalFeedbackManager:
    def __init__(self, config: AnalyzerConfig):
        self.analyzer = EmotionalIntelligenceAnalyzer(config)
        self.feedback_history = deque(maxlen=50)
        
    def process_frame(self, frame, audio_data, metrics: dict) -> tuple:
        """Process frame and generate emotional feedback"""
        # Analyze emotions
        emotion_metrics = self.analyzer.process(frame, audio_data, metrics)
        
        # Generate feedback
        feedback = self._generate_feedback(emotion_metrics)
        
        # Update history
        self.feedback_history.append((emotion_metrics, feedback))
        
        return emotion_metrics, feedback
    
    def _generate_feedback(self, metrics: dict) -> List[str]:
        """Generate emotional feedback messages"""
        feedback = []
        
        # Check emotional congruence
        if metrics.get('emotional_congruence', 1.0) < 0.5:
            feedback.append("Your facial expression and tone don't match")
        
        # Check emotional stability
        if metrics.get('emotional_stability', 1.0) < 0.6:
            feedback.append("Try to maintain more consistent emotional expression")
        
        # Check engagement
        if metrics.get('engagement_score', 1.0) < 0.5:
            feedback.append("Try to show more engagement with your presentation")
        
        return feedback