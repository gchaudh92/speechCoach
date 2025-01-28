import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import time 
from typing import Dict, List, Optional, Tuple
from .utils import FeatureModule, AnalyzerConfig

class GestureAnalyzer(FeatureModule):
    def __init__(self, config: AnalyzerConfig):
        super().__init__(config)
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7
        )
        self.gesture_history = deque(maxlen=30)
        self.movement_threshold = 0.1
        
    def process(self, frame, audio_data, metrics: dict) -> dict:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        gesture_metrics = {
            'gesture_detected': False,
            'gesture_type': 'none',
            'movement_score': 0.0,
            'hand_position': 'neutral'
        }
        
        if results.multi_hand_landmarks:
            gesture_metrics.update(self._analyze_gestures(results.multi_hand_landmarks))
            
        return gesture_metrics
    
    def _analyze_gestures(self, hand_landmarks: list) -> dict:
        """Analyze hand gestures and movements"""
        movement = self._calculate_movement(hand_landmarks)
        gesture_type = self._classify_gesture(hand_landmarks)
        position = self._get_hand_position(hand_landmarks)
        
        self.gesture_history.append((gesture_type, movement))
        
        return {
            'gesture_type': gesture_type,
            'movement_score': movement,
            'hand_position': position,
            'gesture_feedback': self._generate_feedback(gesture_type, movement)
        }

class EyeContactTracker(FeatureModule):
    def __init__(self, config: AnalyzerConfig):
        super().__init__(config)
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.eye_history = deque(maxlen=30)
        self.contact_threshold = 0.7
        
    def process(self, frame, audio_data, metrics: dict) -> dict:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        eye_metrics = {
            'eye_contact': 0.0,
            'gaze_direction': 'center',
            'attention_score': 0.0
        }
        
        if results.multi_face_landmarks:
            eye_metrics.update(self._analyze_eye_contact(results.multi_face_landmarks[0]))
            
        return eye_metrics
    
    def _analyze_eye_contact(self, landmarks) -> dict:
        """Analyze eye contact with improved metrics"""
        # Define key points for eye detection
        LEFT_EYE = [33, 133]  # Outer, inner corners
        RIGHT_EYE = [362, 263]
        NOSE_TIP = 4

        try:
            # Get eye points
            left_eye = [landmarks.landmark[i] for i in LEFT_EYE]
            right_eye = [landmarks.landmark[i] for i in RIGHT_EYE]
            nose = landmarks.landmark[NOSE_TIP]

            # Calculate gaze direction based on eye and nose positions
            gaze_x = (left_eye[0].x + right_eye[0].x) / 2 - nose.x
            gaze_y = (left_eye[0].y + right_eye[0].y) / 2 - nose.y

            # Calculate eye contact score
            # Higher score when looking straight ahead (gaze near 0,0)
            gaze_distance = np.sqrt(gaze_x**2 + gaze_y**2)
            eye_contact_score = max(0, 1 - (gaze_distance * 5))  # Scale for sensitivity

            # Determine gaze direction
            if abs(gaze_x) > abs(gaze_y):
                direction = "left" if gaze_x < 0 else "right"
            else:
                direction = "up" if gaze_y < 0 else "down"

            if gaze_distance < 0.1:
                direction = "center"

            # Update history for smoothing
            self.eye_history.append(eye_contact_score)
            smoothed_score = np.mean(list(self.eye_history))

            return {
                'eye_contact_score': smoothed_score,
                'gaze_direction': direction,
                'attention_score': smoothed_score,
                'raw_gaze': {'x': gaze_x, 'y': gaze_y}
            }

        except Exception as e:
            print(f"Error analyzing eye contact: {str(e)}")
            return {
                'eye_contact_score': 0.5,
                'gaze_direction': 'unknown',
                'attention_score': 0.5,
                'raw_gaze': {'x': 0, 'y': 0}
            }
        
    def _calculate_gaze_direction(self, left_eye, right_eye) -> dict:
        """Calculate gaze direction based on eye landmarks
        
        Args:
            left_eye: Left eye landmarks
            right_eye: Right eye landmarks
            
        Returns:
            dict: Gaze metrics including direction and contact score
        """
        if not left_eye or not right_eye:
            return {'direction': 'unknown', 'contact_score': 0.0}
            
        # Calculate average eye position
        eye_center_x = (left_eye[0].x + right_eye[0].x) / 2
        eye_center_y = (left_eye[0].y + right_eye[0].y) / 2
        
        # Determine gaze direction based on eye center position
        gaze_x = 'center'
        gaze_y = 'center'
        contact_score = 1.0
        
        if eye_center_x < 0.4:
            gaze_x = 'left'
            contact_score *= 0.5
        elif eye_center_x > 0.6:
            gaze_x = 'right'
            contact_score *= 0.5
            
        if eye_center_y < 0.4:
            gaze_y = 'up'
            contact_score *= 0.5
        elif eye_center_y > 0.6:
            gaze_y = 'down'
            contact_score *= 0.5
        
        return {
            'direction': f"{gaze_y}-{gaze_x}",
            'contact_score': contact_score
        }

    def _get_eye_landmarks(self, landmarks, side: str) -> list:
        """Get landmarks for specified eye
        
        Args:
            landmarks: Face landmarks from MediaPipe
            side: 'left' or 'right' eye
            
        Returns:
            list: Eye landmark points
        """
        # MediaPipe face mesh eye indices
        LEFT_EYE_INDICES = [33, 133]  # Simplified indices for example
        RIGHT_EYE_INDICES = [362, 263]
        
        indices = LEFT_EYE_INDICES if side == 'left' else RIGHT_EYE_INDICES
        return [landmarks.landmark[idx] for idx in indices]

    def _calculate_eye_contact_score(self, gaze_direction: str) -> float:
        """Calculate eye contact score based on gaze direction
        
        Args:
            gaze_direction: String indicating gaze direction
            
        Returns:
            float: Eye contact score between 0 and 1
        """
        # Perfect eye contact when looking center
        if gaze_direction == 'center-center':
            return 1.0
            
        # Reduced scores for slight deviations
        if '-center' in gaze_direction:
            return 0.7
        if 'center-' in gaze_direction:
            return 0.7
            
        # Low scores for looking away
        return 0.3

class MovementAnalyzer(FeatureModule):
    def __init__(self, config: AnalyzerConfig):
        super().__init__(config)
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            min_detection_confidence=0.5
        )
        self.movement_history = deque(maxlen=30)
        self.position_baseline = None
        
    def process(self, frame, audio_data, metrics: dict) -> dict:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)
        
        movement_metrics = {
            'movement_level': 0.0,
            'posture_quality': 'unknown',
            'stability_score': 0.0
        }
        
        if results.pose_landmarks:
            movement_metrics.update(self._analyze_movement(results.pose_landmarks))
            
        return movement_metrics
    
    def _analyze_movement(self, pose_landmarks) -> dict:
        """Analyze body movement and posture"""
        current_position = self._get_key_points(pose_landmarks)
        
        if self.position_baseline is None:
            self.position_baseline = current_position
            return {'movement_level': 0.0, 'posture_quality': 'calibrating'}
            
        movement = self._calculate_movement(current_position)
        posture = self._analyze_posture(current_position)
        stability = self._calculate_stability(movement)
        
        self.movement_history.append(movement)
        
        return {
            'movement_level': movement,
            'posture_quality': posture,
            'stability_score': stability,
            'movement_feedback': self._generate_feedback(movement, posture)
        }

    def _get_key_points(self, pose_landmarks) -> dict:
        """Extract key body points for movement analysis
        
        Args:
            pose_landmarks: Pose landmarks from MediaPipe
            
        Returns:
            dict: Dictionary of key body points
        """
        # MediaPipe pose landmark indices
        KEYPOINTS = {
            'nose': 0,
            'left_shoulder': 11,
            'right_shoulder': 12,
            'left_hip': 23,
            'right_hip': 24,
            'left_wrist': 15,
            'right_wrist': 16
        }
        
        points = {}
        try:
            for name, idx in KEYPOINTS.items():
                landmark = pose_landmarks.landmark[idx]
                points[name] = {
                    'x': landmark.x,
                    'y': landmark.y,
                    'z': landmark.z,
                    'visibility': landmark.visibility
                }
            return points
        except:
            return {}
        
    def _calculate_movement(self, current_position: dict) -> float:
        """Calculate movement level from current position
        
        Args:
            current_position: Dictionary of current key points
            
        Returns:
            float: Movement score between 0 and 1
        """
        if not current_position:
            return 0.0
            
        if self.position_baseline is None:
            self.position_baseline = current_position
            return 0.0
        
        movement_score = 0.0
        valid_points = 0
        
        # Weight factors for different body parts
        weights = {
            'nose': 1.0,
            'left_shoulder': 0.8,
            'right_shoulder': 0.8,
            'left_hip': 0.6,
            'right_hip': 0.6,
            'left_wrist': 0.4,
            'right_wrist': 0.4
        }
        
        try:
            # Compare each point that exists in both current and baseline
            for point_name, weight in weights.items():
                if point_name in current_position and point_name in self.position_baseline:
                    current = current_position[point_name]
                    baseline = self.position_baseline[point_name]
                    
                    # Only consider points with good visibility
                    if (current.get('visibility', 0) > 0.5 and 
                        baseline.get('visibility', 0) > 0.5):
                        
                        # Calculate weighted Euclidean distance
                        dist = np.sqrt(
                            (current['x'] - baseline['x'])**2 +
                            (current['y'] - baseline['y'])**2
                        ) * weight
                        
                        movement_score += dist
                        valid_points += 1
            
            # Normalize movement score
            if valid_points > 0:
                movement_score /= valid_points
                
            # Update baseline with slight smoothing
            alpha = 0.3
            for point_name in current_position:
                if point_name in self.position_baseline:
                    current = current_position[point_name]
                    baseline = self.position_baseline[point_name]
                    
                    for coord in ['x', 'y']:
                        if coord in current and coord in baseline:
                            baseline[coord] = (alpha * current[coord] + 
                                            (1 - alpha) * baseline[coord])
            
            return min(1.0, movement_score * 5)  # Scale up for better sensitivity
            
        except Exception as e:
            print(f"Error calculating movement: {str(e)}")
            return 0.0
    
    def _analyze_posture(self, current_position: dict) -> str:
        """Analyze posture based on key points
        
        Args:
            current_position: Dictionary of current key points
            
        Returns:
            str: Posture quality description
        """
        if not current_position:
            return "unknown"
            
        try:
            # Get key points for posture analysis
            nose = current_position.get('nose', {})
            left_shoulder = current_position.get('left_shoulder', {})
            right_shoulder = current_position.get('right_shoulder', {})
            left_hip = current_position.get('left_hip', {})
            right_hip = current_position.get('right_hip', {})
            
            # Check if we have enough points for analysis
            if not all(p.get('visibility', 0) > 0.5 for p in [left_shoulder, right_shoulder]):
                return "unknown"
            
            # Calculate shoulder tilt
            shoulder_tilt = abs(left_shoulder['y'] - right_shoulder['y'])
            
            # Calculate lean (using nose if visible, otherwise shoulder midpoint)
            if nose.get('visibility', 0) > 0.5:
                head_x = nose['x']
            else:
                head_x = (left_shoulder['x'] + right_shoulder['x']) / 2
                
            shoulder_x = (left_shoulder['x'] + right_shoulder['x']) / 2
            lean = abs(head_x - shoulder_x)
            
            # Define thresholds
            TILT_THRESHOLD = 0.05
            LEAN_THRESHOLD = 0.05
            
            if shoulder_tilt > TILT_THRESHOLD:
                return "shoulders not level"
            elif lean > LEAN_THRESHOLD:
                return "leaning"
            else:
                return "good"
                
        except Exception as e:
            print(f"Error analyzing posture: {str(e)}")
            return "unknown"

    def _calculate_stability(self, movement: float) -> float:
        """Calculate stability score based on movement history
        
        Args:
            movement: Current movement score
            
        Returns:
            float: Stability score between 0 and 1
        """
        # Add current movement to history
        if not hasattr(self, 'movement_history'):
            self.movement_history = deque(maxlen=30)
        self.movement_history.append(movement)
        
        if len(self.movement_history) < 2:
            return 1.0  # Default to stable when not enough history
            
        try:
            # Calculate stability based on movement variation
            movement_std = np.std(list(self.movement_history))
            movement_mean = np.mean(list(self.movement_history))
            
            # Normalize stability score
            stability = 1.0 - min(1.0, (movement_std / max(movement_mean, 0.001)))
            
            # Apply smoothing
            if not hasattr(self, 'last_stability'):
                self.last_stability = stability
            
            alpha = 0.3  # Smoothing factor
            smoothed_stability = alpha * stability + (1 - alpha) * self.last_stability
            self.last_stability = smoothed_stability
            
            return max(0.0, min(1.0, smoothed_stability))
            
        except Exception as e:
            print(f"Error calculating stability: {str(e)}")
            return 1.0  # Default to stable if calculation fails
        
    def _generate_feedback(self, movement: float, posture: str) -> List[str]:
        """Generate feedback based on movement and posture analysis
        
        Args:
            movement: Current movement score
            posture: Current posture assessment
            
        Returns:
            List[str]: List of feedback messages
        """
        feedback = []
        
        # Movement feedback
        if movement > 0.7:
            feedback.append("Try to minimize excessive movement")
        elif movement > 0.4:
            feedback.append("Reduce movement slightly")
        elif movement < 0.1:
            feedback.append("You can be a bit more dynamic")
            
        # Posture feedback
        if posture != "good":
            if posture == "shoulders not level":
                feedback.append("Level your shoulders")
            elif posture == "leaning":
                feedback.append("Stand up straight")
            elif posture != "unknown":
                feedback.append(f"Correct your posture: {posture}")
                
        # Add positive feedback if everything is good
        if not feedback:
            feedback.append("Good posture and movement")
            
        return feedback

class BehavioralFeedbackManager:
    def __init__(self, config: AnalyzerConfig):
        self.gesture_analyzer = GestureAnalyzer(config)
        self.eye_tracker = EyeContactTracker(config)
        self.movement_analyzer = MovementAnalyzer(config)
        
    def process_frame(self, frame, audio_data, metrics: dict) -> tuple:
        """Process frame through all behavioral analyzers"""
        # Analyze gestures
        gesture_metrics = self.gesture_analyzer.process(frame, audio_data, metrics)
        
        # Analyze eye contact
        eye_metrics = self.eye_tracker.process(frame, audio_data, metrics)
        
        # Analyze movement
        movement_metrics = self.movement_analyzer.process(frame, audio_data, metrics)
        
        # Combine metrics
        combined_metrics = {
            **gesture_metrics,
            **eye_metrics,
            **movement_metrics
        }
        
        # Generate feedback
        feedback = self._generate_feedback(combined_metrics)
        
        return combined_metrics, feedback
    
    def _generate_feedback(self, metrics: dict) -> List[str]:
        """Generate behavioral feedback messages"""
        feedback = []
        
        # Gesture feedback
        if metrics.get('gesture_feedback'):
            feedback.append(metrics['gesture_feedback'])
        
        # Eye contact feedback
        if metrics.get('eye_contact', 0) < 0.5:
            feedback.append("Try to maintain more eye contact")
        
        # Movement feedback
        if metrics.get('movement_level', 0) > 0.3:
            feedback.append("Try to minimize excessive movement")
        
        return feedback