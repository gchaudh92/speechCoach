import cv2
import mediapipe as mp
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import deque
import time
from .utils import FeatureModule, AnalyzerConfig

class VideoProcessor:
    """Handles low-level video processing operations"""
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Initialize detectors
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Define key facial landmarks
        self.NOSE_TIP = 4
        self.CHIN = 152
        self.LEFT_EYE = [33, 133]
        self.RIGHT_EYE = [362, 263]
        self.MOUTH_CORNERS = [61, 291]
        self.LEFT_EYEBROW = [70, 63, 105, 66, 107]
        self.RIGHT_EYEBROW = [336, 296, 334, 293, 300]

class VideoAnalyzer(FeatureModule):
    """Main video analysis class"""
    def __init__(self, config: AnalyzerConfig):
        super().__init__(config)
        self.processor = VideoProcessor()
        self.pose_history = deque(maxlen=30)
        self.expression_history = deque(maxlen=10)
        self.movement_baseline = None
        self.position_baseline = None 
        
        # Analysis thresholds
        self.thresholds = {
            'head_movement': 30.0,  # degrees
            'eye_contact': 0.7,     # ratio
            'posture': 0.15,        # ratio
            'movement': 0.2         # ratio
        }
        self.MOUTH_CORNERS = [61, 291]
        self.LEFT_EYEBROW = [70, 63, 105, 66, 107]
        self.RIGHT_EYEBROW = [336, 296, 334, 293, 300]
    
    def analyze_frame(self, frame: np.ndarray) -> Tuple[Dict, List[str], np.ndarray]:
        """Analyze a single frame and return metrics, feedback, and annotated frame"""
        if frame is None:
            return {}, [], None
            
        metrics = {}
        feedback = []
        annotated_frame = frame.copy()
        
        try:
            # Process with MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_results = self.processor.face_mesh.process(rgb_frame)
            pose_results = self.processor.pose.process(rgb_frame)
            
            # Analyze face if detected
            if face_results.multi_face_landmarks:
                face_metrics = self._analyze_face(face_results.multi_face_landmarks[0])
                metrics.update(face_metrics)
                
                # Generate face-related feedback
                face_feedback = self._generate_face_feedback(face_metrics)
                feedback.extend(face_feedback)
                
                # Draw facial annotations
                annotated_frame = self._draw_face_annotations(
                    annotated_frame, 
                    face_results.multi_face_landmarks[0],
                    face_metrics
                )
            
            # Analyze pose if detected
            if pose_results.pose_landmarks:
                pose_metrics = self._analyze_pose(pose_results.pose_landmarks)
                metrics.update(pose_metrics)
                
                # Generate pose-related feedback
                pose_feedback = self._generate_pose_feedback(pose_metrics)
                feedback.extend(pose_feedback)
                
                # Draw pose annotations
                annotated_frame = self._draw_pose_annotations(
                    annotated_frame,
                    pose_results.pose_landmarks,
                    pose_metrics
                )
            
        except Exception as e:
            print(f"Error in video analysis: {str(e)}")
            
        return metrics, feedback, annotated_frame
    
    def _calculate_eye_aspect_ratio(self, eye_landmarks):
        """Calculate eye aspect ratio to detect blinks/closure
        
        Args:
            eye_landmarks: List of eye landmark points
            
        Returns:
            float: Eye aspect ratio
        """
        if len(eye_landmarks) < 2:
            return 0.0
            
        # Calculate height of eye
        eye_height = abs(eye_landmarks[0].y - eye_landmarks[1].y)
        
        # Calculate width of eye
        eye_width = abs(eye_landmarks[0].x - eye_landmarks[1].x)
        
        # Calculate eye aspect ratio
        ear = eye_height / max(eye_width, 0.01)  # Avoid division by zero
        
        return ear

    def _analyze_face(self, landmarks) -> Dict:
        """Analyze facial features and expressions"""
        metrics = {}
        
        # Calculate head pose
        pose_angles = self._calculate_head_pose(landmarks)
        metrics['head_pose'] = {
            'pitch': pose_angles[0],
            'yaw': pose_angles[1],
            'roll': pose_angles[2]
        }
        
        # Analyze eye contact
        eye_metrics = self._analyze_eye_contact(landmarks)
        metrics.update(eye_metrics)
        
        # Analyze expression
        expression = self._analyze_expression(landmarks)
        metrics['expression'] = expression
        
        return metrics
    
    def _calculate_head_pose(self, landmarks) -> Tuple[float, float, float]:
        """Calculate head pose angles (pitch, yaw, roll)"""
        # Get key points
        nose = landmarks.landmark[self.processor.NOSE_TIP]
        left_eye = landmarks.landmark[self.processor.LEFT_EYE[0]]
        right_eye = landmarks.landmark[self.processor.RIGHT_EYE[0]]
        chin = landmarks.landmark[self.processor.CHIN]
        
        # Calculate angles
        pitch = np.arctan2(nose.y - chin.y, nose.z - chin.z)
        yaw = np.arctan2(nose.x - (left_eye.x + right_eye.x)/2,
                        nose.z - (left_eye.z + right_eye.z)/2)
        roll = np.arctan2(right_eye.y - left_eye.y,
                         right_eye.x - left_eye.x)
        
        return np.degrees([pitch, yaw, roll])
    
    def _analyze_eye_contact(self, landmarks) -> Dict:
        """Analyze eye contact and gaze direction"""
        left_eye = [landmarks.landmark[i] for i in self.processor.LEFT_EYE]
        right_eye = [landmarks.landmark[i] for i in self.processor.RIGHT_EYE]
        
        # Calculate eye aspect ratios
        left_ear = self._calculate_eye_aspect_ratio(left_eye)
        right_ear = self._calculate_eye_aspect_ratio(right_eye)
        
        # Determine gaze direction
        gaze_metrics = self._calculate_gaze_direction(left_eye, right_eye)
        
        return {
            'eye_contact_score': gaze_metrics['contact_score'],
            'gaze_direction': gaze_metrics['direction'],
            'blink_rate': (left_ear + right_ear) / 2
        }
    
    def _get_hip_points(self, landmarks) -> tuple:
        """Get hip landmark points with fallback defaults
        
        Args:
            landmarks: Pose landmarks from MediaPipe
            
        Returns:
            tuple: Left and right hip points
        """
        # MediaPipe pose landmark indices for hips
        LEFT_HIP = 23
        RIGHT_HIP = 24
        
        # Default values - estimate hips as lower than shoulders at fixed distance
        default_hip = lambda x, y: type('Point', (), {
            'x': x,
            'y': y + 0.25,  # 25% lower than shoulders
            'z': 0,
            'visibility': 0.5
        })
        
        try:
            # Get shoulder points as reference
            left_shoulder, right_shoulder = self._get_shoulder_points(landmarks)
            
            # Try to get actual hip points
            if landmarks.landmark[LEFT_HIP].visibility > 0.5:
                left_hip = landmarks.landmark[LEFT_HIP]
            else:
                left_hip = default_hip(left_shoulder.x, left_shoulder.y)
                
            if landmarks.landmark[RIGHT_HIP].visibility > 0.5:
                right_hip = landmarks.landmark[RIGHT_HIP]
            else:
                right_hip = default_hip(right_shoulder.x, right_shoulder.y)
                
            return left_hip, right_hip
        except:
            # If shoulders not found, use center-based defaults
            left_hip = default_hip(0.4, 0.6)
            right_hip = default_hip(0.6, 0.6)
            return left_hip, right_hip
        
    def _analyze_pose(self, landmarks) -> Dict:
        """Analyze body posture and movement"""
        # Get key pose points
        shoulders = self._get_shoulder_points(landmarks)
        hips = self._get_hip_points(landmarks)
        
        # Calculate posture metrics
        posture_quality = self._analyze_posture_quality(shoulders, hips)
        movement = self._calculate_movement(shoulders)
        
        return {
            'posture': posture_quality,
            'movement_level': movement,
            'shoulder_alignment': self._calculate_shoulder_alignment(shoulders),
            'stability_score': self._calculate_stability(shoulders, hips)
        }
    
    def _draw_face_annotations(self, frame: np.ndarray, landmarks, metrics: Dict) -> np.ndarray:
        """Draw facial feature annotations on frame"""
        # Draw facial landmarks
        self.processor.mp_drawing.draw_landmarks(
            image=frame,
            landmark_list=landmarks,
            connections=self.processor.mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=self.processor.mp_drawing.DrawingSpec(
                color=(0, 255, 0), thickness=1, circle_radius=1),
            connection_drawing_spec=self.processor.mp_drawing.DrawingSpec(
                color=(0, 255, 0), thickness=1)
        )
        
        # Add metrics display
        h, w = frame.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        y_offset = 30
        
        # Display head pose angles
        for angle_name, value in metrics['head_pose'].items():
            text = f"{angle_name}: {value:.1f}°"
            cv2.putText(frame, text, (10, y_offset), font, 0.7,
                       (255, 255, 255), 2)
            y_offset += 30
        
        # Display expression
        if 'expression' in metrics:
            cv2.putText(frame, f"Expression: {metrics['expression']}",
                       (10, y_offset), font, 0.7, (255, 255, 255), 2)
        
        return frame
    
    def _draw_pose_annotations(self, frame: np.ndarray, landmarks, metrics: Dict) -> np.ndarray:
        """Draw pose annotations on frame"""
        # Draw pose landmarks
        self.processor.mp_drawing.draw_landmarks(
            image=frame,
            landmark_list=landmarks,
            connections=self.processor.mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=self.processor.mp_drawing.DrawingSpec(
                color=(255, 0, 0), thickness=2, circle_radius=2),
            connection_drawing_spec=self.processor.mp_drawing.DrawingSpec(
                color=(0, 0, 255), thickness=2)
        )
        
        # Add posture indicator
        if 'posture' in metrics:
            self._draw_posture_indicator(frame, metrics['posture'])
        
        return frame
    
    def _draw_posture_indicator(self, frame: np.ndarray, posture: str) -> None:
        """Draw posture quality indicator"""
        h, w = frame.shape[:2]
        indicator_color = (0, 255, 0) if posture == "Good" else (0, 0, 255)
        
        cv2.putText(frame, f"Posture: {posture}", (w-200, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, indicator_color, 2)
    
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

    def _analyze_expression(self, landmarks) -> str:
        """Analyze facial landmarks to determine expression
        
        Args:
            landmarks: Face landmarks from MediaPipe
            
        Returns:
            str: Detected expression
        """
        try:
            # Extract key facial features
            # Mouth corners for smile detection
            left_mouth = landmarks.landmark[self.processor.MOUTH_CORNERS[0]]
            right_mouth = landmarks.landmark[self.processor.MOUTH_CORNERS[1]]
            
            # Eyebrow positions for various expressions
            left_brow = [landmarks.landmark[i] for i in self.processor.LEFT_EYEBROW]
            right_brow = [landmarks.landmark[i] for i in self.processor.RIGHT_EYEBROW]
            
            # Check for smile (mouth corners up)
            mouth_angle = (right_mouth.y - left_mouth.y) / abs(right_mouth.x - left_mouth.x)
            is_smiling = mouth_angle < -0.1
            
            # Check eyebrow positions
            left_brow_height = sum(p.y for p in left_brow) / len(left_brow)
            right_brow_height = sum(p.y for p in right_brow) / len(right_brow)
            avg_brow_height = (left_brow_height + right_brow_height) / 2
            
            # Determine expression based on features
            if is_smiling:
                return "happy"
            elif avg_brow_height < 0.3:  # Raised eyebrows
                return "surprised"
            elif avg_brow_height > 0.4:  # Lowered eyebrows
                return "concerned"
            else:
                return "neutral"
                
        except Exception as e:
            print(f"Error in expression analysis: {str(e)}")
            return "neutral"
        
    def _generate_face_feedback(self, face_metrics: dict) -> List[str]:
        """Generate feedback messages based on facial analysis metrics
        
        Args:
            face_metrics: Dictionary containing face analysis results
            
        Returns:
            List[str]: List of feedback messages
        """
        feedback = []
        
        # Check head pose
        if 'head_pose' in face_metrics:
            pose = face_metrics['head_pose']
            if abs(pose['pitch']) > 20:
                feedback.append("Keep your head level")
            if abs(pose['yaw']) > 30:
                feedback.append("Face the camera more directly")
            if abs(pose['roll']) > 15:
                feedback.append("Keep your head straight")
        
        # Check eye contact
        if 'eye_contact_score' in face_metrics:
            score = face_metrics['eye_contact_score']
            if score < 0.5:
                feedback.append("Try to maintain more eye contact")
            elif score > 0.8:
                feedback.append("Good eye contact")
                
        # Check expression
        if 'expression' in face_metrics:
            exp = face_metrics['expression']
            if exp == 'neutral':
                feedback.append("Try to be more expressive")
            elif exp == 'happy':
                feedback.append("Good positive expression")
                
        return feedback

    def _get_shoulder_points(self, landmarks) -> tuple:
        """Get shoulder landmark points from pose estimation
        
        Args:
            landmarks: Pose landmarks from MediaPipe
            
        Returns:
            tuple: Left and right shoulder points
        """
        # MediaPipe pose landmark indices for shoulders
        LEFT_SHOULDER = 11
        RIGHT_SHOULDER = 12
        
        try:
            left_shoulder = landmarks.landmark[LEFT_SHOULDER]
            right_shoulder = landmarks.landmark[RIGHT_SHOULDER]
            return left_shoulder, right_shoulder
        except:
            return None, None

    def _analyze_posture_quality(self, shoulders: tuple, hips: tuple) -> dict:
        """Analyze posture quality from shoulder and hip positions
        
        Args:
            shoulders: Tuple of (left_shoulder, right_shoulder) landmarks
            hips: Tuple of (left_hip, right_hip) landmarks
            
        Returns:
            dict: Posture quality metrics
        """
        try:
            left_shoulder, right_shoulder = shoulders
            left_hip, right_hip = hips
            
            # Calculate alignment scores
            shoulder_alignment = abs(left_shoulder.y - right_shoulder.y)
            hip_alignment = abs(left_hip.y - right_hip.y)
            
            # Calculate vertical alignment
            shoulder_center_x = (left_shoulder.x + right_shoulder.x) / 2
            hip_center_x = (left_hip.x + right_hip.x) / 2
            vertical_alignment = abs(shoulder_center_x - hip_center_x)
            
            # Define thresholds
            ALIGNMENT_THRESHOLD = 0.05
            VERTICAL_THRESHOLD = 0.05
            
            # Calculate overall quality score
            quality_score = 1.0
            quality_score -= min(0.3, shoulder_alignment)  # Reduce score for shoulder tilt
            quality_score -= min(0.3, hip_alignment)      # Reduce score for hip tilt
            quality_score -= min(0.4, vertical_alignment) # Reduce score for leaning
            
            # Determine posture status
            if quality_score > 0.8:
                status = "Good Posture"
            elif quality_score > 0.6:
                issues = []
                if shoulder_alignment > ALIGNMENT_THRESHOLD:
                    issues.append("shoulders not level")
                if hip_alignment > ALIGNMENT_THRESHOLD:
                    issues.append("hips not level")
                if vertical_alignment > VERTICAL_THRESHOLD:
                    issues.append("leaning")
                status = f"Adjust posture: {', '.join(issues)}"
            else:
                status = "Poor posture"
            
            return {
                'posture_quality': quality_score,
                'posture_status': status,
                'shoulder_alignment': shoulder_alignment,
                'hip_alignment': hip_alignment,
                'vertical_alignment': vertical_alignment
            }
            
        except Exception as e:
            print(f"Error analyzing posture quality: {str(e)}")
            return {
                'posture_quality': 0.5,
                'posture_status': "Unknown",
                'shoulder_alignment': 0,
                'hip_alignment': 0,
                'vertical_alignment': 0
            }
        
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
            for point_name, weight in weights.items():
                if point_name in current_position and point_name in self.position_baseline:
                    current = current_position[point_name]
                    baseline = self.position_baseline[point_name]
                    
                    # Skip if either point is not a dict
                    if not isinstance(current, dict) or not isinstance(baseline, dict):
                        continue
                    
                    # Only consider points with good visibility
                    if (current.get('visibility', 0) > 0.5 and 
                        baseline.get('visibility', 0) > 0.5):
                        
                        # Calculate weighted Euclidean distance
                        dist = np.sqrt(
                            (current.get('x', 0) - baseline.get('x', 0))**2 +
                            (current.get('y', 0) - baseline.get('y', 0))**2
                        ) * weight
                        
                        movement_score += dist
                        valid_points += 1
                
            # Normalize movement score
            if valid_points > 0:
                movement_score /= valid_points
                
            # Update baseline with slight smoothing
            for point_name in current_position:
                if (point_name in self.position_baseline and 
                    isinstance(current_position[point_name], dict) and 
                    isinstance(self.position_baseline[point_name], dict)):
                        
                    alpha = 0.3  # Smoothing factor
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

    def _calculate_shoulder_alignment(self, shoulders: tuple) -> float:
        """Calculate shoulder alignment score
        
        Args:
            shoulders: Tuple of (left_shoulder, right_shoulder) landmarks
            
        Returns:
            float: Alignment score between 0 and 1
        """
        try:
            left_shoulder, right_shoulder = shoulders
            if not left_shoulder or not right_shoulder:
                return 0.5  # Default value if shoulders not detected
                
            # Calculate vertical alignment (shoulders should be level)
            vertical_diff = abs(left_shoulder.y - right_shoulder.y)
            
            # Calculate horizontal alignment (shoulders should be apart)
            horizontal_diff = abs(left_shoulder.x - right_shoulder.x)
            
            # Good shoulder width is about 0.2-0.3 in normalized coordinates
            ideal_width = 0.25
            width_score = 1.0 - min(1.0, abs(horizontal_diff - ideal_width) / ideal_width)
            
            # Convert vertical difference to score (smaller is better)
            vertical_score = 1.0 - min(1.0, vertical_diff * 5)  # Scale up for sensitivity
            
            # Combine scores (vertical alignment is more important)
            alignment_score = 0.7 * vertical_score + 0.3 * width_score
            
            return max(0.0, min(1.0, alignment_score))
            
        except Exception as e:
            print(f"Error calculating shoulder alignment: {str(e)}")
            return 0.5  # Default middle value on error

    def _calculate_stability(self, shoulders: tuple, hips: tuple) -> float:
        """Calculate stability score from shoulder and hip positions
        
        Args:
            shoulders: Tuple of (left_shoulder, right_shoulder) landmarks
            hips: Tuple of (left_hip, right_hip) landmarks
            
        Returns:
            float: Stability score between 0 and 1
        """
        try:
            left_shoulder, right_shoulder = shoulders
            left_hip, right_hip = hips
            
            if not all([left_shoulder, right_shoulder, left_hip, right_hip]):
                return 0.5  # Default if points missing
                
            # Calculate center points
            shoulder_center_x = (left_shoulder.x + right_shoulder.x) / 2
            hip_center_x = (left_hip.x + right_hip.x) / 2
            
            # Calculate sway (horizontal deviation)
            sway = abs(shoulder_center_x - hip_center_x)
            
            # Calculate shoulder width for normalization
            shoulder_width = abs(left_shoulder.x - right_shoulder.x)
            
            # Normalize sway by shoulder width
            normalized_sway = sway / max(shoulder_width, 0.1)
            
            # Convert to stability score (less sway = more stable)
            stability = 1.0 - min(1.0, normalized_sway * 3)  # Scale for sensitivity
            
            return stability
            
        except Exception as e:
            print(f"Error calculating stability: {str(e)}")
            return 0.5  # Default middle value on error

    def _generate_pose_feedback(self, pose_metrics: dict) -> List[str]:
        """Generate feedback based on pose analysis
        
        Args:
            pose_metrics: Dictionary containing pose analysis results
            
        Returns:
            List[str]: List of feedback messages
        """
        feedback = []
        
        try:
            # Check posture quality
            if 'posture_quality' in pose_metrics:
                quality = pose_metrics['posture_quality']
                if quality < 0.5:
                    feedback.append("Correct your posture")
                elif quality < 0.7:
                    feedback.append(pose_metrics.get('posture_status', 'Improve posture'))
                elif quality > 0.8:
                    feedback.append("Good posture")
            
            # Check shoulder alignment
            if 'shoulder_alignment' in pose_metrics:
                if pose_metrics['shoulder_alignment'] > 0.1:
                    feedback.append("Level your shoulders")
                    
            # Check vertical alignment
            if 'vertical_alignment' in pose_metrics:
                if pose_metrics['vertical_alignment'] > 0.1:
                    feedback.append("Stand up straight")
                    
            # If no issues found and no positive feedback given
            if not feedback:
                feedback.append("Pose looks good")
                
            return feedback
            
        except Exception as e:
            print(f"Error generating pose feedback: {str(e)}")
            return ["Check your posture"]

    def process(self, frame: np.ndarray, audio_data: Optional[np.ndarray],
                metrics: Dict) -> Dict:
        """Process frame and return metrics (implements FeatureModule interface)"""
        video_metrics, _, _ = self.analyze_frame(frame)
        return video_metrics