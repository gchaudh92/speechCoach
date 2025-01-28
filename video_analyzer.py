import cv2
import mediapipe as mp
import numpy as np
from collections import deque

# class VideoAnalyzer:
#     def __init__(self):
#         self.mp_face_mesh = mp.solutions.face_mesh
#         self.mp_drawing = mp.solutions.drawing_utils
        
#         self.face_mesh = self.mp_face_mesh.FaceMesh(
#             max_num_faces=1,
#             refine_landmarks=True,
#             min_detection_confidence=0.5,
#             min_tracking_confidence=0.5
#         )
        
#         # Track face orientation history
#         self.orientation_history = deque(maxlen=30)
#         self.position_history = deque(maxlen=30)
        
#         # Key facial landmarks
#         self.NOSE_TIP = 4
#         self.CHIN = 152
#         self.LEFT_EYE = [33, 133]  # Left eye corners
#         self.RIGHT_EYE = [362, 263]  # Right eye corners
#         self.MOUTH_CORNERS = [61, 291]  # Left and right mouth corners
#         self.MOUTH_TOP = 13
#         self.MOUTH_BOTTOM = 14
#         self.LEFT_EYEBROW = [107, 105, 70]  # Left eyebrow points
#         self.RIGHT_EYEBROW = [336, 334, 300]  # Right eyebrow points
        
#         # Initialize baseline values
#         self.baseline_pose = None
#         self.baseline_established = False
        
#         # Expression tracking
#         self.last_expression = None
#         self.expression_start_time = None
#         self.sustained_duration = 0
        
#     def _calculate_3d_angle(self, point1, point2, point3):
#         """Calculate angle between three 3D points."""
#         vector1 = np.array([point1.x - point2.x, point1.y - point2.y, point1.z - point2.z])
#         vector2 = np.array([point3.x - point2.x, point3.y - point2.y, point3.z - point2.z])
        
#         cosine = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
#         angle = np.arccos(np.clip(cosine, -1.0, 1.0))
#         return np.degrees(angle)

#     def _get_head_pose(self, landmarks):
#         """Calculate head pose angles."""
#         # Get nose and eye positions
#         nose = landmarks.landmark[self.NOSE_TIP]
#         left_eye = landmarks.landmark[self.LEFT_EYE[0]]
#         right_eye = landmarks.landmark[self.RIGHT_EYE[0]]
#         chin = landmarks.landmark[self.CHIN]
        
#         # Calculate angles
#         # Yaw (left-right rotation)
#         yaw = np.arctan2(nose.x - (left_eye.x + right_eye.x)/2, nose.z - (left_eye.z + right_eye.z)/2)
        
#         # Pitch (up-down rotation)
#         pitch = np.arctan2(nose.y - chin.y, nose.z - chin.z)
        
#         # Roll (tilt)
#         roll = np.arctan2(right_eye.y - left_eye.y, right_eye.x - left_eye.x)
        
#         return np.degrees([pitch, yaw, roll])

#     def _analyze_mouth(self, landmarks):
#         """Analyze mouth state and movement."""
#         mouth_top = landmarks.landmark[self.MOUTH_TOP]
#         mouth_bottom = landmarks.landmark[self.MOUTH_BOTTOM]
#         left_corner = landmarks.landmark[self.MOUTH_CORNERS[0]]
#         right_corner = landmarks.landmark[self.MOUTH_CORNERS[1]]
        
#         # Calculate mouth aspect ratio
#         mouth_height = np.linalg.norm([mouth_top.x - mouth_bottom.x, 
#                                      mouth_top.y - mouth_bottom.y])
#         mouth_width = np.linalg.norm([left_corner.x - right_corner.x,
#                                     left_corner.y - right_corner.y])
        
#         mouth_ratio = mouth_height / mouth_width if mouth_width > 0 else 0
#         return mouth_ratio

#     def _analyze_eyebrows(self, landmarks):
#         """Analyze eyebrow position for expressions."""
#         left_brow_height = np.mean([landmarks.landmark[i].y for i in self.LEFT_EYEBROW])
#         right_brow_height = np.mean([landmarks.landmark[i].y for i in self.RIGHT_EYEBROW])
#         return (left_brow_height + right_brow_height) / 2

#     def _determine_expression(self, landmarks):
#         """Determine facial expression based on multiple features."""
#         mouth_ratio = self._analyze_mouth(landmarks)
#         brow_height = self._analyze_eyebrows(landmarks)
        
#         # Expression classification
#         if mouth_ratio > 0.7:  # Wide open mouth
#             return "Wide Open Mouth"
#         elif mouth_ratio > 0.4:  # Speaking or slight open
#             return "Speaking"
#         elif mouth_ratio > 0.25:  # Smile
#             if brow_height < 0.3:  # Raised eyebrows
#                 return "Happy"
#             return "Slight Smile"
#         else:  # Closed mouth
#             if brow_height < 0.28:  # Raised eyebrows
#                 return "Surprised"
#             elif brow_height > 0.35:  # Lowered eyebrows
#                 return "Frowning"
#             return "Neutral"

#     def analyze_frame(self, frame):
#         """Analyze facial expressions and movements in a frame."""
#         metrics = {
#             'head_pose': {'pitch': 0, 'yaw': 0, 'roll': 0},
#             'expression': 'No Face Detected',
#             'movement_alert': '',
#             'pose_alert': ''
#         }
        
#         feedback = []  # Initialize feedback list
        
#         rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         results = self.face_mesh.process(rgb_frame)
        
#         if not results.multi_face_landmarks:
#             return metrics, feedback, frame
            
#         landmarks = results.multi_face_landmarks[0]
        
#         # Calculate head pose
#         pose_angles = self._get_head_pose(landmarks)
#         metrics['head_pose'] = {
#             'pitch': pose_angles[0],
#             'yaw': pose_angles[1],
#             'roll': pose_angles[2]
#         }
        
#         # Analyze expression
#         metrics['expression'] = self._determine_expression(landmarks)
        
#         # Generate alerts for significant head movements
#         if abs(pose_angles[1]) > 30:  # Yaw
#             feedback.append('Head turned too far')
#         elif abs(pose_angles[0]) > 20:  # Pitch
#             feedback.append('Head tilted up/down too much')
#         elif abs(pose_angles[2]) > 15:  # Roll
#             feedback.append('Head tilted sideways too much')
            
#         # Draw annotations
#         annotated_frame = self._draw_annotations(frame.copy(), landmarks, metrics)
        
#         return metrics, feedback, annotated_frame

#     def _draw_annotations(self, frame, landmarks, metrics):
#         """Draw facial landmarks and metrics on frame."""
#         # Draw face mesh
#         self.mp_drawing.draw_landmarks(
#             image=frame,
#             landmark_list=landmarks,
#             connections=self.mp_face_mesh.FACEMESH_TESSELATION,
#             landmark_drawing_spec=self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
#             connection_drawing_spec=self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1)
#         )
        
#         # Draw metrics
#         y_offset = 30
#         font = cv2.FONT_HERSHEY_SIMPLEX
        
#         # Draw expression and pose information
#         cv2.putText(frame, f"Expression: {metrics['expression']}", 
#                    (10, y_offset), font, 1.0, (0, 0, 255), 2)
#         y_offset += 40
        
#         pose = metrics['head_pose']
#         cv2.putText(frame, f"Head Pitch: {pose['pitch']:.1f}", 
#                    (10, y_offset), font, 1.0, (0, 0, 255), 2)
#         y_offset += 40
        
#         cv2.putText(frame, f"Head Yaw: {pose['yaw']:.1f}", 
#                    (10, y_offset), font, 1.0, (0, 0, 255), 2)
#         y_offset += 40
        
#         cv2.putText(frame, f"Head Roll: {pose['roll']:.1f}", 
#                    (10, y_offset), font, 1.0, (0, 0, 255), 2)
#         y_offset += 40
        
#         # Draw alerts if any
#         if metrics['pose_alert']:
#             cv2.putText(frame, f"Alert: {metrics['pose_alert']}", 
#                        (10, y_offset), font, 1.0, (0, 0, 255), 2)
        
#         return frame
    
import cv2
import mediapipe as mp
import numpy as np
from collections import deque

class VideoAnalyzer:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            enable_segmentation=False,
            min_detection_confidence=0.5
        )
        
        # Key facial landmarks
        self.NOSE_TIP = 4
        self.CHIN = 152
        self.LEFT_EYE = [33, 133]
        self.RIGHT_EYE = [362, 263]
        self.MOUTH_CORNERS = [61, 291]
        self.MOUTH_TOP = 13
        self.MOUTH_BOTTOM = 14

    def _get_head_pose(self, landmarks):
        """Calculate head pose angles (pitch, yaw, roll) based on facial landmarks."""
        # Get nose and eye positions
        nose = landmarks.landmark[self.NOSE_TIP]
        left_eye = landmarks.landmark[self.LEFT_EYE[0]]
        right_eye = landmarks.landmark[self.RIGHT_EYE[0]]
        chin = landmarks.landmark[self.CHIN]

        # Calculate angles
        yaw = np.arctan2(nose.x - (left_eye.x + right_eye.x) / 2, nose.z - (left_eye.z + right_eye.z) / 2)
        pitch = np.arctan2(nose.y - chin.y, nose.z - chin.z)
        roll = np.arctan2(right_eye.y - left_eye.y, right_eye.x - left_eye.x)
        
        return np.degrees([pitch, yaw, roll])

    def analyze_frame(self, frame):
        """Analyze facial expressions, head movements, and posture."""
        metrics = {
            'head_pose': {'pitch': 0, 'yaw': 0, 'roll': 0},
            'expression': 'No Face Detected',
            'posture': 'Good'
        }
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Analyze face
        face_results = self.face_mesh.process(rgb_frame)
        if face_results.multi_face_landmarks:
            landmarks = face_results.multi_face_landmarks[0]
            pose_angles = self._get_head_pose(landmarks)
            metrics['head_pose'] = {
                'pitch': pose_angles[0],
                'yaw': pose_angles[1],
                'roll': pose_angles[2]
            }
            metrics['expression'] = self._determine_expression(landmarks)
        
        # Analyze posture
        pose_results = self.pose.process(rgb_frame)
        if pose_results.pose_landmarks:
            metrics['posture'] = self._analyze_posture(pose_results.pose_landmarks)
        
        # Draw annotations
        annotated_frame = self._draw_annotations(frame.copy(), face_results, pose_results, metrics)
        
        return metrics, [], annotated_frame

    def _analyze_posture(self, landmarks):
        """Analyze posture using pose landmarks."""
        left_shoulder = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
        
        shoulder_diff = abs(left_shoulder.y - right_shoulder.y)
        if shoulder_diff > 0.1:
            return "Uneven Shoulders"
        
        return "Good Posture"

    def _draw_annotations(self, frame, face_results, pose_results, metrics):
        """Draw facial and posture landmarks on the frame."""
        if face_results.multi_face_landmarks:
            self.mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_results.multi_face_landmarks[0],
                connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
                connection_drawing_spec=self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1)
            )
        
        if pose_results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=pose_results.pose_landmarks,
                connections=self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                connection_drawing_spec=self.mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
            )
        
        return frame
     