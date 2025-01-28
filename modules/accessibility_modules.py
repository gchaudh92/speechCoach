import cv2
import pyttsx3
import numpy as np
from collections import deque
import mediapipe as mp
from typing import Dict, List, Optional, Tuple
import queue
import threading
import time 
from .utils import FeatureModule, AnalyzerConfig

class AccessibilityFeatures(FeatureModule):
    def __init__(self, config: AnalyzerConfig):
        super().__init__(config)
        self.caption_generator = CaptionGenerator(config)
        self.color_enhancer = ColorEnhancer(config)
        self.sign_language = SignLanguageRecognizer(config)
        self.text_to_speech = TextToSpeechEngine(config)
        self.visual_aids = VisualAidsManager(config)
        
        # Feature toggles
        self.features_enabled = {
            'captions': True,
            'color_enhancement': True,
            'sign_language': True,
            'text_to_speech': True,
            'visual_aids': True
        }
        
    def process(self, frame, audio_data, metrics: dict) -> Tuple[np.ndarray, dict]:
        """Process frame with all enabled accessibility features"""
        accessibility_metrics = {}
        enhanced_frame = frame.copy()
        
        try:
            # Generate captions if speech detected
            if self.features_enabled['captions'] and 'transcribed_text' in metrics:
                captions = self.caption_generator.process(frame, audio_data, metrics)
                accessibility_metrics.update(captions)
                enhanced_frame = self.caption_generator.draw_captions(enhanced_frame, captions)
            
            # Enhance colors for color-blind accessibility
            if self.features_enabled['color_enhancement']:
                color_metrics = self.color_enhancer.process(frame, audio_data, metrics)
                accessibility_metrics.update(color_metrics)
                enhanced_frame = self.color_enhancer.enhance_frame(enhanced_frame)
            
            # Process sign language if enabled
            if self.features_enabled['sign_language']:
                sign_metrics = self.sign_language.process(frame, audio_data, metrics)
                accessibility_metrics.update(sign_metrics)
                enhanced_frame = self.sign_language.draw_recognition(enhanced_frame, sign_metrics)
            
            # Process text-to-speech if enabled
            if self.features_enabled['text_to_speech'] and 'transcribed_text' in metrics:
                tts_metrics = self.text_to_speech.process(frame, audio_data, metrics)
                accessibility_metrics.update(tts_metrics)
            
            # Add visual aids
            if self.features_enabled['visual_aids']:
                visual_metrics = self.visual_aids.process(frame, audio_data, metrics)
                accessibility_metrics.update(visual_metrics)
                enhanced_frame = self.visual_aids.draw_aids(enhanced_frame, visual_metrics)
            
        except Exception as e:
            print(f"Error in accessibility processing: {str(e)}")
        
        return enhanced_frame, accessibility_metrics
    
    def toggle_feature(self, feature_name: str) -> bool:
        """Toggle specific accessibility feature"""
        if feature_name in self.features_enabled:
            self.features_enabled[feature_name] = not self.features_enabled[feature_name]
            return self.features_enabled[feature_name]
        return False

class CaptionGenerator(FeatureModule):
    def __init__(self, config: AnalyzerConfig):
        super().__init__(config)
        self.caption_history = deque(maxlen=5)
        self.caption_queue = queue.Queue()
        self.caption_settings = {
            'font_size': 1.0,
            'position': 'bottom',  # bottom, top
            'background_opacity': 0.7,
            'text_color': (255, 255, 255),
            'background_color': (0, 0, 0)
        }
        
    def process(self, frame, audio_data, metrics: dict) -> dict:
        """Process speech to generate captions"""
        if 'transcribed_text' not in metrics:
            return {}
            
        text = metrics['transcribed_text']
        timestamp = metrics.get('timestamp', 0)
        
        caption = {
            'text': text,
            'timestamp': timestamp,
            'duration': 2.0  # seconds to display
        }
        
        self.caption_history.append(caption)
        self.caption_queue.put(caption)
        
        return {
            'caption_text': text,
            'caption_timestamp': timestamp
        }
    
    def draw_captions(self, frame: np.ndarray, caption_metrics: dict) -> np.ndarray:
        """Draw captions on frame"""
        if not caption_metrics or 'caption_text' not in caption_metrics:
            return frame
            
        h, w = frame.shape[:2]
        text = caption_metrics['caption_text']
        
        # Calculate text size and position
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = self.caption_settings['font_size']
        thickness = 2
        
        (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        
        # Calculate position
        if self.caption_settings['position'] == 'bottom':
            text_x = (w - text_w) // 2
            text_y = h - 50
        else:
            text_x = (w - text_w) // 2
            text_y = 50
        
        # Draw background
        padding = 10
        bg_rect = (
            text_x - padding,
            text_y - text_h - padding,
            text_w + 2 * padding,
            text_h + 2 * padding
        )
        
        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (bg_rect[0], bg_rect[1]),
            (bg_rect[0] + bg_rect[2], bg_rect[1] + bg_rect[3]),
            self.caption_settings['background_color'],
            -1
        )
        
        # Blend background
        alpha = self.caption_settings['background_opacity']
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
        
        # Draw text
        cv2.putText(
            frame,
            text,
            (text_x, text_y),
            font,
            font_scale,
            self.caption_settings['text_color'],
            thickness
        )
        
        return frame

class ColorEnhancer(FeatureModule):
    def __init__(self, config: AnalyzerConfig):
        super().__init__(config)
        self.enhancement_mode = 'deuteranopia'  # deuteranopia, protanopia, tritanopia
        self.enhancement_strength = 1.0
        
    def process(self, frame, audio_data=None, metrics=None) -> dict:
        """Process frame for color enhancement
        
        Args:
            frame: Video frame to process
            audio_data: Optional audio data (not used)
            metrics: Optional metrics dictionary (not used)
            
        Returns:
            dict: Color analysis metrics
        """
        if frame is None:
            return {}
                
        try:
            color_metrics = self._analyze_colors(frame)
            return {
                'color_mode': self.enhancement_mode,
                'color_metrics': color_metrics
            }
        except Exception as e:
            print(f"Error in color enhancement: {str(e)}")
            return {
                'color_mode': self.enhancement_mode,
                'color_metrics': {}
            }
    
    def enhance_frame(self, frame: np.ndarray) -> np.ndarray:
        """Enhance frame colors for color-blind accessibility"""
        if self.enhancement_mode == 'deuteranopia':
            return self._enhance_deuteranopia(frame)
        elif self.enhancement_mode == 'protanopia':
            return self._enhance_protanopia(frame)
        elif self.enhancement_mode == 'tritanopia':
            return self._enhance_tritanopia(frame)
        return frame
    
    def _analyze_colors(self, frame: np.ndarray) -> dict:
        """Analyze color distribution in frame"""
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        return {
            'luminance_mean': np.mean(l),
            'a_mean': np.mean(a),
            'b_mean': np.mean(b),
            'contrast_ratio': self._calculate_contrast_ratio(frame)
        }
    
    def _enhance_deuteranopia(self, frame: np.ndarray) -> np.ndarray:
        """Enhance colors for deuteranopia"""
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Enhance red-green contrast
        a = cv2.multiply(a, 1.5)
        
        enhanced = cv2.merge([l, a, b])
        return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    
    def _calculate_contrast_ratio(self, frame: np.ndarray) -> float:
        """Calculate contrast ratio of frame"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        min_val = np.min(gray)
        max_val = np.max(gray)
        
        if min_val == 0:
            min_val = 1
            
        return max_val / min_val

class SignLanguageRecognizer(FeatureModule):
    def __init__(self, config: AnalyzerConfig):
        super().__init__(config)
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7
        )
        self.gesture_history = deque(maxlen=30)
        
    def process(self, frame, audio_data, metrics: dict) -> dict:
        """Process frame for sign language recognition"""
        if frame is None:
            return {}
            
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        sign_metrics = {
            'sign_detected': False,
            'recognized_sign': None,
            'confidence': 0.0
        }
        
        if results.multi_hand_landmarks:
            sign_metrics.update(self._recognize_sign(results.multi_hand_landmarks))
            
        return sign_metrics
    
    def draw_recognition(self, frame: np.ndarray, metrics: dict) -> np.ndarray:
        """Draw sign language recognition results"""
        if not metrics.get('sign_detected', False):
            return frame
            
        h, w = frame.shape[:2]
        sign = metrics.get('recognized_sign', '')
        confidence = metrics.get('confidence', 0.0)
        
        # Draw recognition result
        text = f"Sign: {sign} ({confidence:.2f})"
        cv2.putText(frame, text, (10, h-20), cv2.FONT_HERSHEY_SIMPLEX,
                   0.7, (0, 255, 0), 2)
        
        return frame
    
    def _recognize_sign(self, hand_landmarks: list) -> dict:
        """Recognize sign language gestures"""
        # Placeholder for sign recognition logic
        return {
            'sign_detected': True,
            'recognized_sign': 'Hello',
            'confidence': 0.8
        }

class TextToSpeechEngine(FeatureModule):
    def __init__(self, config: AnalyzerConfig):
        super().__init__(config)
        self.tts_queue = queue.Queue()
        self.speaking = False
        self.settings = {
            'voice': 'default',
            'speed': 1.0,
            'pitch': 1.0
        }
        
    def process(self, frame, audio_data, metrics: dict) -> dict:
        """Process text for speech synthesis"""
        if 'transcribed_text' not in metrics:
            return {}
            
        text = metrics['transcribed_text']
        self.tts_queue.put(text)
        
        return {
            'tts_text': text,
            'tts_status': 'queued' if self.speaking else 'ready'
        }

class VisualAidsManager(FeatureModule):
    def __init__(self, config: AnalyzerConfig):
        super().__init__(config)
        self.aids_enabled = {
            'high_contrast': True,
            'focus_highlight': True,
            'gesture_guides': True
        }
        
    def process(self, frame, audio_data, metrics: dict) -> dict:
        """Process frame for visual aids"""
        aids_metrics = {}
        
        if self.aids_enabled['high_contrast']:
            aids_metrics.update(self._process_contrast(frame))
            
        if self.aids_enabled['focus_highlight']:
            aids_metrics.update(self._process_focus(frame, metrics))
            
        return aids_metrics
    
    def draw_aids(self, frame: np.ndarray, metrics: dict) -> np.ndarray:
        """Draw visual aids on frame"""
        enhanced_frame = frame.copy()
        
        if self.aids_enabled['high_contrast']:
            enhanced_frame = self._apply_contrast(enhanced_frame, metrics)
            
        if self.aids_enabled['focus_highlight']:
            enhanced_frame = self._apply_focus_highlight(enhanced_frame, metrics)
            
        return enhanced_frame
    
    def _process_contrast(self, frame: np.ndarray) -> dict:
        """Process frame for contrast enhancement"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return {
            'contrast_level': np.std(gray),
            'brightness_level': np.mean(gray)
        }
    
    def _process_focus(self, frame: np.ndarray, metrics: dict) -> dict:
        """Process frame for focus areas"""
        # Placeholder for focus detection logic
        return {
            'focus_areas': [(100, 100, 200, 200)],
            'focus_confidence': 0.8
        }
    
    def _apply_contrast(self, frame: np.ndarray, metrics: dict) -> np.ndarray:
        """Apply contrast enhancement"""
        # Enhance contrast based on metrics
        return frame
    
    def _apply_focus_highlight(self, frame: np.ndarray, metrics: dict) -> np.ndarray:
        """Apply focus area highlighting"""
        focus_areas = metrics.get('focus_areas', [])
        
        for area in focus_areas:
            x, y, w, h = area
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
        return frame

class GestureRecognizer(FeatureModule):
    def __init__(self, config: AnalyzerConfig):
        super().__init__(config)
        self.mp_hands = mp.solutions.hands
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Initialize detectors
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5
        )
        
        # Gesture history for smoothing
        self.gesture_history = deque(maxlen=30)
        self.pose_history = deque(maxlen=30)
        
        # Gesture definitions
        self.gesture_templates = {
            'wave': self._define_wave_gesture(),
            'point': self._define_point_gesture(),
            'thumbs_up': self._define_thumbs_up_gesture(),
            'open_palm': self._define_open_palm_gesture(),
            'closed_fist': self._define_closed_fist_gesture()
        }
        
    def process(self, frame, audio_data, metrics: dict) -> dict:
        """Process frame for gesture recognition"""
        if frame is None:
            return {}
            
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process hands
        hand_results = self.hands.process(rgb_frame)
        hand_gestures = self._process_hand_gestures(hand_results) if hand_results.multi_hand_landmarks else {}
        
        # Process pose
        pose_results = self.pose.process(rgb_frame)
        pose_gestures = self._process_pose_gestures(pose_results) if pose_results.pose_landmarks else {}
        
        # Combine and analyze gestures
        combined_metrics = self._combine_gesture_analysis(hand_gestures, pose_gestures)
        
        return combined_metrics
    
    def _process_hand_gestures(self, results) -> dict:
        """Process hand landmarks for gesture recognition"""
        gestures = []
        confidences = []
        
        for hand_landmarks in results.multi_hand_landmarks:
            # Extract hand features
            hand_features = self._extract_hand_features(hand_landmarks)
            
            # Match against gesture templates
            matched_gesture, confidence = self._match_hand_gesture(hand_features)
            
            if matched_gesture:
                gestures.append(matched_gesture)
                confidences.append(confidence)
        
        return {
            'hand_gestures': gestures,
            'hand_confidences': confidences
        }
    
    def _process_pose_gestures(self, results) -> dict:
        """Process body pose for gesture recognition"""
        if not results.pose_landmarks:
            return {}
            
        # Extract pose features
        pose_features = self._extract_pose_features(results.pose_landmarks)
        
        # Match against pose gesture templates
        matched_pose, confidence = self._match_pose_gesture(pose_features)
        
        return {
            'pose_gesture': matched_pose,
            'pose_confidence': confidence
        }
    
    def _extract_hand_features(self, landmarks) -> dict:
        """Extract relevant features from hand landmarks"""
        # Convert landmarks to numpy array
        points = np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark])
        
        # Calculate features
        features = {
            'finger_angles': self._calculate_finger_angles(points),
            'palm_direction': self._calculate_palm_direction(points),
            'finger_distances': self._calculate_finger_distances(points),
            'hand_shape': self._analyze_hand_shape(points)
        }
        
        return features
    
    def _extract_pose_features(self, landmarks) -> dict:
        """Extract relevant features from pose landmarks"""
        points = np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark])
        
        features = {
            'arm_angles': self._calculate_arm_angles(points),
            'body_orientation': self._calculate_body_orientation(points),
            'movement_direction': self._analyze_movement_direction(points)
        }
        
        return features
    
    def _match_hand_gesture(self, features: dict) -> Tuple[str, float]:
        """Match hand features against gesture templates"""
        best_match = None
        best_confidence = 0.0
        
        for gesture_name, template in self.gesture_templates.items():
            confidence = self._calculate_gesture_similarity(features, template)
            if confidence > best_confidence and confidence > self.config.threshold:
                best_match = gesture_name
                best_confidence = confidence
        
        return best_match, best_confidence
    
    def _match_pose_gesture(self, features: dict) -> Tuple[str, float]:
        """Match pose features against pose gesture templates"""
        # Implement pose gesture matching
        return None, 0.0
    
    def _combine_gesture_analysis(self, hand_gestures: dict, pose_gestures: dict) -> dict:
        """Combine hand and pose gesture analysis"""
        combined_metrics = {
            'gestures_detected': bool(hand_gestures or pose_gestures),
            'hand_gestures': hand_gestures.get('hand_gestures', []),
            'hand_confidences': hand_gestures.get('hand_confidences', []),
            'pose_gesture': pose_gestures.get('pose_gesture'),
            'pose_confidence': pose_gestures.get('pose_confidence', 0.0)
        }
        
        # Update gesture history
        if combined_metrics['gestures_detected']:
            self.gesture_history.append(combined_metrics)
        
        # Add gesture stability metrics
        combined_metrics['gesture_stability'] = self._calculate_gesture_stability()
        
        return combined_metrics
    
    def _calculate_gesture_stability(self) -> float:
        """Calculate stability of gestures over time"""
        if len(self.gesture_history) < 2:
            return 1.0
            
        # Count gesture changes
        changes = 0
        for i in range(1, len(self.gesture_history)):
            prev = self.gesture_history[i-1]['hand_gestures']
            curr = self.gesture_history[i]['hand_gestures']
            if prev != curr:
                changes += 1
                
        stability = 1.0 - (changes / len(self.gesture_history))
        return max(0.0, min(1.0, stability))
    
    def _define_wave_gesture(self) -> dict:
        """Define template for wave gesture"""
        return {
            'finger_angles': [0.0, 0.0, 0.0, 0.0, 0.0],  # Extended fingers
            'palm_direction': 'vertical',
            'movement_pattern': 'oscillating'
        }
    
    def _define_point_gesture(self) -> dict:
        """Define template for point gesture"""
        return {
            'finger_angles': [0.0, 180.0, 180.0, 180.0, 180.0],  # Index extended
            'palm_direction': 'vertical',
            'movement_pattern': 'stable'
        }
    
    def _define_thumbs_up_gesture(self) -> dict:
        """Define template for thumbs up gesture"""
        return {
            'finger_angles': [180.0, 0.0, 0.0, 0.0, 0.0],  # Thumb extended
            'palm_direction': 'forward',
            'movement_pattern': 'stable'
        }
    
    def _define_open_palm_gesture(self) -> dict:
        """Define template for open palm gesture"""
        return {
            'finger_angles': [0.0, 0.0, 0.0, 0.0, 0.0],  # All fingers extended
            'palm_direction': 'forward',
            'movement_pattern': 'stable'
        }
    
    def _define_closed_fist_gesture(self) -> dict:
        """Define template for closed fist gesture"""
        return {
            'finger_angles': [180.0, 180.0, 180.0, 180.0, 180.0],  # All fingers closed
            'palm_direction': 'forward',
            'movement_pattern': 'stable'
        }
    
    def _calculate_gesture_similarity(self, features: dict, template: dict) -> float:
        """Calculate similarity between features and template"""
        # Implement similarity calculation
        return 0.8  # Placeholder
    
    def draw_gesture_recognition(self, frame: np.ndarray, metrics: dict) -> np.ndarray:
        """Draw gesture recognition results on frame"""
        if not metrics.get('gestures_detected', False):
            return frame
            
        h, w = frame.shape[:2]
        
        # Draw hand gestures
        for gesture, confidence in zip(
            metrics.get('hand_gestures', []),
            metrics.get('hand_confidences', [])
        ):
            text = f"{gesture}: {confidence:.2f}"
            cv2.putText(frame, text, (10, h-30), cv2.FONT_HERSHEY_SIMPLEX,
                       0.7, (0, 255, 0), 2)
            
        # Draw pose gesture if detected
        pose_gesture = metrics.get('pose_gesture')
        if pose_gesture:
            pose_conf = metrics.get('pose_confidence', 0.0)
            text = f"Pose: {pose_gesture} ({pose_conf:.2f})"
            cv2.putText(frame, text, (10, h-60), cv2.FONT_HERSHEY_SIMPLEX,
                       0.7, (0, 255, 0), 2)
            
        return frame 
    

class TextToSpeech(FeatureModule):
    def __init__(self, config: AnalyzerConfig):
        super().__init__(config)
        # Initialize TTS engine
        self.engine = pyttsx3.init()
        
        # Configure default properties
        self.settings = {
            'rate': 150,           # Speed of speech
            'volume': 1.0,         # Volume level (0.0 to 1.0)
            'voice_id': None,      # Default system voice
            'pitch': 100           # Voice pitch
        }
        
        # Initialize voice settings
        self._configure_engine()
        
        # Setup speech queue and processing thread
        self.speech_queue = queue.Queue()
        self.is_speaking = False
        self.should_stop = False
        self.current_speech = None
        self.speech_history = []
        
        # Start processing thread
        self.speech_thread = threading.Thread(target=self._process_speech_queue)
        self.speech_thread.daemon = True
        self.speech_thread.start()
        
        # Available voices cache
        self.available_voices = self._get_available_voices()
    
    def _configure_engine(self):
        """Configure TTS engine with current settings"""
        self.engine.setProperty('rate', self.settings['rate'])
        self.engine.setProperty('volume', self.settings['volume'])
        if self.settings['voice_id']:
            self.engine.setProperty('voice', self.settings['voice_id'])
    
    def _get_available_voices(self) -> List[Dict]:
        """Get list of available voices"""
        voices = []
        for voice in self.engine.getProperty('voices'):
            voices.append({
                'id': voice.id,
                'name': voice.name,
                'languages': voice.languages,
                'gender': voice.gender
            })
        return voices
    
    def process(self, frame, audio_data, metrics: dict) -> dict:
        """Process text for speech synthesis"""
        if 'transcribed_text' not in metrics:
            return {}
        
        text = metrics['transcribed_text']
        self.speak(text)
        
        return {
            'tts_text': text,
            'tts_status': 'speaking' if self.is_speaking else 'ready',
            'tts_queue_size': self.speech_queue.qsize()
        }
    
    def speak(self, text: str, priority: bool = False):
        """Add text to speech queue"""
        if not text:
            return
            
        speech_item = {
            'text': text,
            'timestamp': time.time(),
            'priority': priority,
            'settings': self.settings.copy()
        }
        
        if priority:
            # Clear queue for priority messages
            with self.speech_queue.mutex:
                self.speech_queue.queue.clear()
                
        self.speech_queue.put(speech_item)
    
    def _process_speech_queue(self):
        """Process speech queue in background thread"""
        while not self.should_stop:
            try:
                if not self.is_speaking and not self.speech_queue.empty():
                    speech_item = self.speech_queue.get()
                    self._speak_text(speech_item)
                    
                time.sleep(0.1)  # Prevent busy waiting
                    
            except Exception as e:
                print(f"Error in speech processing: {str(e)}")
    
    def _speak_text(self, speech_item: Dict):
        """Speak single text item"""
        try:
            self.is_speaking = True
            self.current_speech = speech_item
            
            # Apply specific settings for this speech item
            self._apply_speech_settings(speech_item['settings'])
            
            # Speak the text
            self.engine.say(speech_item['text'])
            self.engine.runAndWait()
            
            # Store in history
            self.speech_history.append(speech_item)
            
            # Cleanup
            self.current_speech = None
            self.is_speaking = False
            
        except Exception as e:
            print(f"Error in speech synthesis: {str(e)}")
            self.is_speaking = False
    
    def _apply_speech_settings(self, settings: Dict):
        """Apply specific speech settings"""
        self.engine.setProperty('rate', settings['rate'])
        self.engine.setProperty('volume', settings['volume'])
        if settings['voice_id']:
            self.engine.setProperty('voice', settings['voice_id'])
    
    def stop(self):
        """Stop current speech and clear queue"""
        self.engine.stop()
        with self.speech_queue.mutex:
            self.speech_queue.queue.clear()
        self.is_speaking = False
        self.current_speech = None
    
    def pause(self):
        """Pause speech"""
        self.engine.pause()
    
    def resume(self):
        """Resume speech"""
        self.engine.resume()
    
    def set_voice(self, voice_id: str):
        """Set voice by ID"""
        if voice_id in [v['id'] for v in self.available_voices]:
            self.settings['voice_id'] = voice_id
            self.engine.setProperty('voice', voice_id)
    
    def set_rate(self, rate: int):
        """Set speech rate (words per minute)"""
        self.settings['rate'] = max(50, min(300, rate))
        self.engine.setProperty('rate', self.settings['rate'])
    
    def set_volume(self, volume: float):
        """Set volume level (0.0 to 1.0)"""
        self.settings['volume'] = max(0.0, min(1.0, volume))
        self.engine.setProperty('volume', self.settings['volume'])
    
    def set_pitch(self, pitch: int):
        """Set voice pitch (0 to 200)"""
        self.settings['pitch'] = max(0, min(200, pitch))
    
    def get_current_state(self) -> Dict:
        """Get current TTS state"""
        return {
            'is_speaking': self.is_speaking,
            'queue_size': self.speech_queue.qsize(),
            'current_speech': self.current_speech,
            'settings': self.settings.copy()
        }
    
    def __del__(self):
        """Cleanup resources"""
        self.should_stop = True
        if hasattr(self, 'speech_thread'):
            self.speech_thread.join()
        if hasattr(self, 'engine'):
            self.engine.stop()
            del self.engine

class SpeechManager:
    """Manager class for handling TTS in the presentation context"""
    def __init__(self, config: AnalyzerConfig):
        self.tts = TextToSpeech(config)
        self.speech_settings = {
            'auto_speak_feedback': True,
            'interrupt_priority': True,
            'min_confidence': 0.7
        }
    
    def process_feedback(self, feedback_text: str, confidence: float = 1.0,
                        priority: bool = False):
        """Process feedback for speech synthesis"""
        if not self.speech_settings['auto_speak_feedback']:
            return
            
        if confidence >= self.speech_settings['min_confidence']:
            self.tts.speak(feedback_text, 
                          priority=priority and self.speech_settings['interrupt_priority'])
    
    def update_settings(self, settings: Dict):
        """Update speech settings"""
        self.speech_settings.update(settings)
    
    def toggle_auto_feedback(self):
        """Toggle automatic feedback speech"""
        self.speech_settings['auto_speak_feedback'] = not self.speech_settings['auto_speak_feedback']
        return self.speech_settings['auto_speak_feedback']

