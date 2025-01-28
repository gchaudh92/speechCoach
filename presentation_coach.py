from collections import deque
import cv2

import time
import os

import mediapipe as mp

import numpy as np
import random


class PresentationCoach:
    def __init__(self):
        self.thresholds = {
            'speech_rate': {'min': 120, 'max': 160},
            'volume': {'min': 0.1, 'max': 0.8},
            'head_movement': {'max': 0.15},
            'eye_contact': {'min': 0.7},
            'pause_duration': {'max': 3.0},
            'energy_level': {'min': 0.4}
        }
        
        self.feedback_history = deque(maxlen=150)
        self.current_alerts = set()
        self.alert_cooldown = 3.0
        self.last_alert_times = {}

    def _check_alert_cooldown(self, alert_type, current_time=None):
        """Check if enough time has passed to show the alert again."""
        if current_time is None:
            current_time = time.time()
        
        if alert_type not in self.last_alert_times:
            return True
        
        time_since_last = current_time - self.last_alert_times[alert_type]
        return time_since_last >= self.alert_cooldown

    def analyze_performance(self, video_metrics, audio_metrics, timestamp):
        """Analyze current performance and generate coaching feedback."""
        current_feedback = []
        alerts = set()
        
        # Speech rate coaching
        if 'speech_rate' in audio_metrics:
            rate = audio_metrics['speech_rate']
            if rate > self.thresholds['speech_rate']['max']:
                alerts.add(('speech_rate_high', "Slow down your speech"))
            elif rate < self.thresholds['speech_rate']['min'] and rate > 0:
                alerts.add(('speech_rate_low', "Try speaking a bit faster"))
        
        # Volume coaching
        if 'volume_level' in audio_metrics:
            volume = audio_metrics['volume_level']
            if volume > self.thresholds['volume']['max']:
                alerts.add(('volume_high', "Lower your voice slightly"))
            elif volume < self.thresholds['volume']['min']:
                alerts.add(('volume_low', "Speak up a bit"))
        
        # Head movement coaching
        if 'head_pose' in video_metrics:
            pose = video_metrics['head_pose']
            if abs(pose['pitch']) > 20:
                alerts.add(('head_pitch', "Keep your head level"))
            if abs(pose['yaw']) > 30:
                alerts.add(('head_yaw', "Face the audience"))
            if abs(pose['roll']) > 15:
                alerts.add(('head_roll', "Straighten your head"))
        
        # Filter alerts based on cooldown
        current_time = timestamp
        active_alerts = set()
        for alert_type, message in alerts:
            if self._check_alert_cooldown(alert_type, current_time):
                active_alerts.add((alert_type, message))
                self.last_alert_times[alert_type] = current_time
        
        return self._format_coaching_feedback(active_alerts)

    def _format_coaching_feedback(self, alerts):
        """Format coaching alerts into display-ready feedback."""
        feedback = []
        for _, message in alerts:
            feedback.append(message)
        
        if not alerts:
            feedback.append("Great job! Keep it up!")
        
        return feedback


class CoachingDisplay:
    def __init__(self, window_name="Presentation Coach"):
        self.window_name = window_name
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.colors = {
            'alert': (0, 0, 255),     # Red
            'positive': (0, 255, 0),   # Green
            'info': (255, 165, 0),     # Orange
            'background': (0, 0, 0)    # Black
        }
    
    def add_coaching_feedback(self, frame, feedback, metrics):
        """Add coaching feedback to the frame."""
        h, w = frame.shape[:2]
        
        # Create a semi-transparent overlay for feedback
        overlay = frame.copy()
        
        # Draw coaching panel background
        panel_height = 150
        cv2.rectangle(overlay, (0, h-panel_height), (w, h), 
                     self.colors['background'], -1)
        
        # Add feedback messages
        y_offset = h - panel_height + 30
        for i, message in enumerate(feedback):
            color = self._get_message_color(message)
            cv2.putText(overlay, message, (20, y_offset + i*30),
                       self.font, 0.7, color, 2)
        
        # Add key metrics
        metrics_start_x = w - 250
        y_offset = h - panel_height + 30
        self._add_metrics_display(overlay, metrics, metrics_start_x, y_offset)
        
        # Combine overlay with original frame
        alpha = 0.7
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        return frame
    
    def _get_message_color(self, message):
        """Determine color based on message content."""
        if any(word in message.lower() for word in ['good', 'great', 'nice', 'excellent']):
            return self.colors['positive']
        elif any(word in message.lower() for word in ['try', 'improve', 'should']):
            return self.colors['alert']
        return self.colors['info']
    
    def _add_metrics_display(self, frame, metrics, x_start, y_start):
        """Add performance metrics display."""
        if 'speech_rate' in metrics:
            cv2.putText(frame, f"Speech Rate: {metrics['speech_rate']:.1f} wpm",
                       (x_start, y_start), self.font, 0.6, self.colors['info'], 2)
        
        if 'volume_level' in metrics:
            cv2.putText(frame, f"Volume: {metrics['volume_level']:.2f}",
                       (x_start, y_start + 30), self.font, 0.6, self.colors['info'], 2)
        
        if 'expression' in metrics:
            cv2.putText(frame, f"Expression: {metrics['expression']}",
                       (x_start, y_start + 60), self.font, 0.6, self.colors['info'], 2)
    
    def add_performance_indicators(self, frame, metrics):
        """Add visual indicators for key performance metrics."""
        h, w = frame.shape[:2]
        
        # Add head pose indicator
        if 'head_pose' in metrics:
            self._draw_head_pose_indicator(frame, metrics['head_pose'], 
                                         (w-150, 100), 50)
        
        # Add speech rate indicator
        if 'speech_rate' in metrics:
            self._draw_gauge(frame, metrics['speech_rate'], 
                           (w-150, 200), "Speech Rate", 0, 200)
        
        # Add volume indicator
        if 'volume_level' in metrics:
            self._draw_gauge(frame, metrics['volume_level']*100, 
                           (w-150, 300), "Volume", 0, 100)
    
    def _draw_head_pose_indicator(self, frame, pose, center, radius):
        """Draw a visual indicator for head pose."""
        # Draw circular guide
        cv2.circle(frame, center, radius, self.colors['info'], 1)
        
        # Calculate head position indicator
        x = int(center[0] + radius * np.sin(np.radians(pose['yaw'])))
        y = int(center[1] + radius * np.sin(np.radians(pose['pitch'])))
        
        # Draw position indicator
        cv2.circle(frame, (x, y), 5, self.colors['alert'], -1)
        
        # Draw guide lines
        cv2.line(frame, center, (x, y), self.colors['info'], 1)
    
    def _draw_gauge(self, frame, value, center, label, min_val, max_val):
        """Draw a gauge indicator for a metric."""
        radius = 40
        start_angle = 180
        end_angle = 360
        
        # Calculate position on gauge
        normalized_value = (value - min_val) / (max_val - min_val)
        angle = start_angle + normalized_value * (end_angle - start_angle)
        
        # Draw gauge background
        cv2.ellipse(frame, center, (radius, radius), 0, 
                   start_angle, end_angle, self.colors['info'], 1)
        
        # Draw current value indicator
        x = int(center[0] + radius * np.cos(np.radians(angle)))
        y = int(center[1] + radius * np.sin(np.radians(angle)))
        cv2.line(frame, center, (x, y), self.colors['alert'], 2)
        
        # Add label
        cv2.putText(frame, f"{label}: {value:.1f}", 
                   (center[0]-radius, center[1]+radius+20),
                   self.font, 0.5, self.colors['info'], 1)