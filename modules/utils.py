from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import json
import time
import cv2
import numpy as np
from collections import deque
import threading
import queue
import os

@dataclass
class AnalyzerConfig:
    """Configuration settings for analysis modules"""
    # General settings
    enabled: bool = True
    sensitivity: float = 1.0
    threshold: float = 0.5
    update_interval: float = 1.0
    
    # Detection settings
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    
    # Analysis settings
    feedback_level: str = "detailed"  # basic, detailed, expert
    alert_cooldown: float = 3.0  # seconds between similar alerts
    history_size: int = 30  # number of frames to keep in history
    
    # Performance settings
    processing_priority: str = "balanced"  # speed, balanced, accuracy
    
    @classmethod
    def from_dict(cls, config_dict: Dict) -> 'AnalyzerConfig':
        """Create config from dictionary"""
        return cls(**config_dict)
    
    def to_dict(self) -> Dict:
        """Convert config to dictionary"""
        return {
            'enabled': self.enabled,
            'sensitivity': self.sensitivity,
            'threshold': self.threshold,
            'update_interval': self.update_interval,
            'min_detection_confidence': self.min_detection_confidence,
            'min_tracking_confidence': self.min_tracking_confidence,
            'feedback_level': self.feedback_level,
            'alert_cooldown': self.alert_cooldown,
            'history_size': self.history_size,
            'processing_priority': self.processing_priority
        }

class FeatureModule(ABC):
    """Base class for all feature modules"""
    def __init__(self, config: AnalyzerConfig):
        self.config = config
        self.enabled = config.enabled
        self.last_update = 0
        self.metrics_history = deque(maxlen=config.history_size)
        self.alert_history = {}
        self._lock = threading.Lock()
        
    def toggle(self) -> bool:
        """Toggle module on/off"""
        with self._lock:
            self.enabled = not self.enabled
            return self.enabled
    
    @abstractmethod
    def process(self, frame: np.ndarray, audio_data: Optional[np.ndarray], 
                metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Process incoming data and return metrics"""
        pass
    
    def get_feedback(self, priority: str = "all") -> List[str]:
        """Get current feedback messages"""
        return []
    
    def _check_alert_cooldown(self, alert_type: str) -> bool:
        """Check if enough time has passed to show alert again"""
        current_time = time.time()
        if alert_type not in self.alert_history:
            self.alert_history[alert_type] = current_time
            return True
            
        time_since_last = current_time - self.alert_history[alert_type]
        if time_since_last >= self.config.alert_cooldown:
            self.alert_history[alert_type] = current_time
            return True
            
        return False
    
    def _update_metrics_history(self, metrics: Dict[str, Any]):
        """Update metrics history"""
        self.metrics_history.append(metrics)
    
    def get_average_metric(self, metric_name: str, window_size: int = None) -> Optional[float]:
        """Calculate average of a metric over recent history"""
        if not self.metrics_history:
            return None
            
        if window_size is None:
            window_size = len(self.metrics_history)
            
        recent_metrics = list(self.metrics_history)[-window_size:]
        values = [m.get(metric_name) for m in recent_metrics if metric_name in m]
        
        if not values:
            return None
            
        return sum(values) / len(values)

class DataManager:
    """Manages data storage and retrieval for analysis modules"""
    def __init__(self, storage_dir: str = "data"):
        self.storage_dir = storage_dir
        self.current_session = str(int(time.time()))
        self.metrics_queue = queue.Queue()
        self.feedback_history = deque(maxlen=1000)
        
        # Create storage directory if it doesn't exist
        os.makedirs(storage_dir, exist_ok=True)
        os.makedirs(os.path.join(storage_dir, "sessions"), exist_ok=True)
        
    def store_metrics(self, metrics: Dict[str, Any]):
        """Store metrics for current session"""
        metrics['timestamp'] = time.time()
        self.metrics_queue.put(metrics)
        
        # Process queue periodically
        if self.metrics_queue.qsize() > 100:
            self._process_metrics_queue()
    
    def store_feedback(self, feedback: str):
        """Store feedback message"""
        self.feedback_history.append({
            'timestamp': time.time(),
            'message': feedback
        })
    
    def _process_metrics_queue(self):
        """Process and save queued metrics"""
        metrics_list = []
        while not self.metrics_queue.empty():
            metrics_list.append(self.metrics_queue.get())
            
        if metrics_list:
            self._save_metrics(metrics_list)
    
    def _save_metrics(self, metrics_list: List[Dict[str, Any]]):
        """Save metrics to file"""
        session_file = os.path.join(
            self.storage_dir, 
            "sessions", 
            f"session_{self.current_session}.json"
        )
        
        try:
            # Load existing data if file exists
            if os.path.exists(session_file):
                with open(session_file, 'r') as f:
                    existing_data = json.load(f)
            else:
                existing_data = []
                
            # Append new metrics
            existing_data.extend(metrics_list)
            
            # Save updated data
            with open(session_file, 'w') as f:
                json.dump(existing_data, f, indent=2)
                
        except Exception as e:
            print(f"Error saving metrics: {str(e)}")
    
    def get_session_metrics(self, session_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get metrics for specific session"""
        if session_id is None:
            session_id = self.current_session
            
        session_file = os.path.join(
            self.storage_dir, 
            "sessions", 
            f"session_{session_id}.json"
        )
        
        try:
            if os.path.exists(session_file):
                with open(session_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            print(f"Error loading session metrics: {str(e)}")
            
        return []
    
    def generate_session_report(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Generate analysis report for session"""
        metrics = self.get_session_metrics(session_id)
        
        if not metrics:
            return {}
            
        # Calculate session statistics
        session_duration = metrics[-1]['timestamp'] - metrics[0]['timestamp']
        
        report = {
            'session_id': session_id or self.current_session,
            'duration': session_duration,
            'start_time': metrics[0]['timestamp'],
            'end_time': metrics[-1]['timestamp'],
            'metrics_count': len(metrics),
            'summary': self._generate_metrics_summary(metrics),
            'feedback_history': list(self.feedback_history)
        }
        
        return report
    
    def _generate_metrics_summary(self, metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate summary statistics for metrics"""
        summary = {}
        
        # Get all metric keys
        metric_keys = set()
        for m in metrics:
            metric_keys.update(m.keys())
            
        # Calculate statistics for each metric
        for key in metric_keys:
            if key == 'timestamp':
                continue
                
            values = [m[key] for m in metrics if key in m]
            if not values:
                continue
                
            # Calculate statistics if values are numeric
            if all(isinstance(v, (int, float)) for v in values):
                summary[key] = {
                    'min': min(values),
                    'max': max(values),
                    'average': sum(values) / len(values),
                    'std': np.std(values) if len(values) > 1 else 0
                }
            else:
                # For non-numeric values, count occurrences
                value_counts = {}
                for v in values:
                    value_counts[str(v)] = value_counts.get(str(v), 0) + 1
                summary[key] = value_counts
                
        return summary

# Helper functions
def calculate_moving_average(values: List[float], window_size: int) -> List[float]:
    """Calculate moving average of values"""
    if not values or window_size <= 0:
        return []
        
    result = []
    for i in range(len(values)):
        start_idx = max(0, i - window_size + 1)
        window = values[start_idx:i + 1]
        average = sum(window) / len(window)
        result.append(average)
        
    return result

def normalize_value(value: float, min_val: float, max_val: float) -> float:
    """Normalize value to range [0, 1]"""
    if max_val == min_val:
        return 0.5
    return (value - min_val) / (max_val - min_val)

def draw_progress_bar(frame: np.ndarray, value: float, position: tuple, 
                     size: tuple, color: tuple = (0, 255, 0)) -> np.ndarray:
    """Draw progress bar on frame"""
    x, y = position
    width, height = size
    
    # Draw background
    cv2.rectangle(frame, (x, y), (x + width, y + height), (255, 255, 255), 1)
    
    # Draw filled portion
    fill_width = int(width * max(0, min(1, value)))
    if fill_width > 0:
        cv2.rectangle(frame, (x, y), (x + fill_width, y + height), color, -1)
        
    return frame

def draw_metric_text(frame: np.ndarray, text: str, position: tuple,
                    font_scale: float = 0.7, color: tuple = (255, 255, 255),
                    thickness: int = 2) -> np.ndarray:
    """Draw metric text on frame"""
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, text, position, font, font_scale, color, thickness)
    return frame