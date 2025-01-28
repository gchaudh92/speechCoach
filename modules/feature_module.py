from abc import ABC, abstractmethod
import cv2
import numpy as np
import json
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
import threading
import queue

@dataclass
class AnalyzerConfig:
    """Configuration for feature modules"""
    enabled: bool = True
    sensitivity: float = 1.0
    threshold: float = 0.5
    update_interval: float = 1.0

class FeatureModule(ABC):
    """Base class for all feature modules"""
    def __init__(self, config: AnalyzerConfig):
        self.config = config
        self.enabled = config.enabled
        self.last_update = 0
        
    def toggle(self) -> bool:
        """Toggle module on/off"""
        self.enabled = not self.enabled
        return self.enabled
    
    @abstractmethod
    def process(self, frame, audio_data, metrics: dict) -> dict:
        """Process incoming data and return metrics"""
        pass
    
    @abstractmethod
    def get_feedback(self) -> List[str]:
        """Get current feedback messages"""
        pass

class PresentationAnalysisSystem:
    def __init__(self):
        # Load configuration
        self.config = self._load_config()
        
        # Initialize module registry
        self.modules = {}
        self.feedback_queue = queue.Queue()
        
        # Initialize core analyzers
        self.initialize_core_modules()
        
        # Initialize feature modules
        self.initialize_feature_modules()
        
        # UI state
        self.show_metrics = True
        self.current_mode = "presentation"  # presentation, interview, pitch
        
    def _load_config(self) -> Dict[str, AnalyzerConfig]:
        """Load module configurations"""
        try:
            with open('analyzer_config.json', 'r') as f:
                config_data = json.load(f)
            
            configs = {}
            for module, settings in config_data.items():
                configs[module] = AnalyzerConfig(**settings)
            return configs
        except FileNotFoundError:
            # Return default configurations
            return {
                'real_time_feedback': AnalyzerConfig(),
                'behavioral_analysis': AnalyzerConfig(),
                'content_analysis': AnalyzerConfig(),
                'audience_engagement': AnalyzerConfig(),
                'analytics': AnalyzerConfig(),
                'emotional_intelligence': AnalyzerConfig(),
                'environmental': AnalyzerConfig(),
                'accessibility': AnalyzerConfig()
            }

    def initialize_core_modules(self):
        """Initialize core analysis modules"""
        self.video_processor = VideoProcessor()
        self.audio_processor = AudioProcessor()
        self.data_manager = DataManager()
        
    def initialize_feature_modules(self):
        """Initialize all feature modules"""
        # Real-time feedback modules
        self.modules['timer'] = PresentationTimer(self.config['real_time_feedback'])
        self.modules['confidence_tracker'] = ConfidenceTracker(self.config['real_time_feedback'])
        self.modules['voice_analysis'] = VoiceAnalyzer(self.config['real_time_feedback'])
        
        # Behavioral analysis modules
        self.modules['gesture_analyzer'] = GestureAnalyzer(self.config['behavioral_analysis'])
        self.modules['eye_contact'] = EyeContactTracker(self.config['behavioral_analysis'])
        self.modules['movement_tracker'] = MovementTracker(self.config['behavioral_analysis'])
        
        # Content analysis modules
        self.modules['speech_analyzer'] = SpeechAnalyzer(self.config['content_analysis'])
        self.modules['pacing_guide'] = PacingGuide(self.config['content_analysis'])
        self.modules['keyword_tracker'] = KeywordTracker(self.config['content_analysis'])
        
        # Additional feature modules
        self.modules['engagement_tracker'] = EngagementTracker(self.config['audience_engagement'])
        self.modules['emotion_analyzer'] = EmotionAnalyzer(self.config['emotional_intelligence'])
        self.modules['environment_monitor'] = EnvironmentMonitor(self.config['environmental'])
        
    def toggle_module(self, module_name: str) -> bool:
        """Toggle specific module on/off"""
        if module_name in self.modules:
            return self.modules[module_name].toggle()
        return False
    
    def set_mode(self, mode: str):
        """Set presentation mode and adjust modules accordingly"""
        self.current_mode = mode
        # Adjust module configurations based on mode
        if mode == "interview":
            self.modules['eye_contact'].config.sensitivity = 1.5
            self.modules['gesture_analyzer'].config.threshold = 0.3
        elif mode == "pitch":
            self.modules['engagement_tracker'].config.sensitivity = 1.2
            self.modules['pacing_guide'].config.threshold = 0.4
            
    def process_frame(self, frame, audio_data):
        """Process a single frame with all enabled modules"""
        metrics = {}
        
        # Process with core modules
        video_metrics = self.video_processor.process(frame)
        audio_metrics = self.audio_processor.process(audio_data)
        metrics.update(video_metrics)
        metrics.update(audio_metrics)
        
        # Process with feature modules
        for module in self.modules.values():
            if module.enabled:
                try:
                    module_metrics = module.process(frame, audio_data, metrics)
                    metrics.update(module_metrics)
                except Exception as e:
                    print(f"Error in module {module.__class__.__name__}: {str(e)}")
        
        # Store metrics
        self.data_manager.store_metrics(metrics)
        
        return metrics
    
    def get_feedback(self) -> List[str]:
        """Get consolidated feedback from all enabled modules"""
        feedback = []
        for module in self.modules.values():
            if module.enabled:
                feedback.extend(module.get_feedback())
        return feedback
    
    def draw_ui(self, frame, metrics):
        """Draw UI elements on frame"""
        if not self.show_metrics:
            return frame
            
        # Create overlay
        overlay = frame.copy()
        
        # Draw module-specific visualizations
        for module in self.modules.values():
            if module.enabled and hasattr(module, 'draw'):
                overlay = module.draw(overlay, metrics)
        
        # Add general metrics display
        self._draw_metrics_panel(overlay, metrics)
        
        # Blend overlay with original frame
        return cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
    
    def _draw_metrics_panel(self, frame, metrics):
        """Draw metrics panel on frame"""
        h, w = frame.shape[:2]
        
        # Draw background panel
        cv2.rectangle(frame, (w-300, 0), (w, h), (0, 0, 0), -1)
        
        # Draw metrics
        y_offset = 30
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                text = f"{key}: {value:.1f}"
            else:
                text = f"{key}: {value}"
            cv2.putText(frame, text, (w-290, y_offset), font, 0.6, (255, 255, 255), 1)
            y_offset += 25
    
    def save_config(self):
        """Save current configuration"""
        config_data = {name: {
            'enabled': module.enabled,
            'sensitivity': module.config.sensitivity,
            'threshold': module.config.threshold,
            'update_interval': module.config.update_interval
        } for name, module in self.modules.items()}
        
        with open('analyzer_config.json', 'w') as f:
            json.dump(config_data, f, indent=4)
    
    def get_module_status(self) -> Dict[str, bool]:
        """Get enabled/disabled status of all modules"""
        return {name: module.enabled for name, module in self.modules.items()}
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive analysis report"""
        return self.data_manager.generate_report(self.modules)

class KeyboardController:
    """Handle keyboard controls for the system"""
    def __init__(self, analysis_system: PresentationAnalysisSystem):
        self.system = analysis_system
        self.controls = {
            'q': ('Quit', self._quit),
            'm': ('Toggle Metrics', self._toggle_metrics),
            'p': ('Pause/Resume', self._toggle_pause),
            's': ('Save Screenshot', self._save_screenshot),
            'r': ('Toggle Recording', self._toggle_recording),
            'h': ('Show Help', self._show_help)
        }
        # Add module toggle controls
        for i, module_name in enumerate(self.system.modules.keys()):
            key = str(i + 1)
            self.controls[key] = (f'Toggle {module_name}', 
                                lambda m=module_name: self._toggle_module(m))
    
    def handle_key(self, key: str) -> bool:
        """Handle keyboard input"""
        if key in self.controls:
            self.controls[key][1]()
            return True
        return False
    
    def _toggle_module(self, module_name: str):
        status = self.system.toggle_module(module_name)
        print(f"Module {module_name} {'enabled' if status else 'disabled'}")
    
    def _toggle_metrics(self):
        self.system.show_metrics = not self.system.show_metrics
    
    def _toggle_pause(self):
        # Implement pause functionality
        pass
    
    def _save_screenshot(self):
        # Implement screenshot saving
        pass
    
    def _toggle_recording(self):
        # Implement recording toggle
        pass
    
    def _show_help(self):
        print("\nAvailable Controls:")
        for key, (description, _) in self.controls.items():
            print(f"{key}: {description}")
    
    def _quit(self):
        self.system.save_config()
        return True

def main():
    # Initialize the system
    analysis_system = PresentationAnalysisSystem()
    controller = KeyboardController(analysis_system)
    
    # Start capture
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Process frame
        metrics = analysis_system.process_frame(frame, None)  # Add audio data when available
        
        # Draw UI
        display_frame = analysis_system.draw_ui(frame, metrics)
        
        # Show frame
        cv2.imshow('Presentation Analysis', display_frame)
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if controller.handle_key(chr(key)):
            break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()