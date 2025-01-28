import numpy as np
from collections import deque
from typing import Dict, List, Optional
import time
from .utils import FeatureModule, AnalyzerConfig

class SpeechAnalyzer(FeatureModule):
    def __init__(self, config: AnalyzerConfig):
        super().__init__(config)
        self.speech_buffer = deque(maxlen=100)
        self.filler_words = set(['um', 'uh', 'like', 'you know', 'so'])
        self.last_pause_time = 0
        self.word_count = 0
        self.pause_threshold = 0.5  # seconds
        
    def process(self, frame, audio_data, metrics: dict) -> dict:
        if not audio_data:
            return {}
            
        speech_metrics = {
            'speaking_rate': 0.0,
            'filler_word_count': 0,
            'clarity_score': 0.0,
            'pause_pattern': 'normal'
        }
        
        # Process audio data
        if self._is_speaking(audio_data):
            speech_metrics.update(self._analyze_speech(audio_data))
            
        return speech_metrics
    
    def _analyze_speech(self, audio_data) -> dict:
        """Analyze speech patterns and quality"""
        volume = self._calculate_volume(audio_data)
        clarity = self._estimate_clarity(audio_data)
        rate = self._calculate_speaking_rate(audio_data)
        
        return {
            'speaking_rate': rate,
            'clarity_score': clarity,
            'volume_level': volume,
            'speech_feedback': self._generate_feedback(rate, clarity, volume)
        }

class ContentAnalyzer(FeatureModule):
    def __init__(self, config: AnalyzerConfig):
        super().__init__(config)
        self.content_buffer = []
        self.keywords = set()
        self.topic_context = None
        self.structure_analysis = {
            'introduction': False,
            'main_points': [],
            'conclusion': False
        }
        
    def process(self, frame, audio_data, metrics: dict) -> dict:
        if 'transcribed_text' not in metrics:
            return {}
            
        text = metrics['transcribed_text']
        
        content_metrics = {
            'keyword_density': 0.0,
            'topic_relevance': 0.0,
            'structure_score': 0.0,
            'complexity_score': 0.0
        }
        
        if text:
            content_metrics.update(self._analyze_content(text))
            
        return content_metrics
    
    def _analyze_content(self, text: str) -> dict:
        """Analyze speech content and structure"""
        self.content_buffer.append(text)
        
        keywords = self._extract_keywords(text)
        relevance = self._calculate_relevance(text)
        structure = self._analyze_structure(self.content_buffer)
        complexity = self._analyze_complexity(text)
        
        return {
            'keyword_density': len(keywords) / len(text.split()),
            'topic_relevance': relevance,
            'structure_score': structure,
            'complexity_score': complexity,
            'content_feedback': self._generate_feedback(keywords, relevance, complexity)
        }

class KeywordTracker(FeatureModule):
    def __init__(self, config: AnalyzerConfig):
        super().__init__(config)
        self.keyword_history = {}
        self.target_keywords = set()
        self.keyword_frequency = {}
        
    def process(self, frame, audio_data, metrics: dict) -> dict:
        if 'transcribed_text' not in metrics:
            return {}
            
        text = metrics['transcribed_text']
        return self._analyze_keywords(text)
    
    def _analyze_keywords(self, text: str) -> dict:
        """Track keyword usage and patterns"""
        words = text.lower().split()
        current_keywords = self._extract_keywords(words)
        
        # Update keyword frequency
        for keyword in current_keywords:
            self.keyword_frequency[keyword] = self.keyword_frequency.get(keyword, 0) + 1
            
        return {
            'keywords_used': list(current_keywords),
            'keyword_frequency': self.keyword_frequency,
            'keyword_variety': len(self.keyword_frequency) / max(1, len(words))
        }

class PacingGuide(FeatureModule):
    def __init__(self, config: AnalyzerConfig):
        super().__init__(config)
        self.start_time = None
        self.section_times = {
            'introduction': 0.2,    # 20% of total time
            'main_content': 0.6,    # 60% of total time
            'conclusion': 0.2       # 20% of total time
        }
        self.current_section = 'introduction'
        
    def process(self, frame, audio_data, metrics: dict) -> dict:
        if not self.start_time:
            self.start_time = time.time()
            
        elapsed = time.time() - self.start_time
        section = self._determine_section(elapsed)
        pacing = self._analyze_pacing(elapsed, section)
        
        return {
            'current_section': section,
            'pacing_score': pacing,
            'time_remaining': self._calculate_remaining_time(elapsed),
            'pacing_feedback': self._generate_pacing_feedback(pacing)
        }

class ContentFeedbackManager:
    def __init__(self, config: AnalyzerConfig):
        self.speech_analyzer = SpeechAnalyzer(config)
        self.content_analyzer = ContentAnalyzer(config)
        self.keyword_tracker = KeywordTracker(config)
        self.pacing_guide = PacingGuide(config)
        
    def process_frame(self, frame, audio_data, metrics: dict) -> tuple:
        """Process frame through all content analyzers"""
        # Analyze speech
        speech_metrics = self.speech_analyzer.process(frame, audio_data, metrics)
        
        # Analyze content
        content_metrics = self.content_analyzer.process(frame, audio_data, metrics)
        
        # Track keywords
        keyword_metrics = self.keyword_tracker.process(frame, audio_data, metrics)
        
        # Monitor pacing
        pacing_metrics = self.pacing_guide.process(frame, audio_data, metrics)
        
        # Combine metrics
        combined_metrics = {
            **speech_metrics,
            **content_metrics,
            **keyword_metrics,
            **pacing_metrics
        }
        
        # Generate feedback
        feedback = self._generate_feedback(combined_metrics)
        
        return combined_metrics, feedback
    
    def _generate_feedback(self, metrics: dict) -> List[str]:
        """Generate content-related feedback"""
        feedback = []
        
        # Speech clarity feedback
        if metrics.get('clarity_score', 1.0) < 0.7:
            feedback.append("Try to speak more clearly")
        
        # Keyword usage feedback
        if metrics.get('keyword_variety', 0) < 0.1:
            feedback.append("Try to use more varied vocabulary")
        
        # Pacing feedback
        if metrics.get('pacing_score', 1.0) < 0.6:
            feedback.append(metrics.get('pacing_feedback', 'Adjust your pacing'))
        
        # Content structure feedback
        if metrics.get('structure_score', 1.0) < 0.5:
            feedback.append("Consider improving your presentation structure")
        
        return feedback