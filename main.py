import cv2
import time
import os
import subprocess
from audio_analyzer import AudioAnalyzer
from video_analyzer import VideoAnalyzer
from presentation_coach import PresentationCoach, CoachingDisplay

class PresentationAnalyzer:
    def __init__(self):
        print("Initializing analyzers...")
        self.video_analyzer = VideoAnalyzer()
        self.audio_analyzer = AudioAnalyzer()
        self.coach = PresentationCoach()
        self.display = CoachingDisplay()
        self.coaching_enabled = True  # Flag to control coaching feedback
        print("Analyzers initialized")
        
    def toggle_coaching(self):
        """Toggle the coaching feedback on and off."""
        self.coaching_enabled = not self.coaching_enabled
        status = "ON" if self.coaching_enabled else "OFF"
        print(f"Coaching feedback is now {status}")

    def add_feedback_to_frame(self, frame, speech_stats, speech_feedback, video_metrics, video_feedback):
        """Add all feedback to the frame in a clear, organized way"""
        h, w = frame.shape[:2]
        
        # Left side: Video metrics and feedback
        y_offset = 30
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Draw video metrics
        metrics_text = [
            f"Head Tilt: {video_metrics['head_tilt']:.1f}°",
            f"Head Movement: {video_metrics['head_movement']:.3f}",
            f"Expression: {video_metrics['expression']}",
            f"Eyes: {video_metrics['eye_state']}",
            f"Mouth: {video_metrics['mouth_state']}"
        ]
        
        # Draw each metric
        for text in metrics_text:
            cv2.putText(frame, text, (10, y_offset),
                       font, 0.6, (0, 255, 0), 2)
            y_offset += 25
        
        # Draw video feedback
        y_offset += 150
        cv2.putText(frame, "Face Analysis:", (10, y_offset),
                   font, 1.2, (0, 0, 255), 2)
        y_offset += 25
        
        for text in video_feedback:
            cv2.putText(frame, text, (10, y_offset),
                       font, 1.2, (0, 0, 255), 2)
            y_offset += 55

        # Right side: Audio metrics and feedback
        x_offset = w - 400  # Increased offset for longer text
        y_offset = 30
        
        # Display speech stats header
        cv2.putText(frame, "Speech Analysis:", (x_offset, y_offset),
                   font, 1.2, (255, 0, 0), 2)
        y_offset += 30
        
        # Format and display speech stats
        formatted_stats = [
            f"Volume: {speech_stats['volume_level']:.2f}",
            f"Speech Rate: {speech_stats['speech_rate']:.1f} wpm",
            f"Duration: {speech_stats['speaking_duration']:.1f}s",
            f"Pauses: {speech_stats['pause_count']}",
            f"Pitch Var: {speech_stats['pitch_variation']:.2f}"
        ]
        
        for stat in formatted_stats:
            cv2.putText(frame, stat, (x_offset, y_offset),
                       font, 0.6, (255, 0, 0), 2)
            y_offset += 25
        
        # Add speech feedback
        y_offset += 10
        cv2.putText(frame, "Speech Feedback:", (x_offset, y_offset),
                   font, 0.7, (255, 0, 0), 2)
        y_offset += 25
        
        # Split and display feedback
        if speech_feedback:
            feedback_parts = speech_feedback.split(" | ")
            for part in feedback_parts:
                cv2.putText(frame, part, (x_offset, y_offset),
                           font, 0.6, (255, 0, 0), 2)
                y_offset += 25

        return frame

    def start_analysis(self, video_output="output_video.mp4", audio_output="temp_audio.wav"):
        print("Initializing video capture...")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise RuntimeError("Could not access webcam")

        # Get frame dimensions
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Initialize video writer
        temp_video = "temp_video.avi"
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        fps = 30.0
        
        out = cv2.VideoWriter(
            temp_video, 
            fourcc, 
            fps, 
            (frame_width, frame_height)
        )

        try:
            print("Starting audio recording...")
            self.audio_analyzer.start_recording()
            
            start_time = time.time()
            frames_recorded = 0
            recording = True
            
            print("\nRecording started...")
            print("Press 'q' to stop recording")
            print("Press 'c' to toggle coaching feedback")
            print("=====================================")
            
            while recording:
                ret, frame = cap.read()
                if not ret:
                    print("Error: Failed to capture frame")
                    break

                try:
                    # Analyze video frame
                    video_metrics, video_feedback, annotated_frame = self.video_analyzer.analyze_frame(frame)
                    
                    # Get audio analysis
                    speech_feedback = self.audio_analyzer.get_speech_feedback()
                    speech_stats = self.audio_analyzer.get_speech_stats()

                    # Combine video and audio metrics for coaching
                    combined_metrics = {
                        **video_metrics,
                        'speech_rate': speech_stats['speech_rate'],
                        'volume_level': speech_stats['volume_level'],
                        'expression': video_metrics['expression']
                    }

                    # Generate coaching feedback if coaching is enabled
                    if self.coaching_enabled:
                        coaching_feedback = self.coach.analyze_performance(
                            video_metrics, 
                            speech_stats,
                            time.time()
                        )
                        
                        # Add coaching display
                        annotated_frame = self.display.add_coaching_feedback(
                            annotated_frame,
                            coaching_feedback,
                            combined_metrics
                        )
                        
                        # Add performance indicators
                        self.display.add_performance_indicators(annotated_frame, combined_metrics)

                    out.write(annotated_frame)
                    frames_recorded += 1
                    
                    # Display frame
                    cv2.imshow('Analysis (Press q to stop)', annotated_frame)
                    
                    # Check for key presses
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):  # Stop recording
                        print("\nStopping recording...")
                        recording = False
                    elif key == ord('c'):  # Toggle coaching feedback
                        self.toggle_coaching()
                    
                except Exception as e:
                    print(f"Error processing frame: {str(e)}")
                    continue

            recording_duration = time.time() - start_time
            print(f"Recorded {frames_recorded} frames over {recording_duration:.1f} seconds")

        except Exception as e:
            print(f"Error during recording: {str(e)}")
        
        finally:
            print("Cleaning up...")
            print("Stopping audio recording...")
            try:
                self.audio_analyzer.stop_recording(audio_output)
                print("Audio saved successfully")
            except Exception as e:
                print(f"Error saving audio: {str(e)}")

            print("Releasing video resources...")
            try:
                cap.release()
                out.release()
                cv2.destroyAllWindows()
                print("Video resources released")
            except Exception as e:
                print(f"Error releasing video resources: {str(e)}")

        # Combine video and audio
        try:
            print("\nProcessing final video...")
            subprocess.run([
                'ffmpeg', '-y',
                '-i', temp_video,
                '-i', audio_output,
                '-c:v', 'libx264',
                '-preset', 'medium',
                '-crf', '23',
                '-c:a', 'aac',
                '-b:a', '320k',
                '-ar', '44100',
                '-ac', '2',
                video_output
            ], check=True, capture_output=True)
            
            # Clean up temporary files
            os.remove(temp_video)
            os.remove(audio_output)
            print("Video processing completed")
            
        except subprocess.CalledProcessError as e:
            print(f"Error processing video: {str(e)}")
        except Exception as e:
            print(f"Error during video processing: {str(e)}")

def main():
    try:
        analyzer = PresentationAnalyzer()
        
        print("=== Presentation Analysis Tool ===")
        print("This tool will analyze:")
        print("1. Facial expressions and head movements")
        print("2. Speech patterns and clarity")
        print("3. Overall presentation style")
        print("\nFeatures tracked:")
        print("- Head tilt and movement")
        print("- Facial expressions")
        print("- Speaking rate and clarity")
        print("- Voice volume and energy")
        print("\nPress Enter to start (or 'q' to quit)")
        
        if input().lower() == 'q':
            return
            
        analyzer.start_analysis()
        
    except Exception as e:
        print(f"Error: {str(e)}")
        raise

if __name__ == "__main__":
    main()