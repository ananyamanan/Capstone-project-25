import os
import cv2
import torch
import torch.nn as nn
import numpy as np
import subprocess
import json
from collections import deque, defaultdict
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn.functional as F
from pathlib import Path
import time
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SlowFastNetwork(nn.Module):
    """Custom SlowFast Network implementation"""
    def __init__(self, num_classes=40):
        super(SlowFastNetwork, self).__init__()
        
        # Slow pathway - ResNet-like architecture for spatial features
        self.slow_pathway = self._make_slow_pathway()
        
        # Fast pathway - Lighter network for temporal features
        self.fast_pathway = self._make_fast_pathway()
        
        # Fusion layer
        self.fusion = nn.AdaptiveAvgPool3d((1, 1, 1))
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512 + 128, 256),  # 512 from slow + 128 from fast
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
    def _make_slow_pathway(self):
        """Create slow pathway for spatial features"""
        # Use ResNet-18 backbone modified for 3D
        resnet = models.resnet18(pretrained=True)
        
        # Convert 2D conv to 3D conv for temporal dimension
        slow_layers = []
        
        # Initial conv layer
        slow_layers.append(nn.Conv3d(3, 64, kernel_size=(1, 7, 7), 
                                   stride=(1, 2, 2), padding=(0, 3, 3)))
        slow_layers.append(nn.BatchNorm3d(64))
        slow_layers.append(nn.ReLU(inplace=True))
        slow_layers.append(nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)))
        
        # ResNet blocks adapted for 3D
        slow_layers.extend([
            self._make_3d_block(64, 64, 2),
            self._make_3d_block(64, 128, 2),
            self._make_3d_block(128, 256, 2),
            self._make_3d_block(256, 512, 2),
            nn.AdaptiveAvgPool3d((1, 1, 1))
        ])
        
        return nn.Sequential(*slow_layers)
    
    def _make_fast_pathway(self):
        """Create fast pathway for temporal features"""
        fast_layers = []
        
        # Lighter network for temporal processing
        fast_layers.extend([
            nn.Conv3d(3, 32, kernel_size=(5, 3, 3), stride=(1, 2, 2), padding=(2, 1, 1)),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 64, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d((1, 1, 1))
        ])
        
        return nn.Sequential(*fast_layers)
    
    def _make_3d_block(self, in_channels, out_channels, num_blocks):
        """Create 3D convolution block"""
        layers = []
        for i in range(num_blocks):
            if i == 0:
                layers.append(nn.Conv3d(in_channels, out_channels, 
                                      kernel_size=(1, 3, 3), padding=(0, 1, 1)))
            else:
                layers.append(nn.Conv3d(out_channels, out_channels, 
                                      kernel_size=(1, 3, 3), padding=(0, 1, 1)))
            layers.append(nn.BatchNorm3d(out_channels))
            layers.append(nn.ReLU(inplace=True))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        """Forward pass with slow and fast pathways"""
        # x should be a list [slow_input, fast_input]
        if isinstance(x, list):
            slow_input, fast_input = x
        else:
            # If single input, use for both pathways
            slow_input = fast_input = x
        
        # Process through pathways
        slow_features = self.slow_pathway(slow_input)
        fast_features = self.fast_pathway(fast_input)
        
        # Flatten features
        slow_features = slow_features.view(slow_features.size(0), -1)
        fast_features = fast_features.view(fast_features.size(0), -1)
        
        # Concatenate features
        combined_features = torch.cat([slow_features, fast_features], dim=1)
        
        # Classify
        output = self.classifier(combined_features)
        
        return output

class SlowFastConfig:
    """Configuration class for SlowFast parameters"""
    def __init__(self):
        self.CLIP_LENGTH = 32  # Total frames for one prediction
        self.SAMPLE_RATE = 2   # Sample every nth frame
        self.SLOW_FAST_ALPHA = 4  # Ratio between slow and fast pathways
        self.INPUT_SIZE = 224  # Input image size
        self.CONFIDENCE_THRESHOLD = 0.3  # Minimum confidence for predictions
        
        # Activity classes for human activity recognition
        self.ACTIVITY_CLASSES = [
            'walking', 'running', 'jogging', 'jumping', 'sitting', 'standing',
            'waving', 'clapping', 'dancing', 'eating', 'drinking', 'talking',
            'reading', 'writing', 'sleeping', 'exercising', 'playing_sports',
            'cooking', 'cleaning', 'driving', 'cycling', 'swimming', 'climbing',
            'carrying', 'pushing', 'pulling', 'throwing', 'catching', 'kicking',
            'punching', 'hugging', 'shaking_hands', 'applauding', 'crawling',
            'falling_down', 'getting_up', 'lying_down', 'bending', 'stretching'
        ]

class VideoProcessor:
    """Handle video processing with FFmpeg integration"""
    
    @staticmethod
    def extract_video_info(video_path):
        """Extract video information using FFmpeg"""
        try:
            cmd = [
                'ffprobe', '-v', 'quiet', '-print_format', 'json',
                '-show_streams', str(video_path)
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            info = json.loads(result.stdout)
            
            video_stream = next(s for s in info['streams'] if s['codec_type'] == 'video')
            
            return {
                'fps': eval(video_stream['r_frame_rate']),
                'width': int(video_stream['width']),
                'height': int(video_stream['height']),
                'duration': float(video_stream.get('duration', 0)),
                'frames': int(video_stream.get('nb_frames', 0))
            }
        except Exception as e:
            logger.error(f"Error extracting video info: {e}")
            return None
    
    @staticmethod
    def preprocess_frame_ffmpeg(frame, target_size=(224, 224)):
        """Preprocess frame for SlowFast input"""
        # Resize frame
        frame_resized = cv2.resize(frame, target_size)
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        
        return frame_rgb

class SlowFastActivityRecognizer:
    """SlowFast Network for activity recognition"""
    
    def __init__(self, config=None, device='auto'):
        self.config = config or SlowFastConfig()
        
        # Device setup
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        logger.info(f"Using device: {self.device}")
        
        # Initialize model
        self.model = self._load_model()
        
        # Frame buffers for temporal analysis
        self.frame_buffer = deque(maxlen=self.config.CLIP_LENGTH)
        self.activity_history = deque(maxlen=10)  # Keep track of recent predictions
        
        # Preprocessing transforms
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
    def _load_model(self):
        """Load custom SlowFast model"""
        try:
            logger.info("Loading custom SlowFast Network...")
            
            # Create model
            model = SlowFastNetwork(num_classes=len(self.config.ACTIVITY_CLASSES))
            model.to(self.device)
            
            # Try to load pretrained weights if available
            model_path = "slowfast_weights.pth"
            if os.path.exists(model_path):
                logger.info(f"Loading pretrained weights from {model_path}")
                checkpoint = torch.load(model_path, map_location=self.device)
                model.load_state_dict(checkpoint)
            else:
                logger.info("No pretrained weights found, using randomly initialized model")
                logger.info("Note: For better results, train the model on activity recognition data")
            
            model.eval()
            logger.info("Model loaded successfully!")
            return model
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def _prepare_slowfast_input(self, frames):
        """Prepare input tensors for SlowFast network"""
        if len(frames) < self.config.CLIP_LENGTH:
            # Pad with last frame if insufficient frames
            while len(frames) < self.config.CLIP_LENGTH:
                frames.append(frames[-1])
        
        # Convert frames to tensors
        processed_frames = []
        for frame in frames:
            # Apply transforms
            frame_tensor = self.transform(frame.astype(np.float32) / 255.0)
            processed_frames.append(frame_tensor)
        
        # Stack frames: [T, C, H, W]
        video_tensor = torch.stack(processed_frames)
        
        # Create slow and fast pathways according to SlowFast paper
        # Slow pathway: sample every alpha frames (lower temporal resolution)
        slow_indices = torch.arange(0, len(processed_frames), self.config.SLOW_FAST_ALPHA)
        if len(slow_indices) > self.config.CLIP_LENGTH // self.config.SLOW_FAST_ALPHA:
            slow_indices = slow_indices[:self.config.CLIP_LENGTH // self.config.SLOW_FAST_ALPHA]
        slow_pathway = video_tensor[slow_indices]
        
        # Fast pathway: use all frames (higher temporal resolution)
        fast_pathway = video_tensor
        
        # Add batch dimension and transpose to [B, C, T, H, W]
        slow_pathway = slow_pathway.unsqueeze(0).transpose(1, 2).to(self.device)
        fast_pathway = fast_pathway.unsqueeze(0).transpose(1, 2).to(self.device)
        
        return [slow_pathway, fast_pathway]
    
    def predict_activity(self, frames):
        """Predict activity from frame sequence"""
        if len(frames) < 8:  # Minimum frames needed
            return "insufficient_data", 0.0, {}
        
        try:
            # Prepare input
            slow_fast_input = self._prepare_slowfast_input(frames)
            
            # Forward pass
            with torch.no_grad():
                start_time = time.time()
                outputs = self.model(slow_fast_input)
                inference_time = time.time() - start_time
                
                probabilities = F.softmax(outputs, dim=1)
                
                # Get top-k predictions
                top_k = min(5, len(self.config.ACTIVITY_CLASSES))
                top_probs, top_indices = torch.topk(probabilities, top_k, dim=1)
                
                # Get top prediction
                predicted_idx = top_indices[0][0].item() % len(self.config.ACTIVITY_CLASSES)
                predicted_class = self.config.ACTIVITY_CLASSES[predicted_idx]
                confidence = top_probs[0][0].item()
                
                # Create detailed results
                results = {
                    'inference_time': inference_time,
                    'top_predictions': [
                        {
                            'class': self.config.ACTIVITY_CLASSES[idx.item() % len(self.config.ACTIVITY_CLASSES)],
                            'confidence': prob.item()
                        }
                        for prob, idx in zip(top_probs[0], top_indices[0])
                    ]
                }
                
                return predicted_class, confidence, results
                
        except Exception as e:
            logger.error(f"Error in activity prediction: {e}")
            return "prediction_error", 0.0, {}
    
    def update_and_predict(self, frame):
        """Update frame buffer and predict if ready"""
        # Add frame to buffer
        self.frame_buffer.append(frame)
        
        # Predict when buffer is full
        if len(self.frame_buffer) == self.config.CLIP_LENGTH:
            activity, confidence, details = self.predict_activity(list(self.frame_buffer))
            
            # Update activity history for smoothing
            if confidence > self.config.CONFIDENCE_THRESHOLD:
                self.activity_history.append((activity, confidence))
            
            return activity, confidence, details
        
        return None, 0.0, {}
    
    def get_smoothed_prediction(self):
        """Get smoothed prediction based on recent history"""
        if not self.activity_history:
            return "no_activity", 0.0
        
        # Count activities in recent history
        activity_counts = defaultdict(list)
        for activity, confidence in self.activity_history:
            activity_counts[activity].append(confidence)
        
        # Get most frequent activity with highest average confidence
        best_activity = max(activity_counts.keys(), 
                          key=lambda x: len(activity_counts[x]) * np.mean(activity_counts[x]))
        avg_confidence = np.mean(activity_counts[best_activity])
        
        return best_activity, avg_confidence

class TrackedVideoAnalyzer:
    """Main class for analyzing tracked videos with SlowFast"""
    
    def __init__(self, config=None):
        self.config = config or SlowFastConfig()
        self.recognizer = SlowFastActivityRecognizer(config)
        self.video_processor = VideoProcessor()
        
    def process_tracked_videos(self, input_folder, output_folder):
        """Process all tracked videos in the folder"""
        input_path = Path(input_folder)
        output_path = Path(output_folder)
        output_path.mkdir(exist_ok=True)
        
        # Find all .avi files
        video_files = list(input_path.glob("*.avi"))
        
        if not video_files:
            logger.error(f"No .avi files found in {input_folder}")
            return
        
        logger.info(f"Found {len(video_files)} videos to process")
        
        # Process each video
        for video_file in video_files:
            output_file = output_path / f"slowfast_{video_file.name}"
            logger.info(f"Processing: {video_file.name}")
            
            try:
                self.process_single_video(video_file, output_file)
                logger.info(f"Completed: {video_file.name}")
            except Exception as e:
                logger.error(f"Error processing {video_file.name}: {e}")
    
    def process_single_video(self, input_path, output_path):
        """Process a single tracked video"""
        # Get video info
        video_info = self.video_processor.extract_video_info(input_path)
        if not video_info:
            logger.error(f"Could not extract video info for {input_path}")
            return
        
        # Open video
        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            logger.error(f"Could not open video: {input_path}")
            return
        
        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        fps = video_info['fps']
        width, height = video_info['width'], video_info['height']
        
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        # Processing variables
        frame_count = 0
        total_frames = video_info.get('frames', 0)
        current_activity = "initializing"
        current_confidence = 0.0
        processing_stats = {
            'total_predictions': 0,
            'avg_inference_time': 0,
            'activities_detected': set()
        }
        
        logger.info(f"Processing {total_frames} frames at {fps} FPS")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Preprocess frame for SlowFast
            processed_frame = self.video_processor.preprocess_frame_ffmpeg(
                frame, (self.config.INPUT_SIZE, self.config.INPUT_SIZE)
            )
            
            # Update activity recognition (process every few frames to optimize performance)
            if frame_count % 4 == 0:  # Process every 4th frame
                activity, confidence, details = self.recognizer.update_and_predict(processed_frame)
                
                if activity is not None:
                    current_activity = activity
                    current_confidence = confidence
                    processing_stats['total_predictions'] += 1
                    processing_stats['activities_detected'].add(activity)
                    
                    if 'inference_time' in details:
                        # Update average inference time
                        prev_avg = processing_stats['avg_inference_time']
                        count = processing_stats['total_predictions']
                        processing_stats['avg_inference_time'] = (
                            (prev_avg * (count - 1) + details['inference_time']) / count
                        )
            
            # Get smoothed prediction for display
            smoothed_activity, smoothed_confidence = self.recognizer.get_smoothed_prediction()
            
            # Draw activity information on frame
            self.draw_activity_overlay(
                frame, smoothed_activity, smoothed_confidence,
                frame_count, total_frames, processing_stats
            )
            
            # Write frame
            out.write(frame)
            frame_count += 1
            
            # Progress logging
            if frame_count % 100 == 0 or frame_count == total_frames:
                progress = (frame_count / total_frames) * 100 if total_frames > 0 else 0
                logger.info(f"Progress: {progress:.1f}% - Current activity: {smoothed_activity}")
        
        # Cleanup
        cap.release()
        out.release()
        
        # Log final statistics
        logger.info(f"Processing completed for {input_path.name}")
        logger.info(f"Total predictions: {processing_stats['total_predictions']}")
        logger.info(f"Average inference time: {processing_stats['avg_inference_time']:.3f}s")
        logger.info(f"Activities detected: {list(processing_stats['activities_detected'])}")
    
    def draw_activity_overlay(self, frame, activity, confidence, frame_num, total_frames, stats):
        """Draw comprehensive activity information overlay"""
        height, width = frame.shape[:2]
        
        # Create semi-transparent overlay
        overlay = frame.copy()
        
        # Main info panel
        panel_height = 120
        cv2.rectangle(overlay, (10, 10), (width - 10, panel_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Activity information
        activity_display = activity.replace('_', ' ').title()
        cv2.putText(frame, f"Activity: {activity_display}", (20, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Confidence with color coding
        conf_color = (0, 255, 0) if confidence > 0.7 else (0, 255, 255) if confidence > 0.4 else (0, 0, 255)
        cv2.putText(frame, f"Confidence: {confidence:.2f}", (20, 65),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, conf_color, 2)
        
        # Progress information
        progress = (frame_num / total_frames) * 100 if total_frames > 0 else 0
        cv2.putText(frame, f"Frame: {frame_num}/{total_frames} ({progress:.1f}%)", (20, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Performance stats
        if stats['total_predictions'] > 0:
            cv2.putText(frame, f"Avg Inference: {stats['avg_inference_time']:.3f}s", (20, 110),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Confidence bar
        bar_x = width - 320
        bar_width = int(300 * confidence)
        cv2.rectangle(frame, (bar_x, 30), (bar_x + bar_width, 50), conf_color, -1)
        cv2.rectangle(frame, (bar_x, 30), (bar_x + 300, 50), (255, 255, 255), 2)
        
        # Activity history (small indicators)
        if len(self.recognizer.activity_history) > 1:
            y_pos = height - 30
            for i, (hist_activity, hist_conf) in enumerate(list(self.recognizer.activity_history)[-10:]):
                x_pos = 20 + i * 15
                color = (0, 255, 0) if hist_conf > 0.5 else (0, 255, 255)
                cv2.circle(frame, (x_pos, y_pos), 5, color, -1)

def main():
    """Main function to run the complete pipeline"""
    # Configuration
    config = SlowFastConfig()
    
    # Paths
    input_folder = "tracked_videos"
    output_folder = "slowfast_activity_videos"
    
    # Verify paths
    if not os.path.exists(input_folder):
        logger.error(f"Input folder '{input_folder}' not found!")
        logger.info("Please ensure your tracked videos are in the 'tracked_videos' folder.")
        return
    
    # Create analyzer
    analyzer = TrackedVideoAnalyzer(config)
    
    logger.info("SlowFast Activity Recognition Pipeline Starting")
    logger.info("=" * 60)
    logger.info(f"Input folder: {input_folder}")
    logger.info(f"Output folder: {output_folder}")
    logger.info(f"Device: {analyzer.recognizer.device}")
    logger.info(f"Activity classes: {len(config.ACTIVITY_CLASSES)}")
    
    try:
        # Process all tracked videos
        start_time = time.time()
        analyzer.process_tracked_videos(input_folder, output_folder)
        total_time = time.time() - start_time
        
        logger.info("=" * 60)
        logger.info("Pipeline completed successfully!")
        logger.info(f"Total processing time: {total_time:.2f} seconds")
        logger.info(f"Results saved in: {output_folder}")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise

if __name__ == "__main__":
    main()
    