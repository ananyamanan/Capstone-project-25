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
import torch.optim as optim
from pathlib import Path
import time
import logging
import random

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SlowFastNetwork(nn.Module):
    """Custom SlowFast Network implementation"""
    def __init__(self, num_classes=7):
        super(SlowFastNetwork, self).__init__()
        
        # Load pre-trained ResNet18 for slow pathway
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.slow_pathway = self._make_slow_pathway(resnet)
        
        # Fast pathway - Lighter network for temporal features
        self.fast_pathway = self._make_fast_pathway()
        
        # Fusion layer
        self.fusion = nn.AdaptiveAvgPool3d((1, 1, 1))
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512 + 128, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
        # Initialize classifier with zeros (to be fine-tuned)
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
        
    def _make_slow_pathway(self, resnet):
        """Create slow pathway using pre-trained ResNet18 adapted for 3D"""
        slow_layers = []
        
        # Modify first conv layer for 3D
        conv1 = nn.Conv3d(3, 64, kernel_size=(1, 7, 7), 
                         stride=(1, 2, 2), padding=(0, 3, 3))
        with torch.no_grad():
            conv1.weight.data = torch.stack([resnet.conv1.weight.data] * 1, dim=2)
        slow_layers.append(conv1)
        
        slow_layers.append(nn.BatchNorm3d(64))
        slow_layers.append(nn.ReLU(inplace=True))
        slow_layers.append(nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)))
        
        # Adapt ResNet layers for 3D
        slow_layers.extend([
            self._make_3d_block(64, 64, 2, resnet.layer1),
            self._make_3d_block(64, 128, 2, resnet.layer2),
            self._make_3d_block(128, 256, 2, resnet.layer3),
            self._make_3d_block(256, 512, 2, resnet.layer4),
            nn.AdaptiveAvgPool3d((1, 1, 1))
        ])
        
        return nn.Sequential(*slow_layers)
    
    def _make_fast_pathway(self):
        """Create fast pathway for temporal features"""
        fast_layers = []
        
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
    
    def _make_3d_block(self, in_channels, out_channels, num_blocks, resnet_layer=None):
        """Create 3D convolution block with optional ResNet layer adaptation"""
        layers = []
        for i in range(num_blocks):
            if resnet_layer is not None and i == 0:
                conv2d = resnet_layer[i].conv1
                conv3d = nn.Conv3d(in_channels, out_channels, 
                                 kernel_size=(1, 3, 3), padding=(0, 1, 1))
                with torch.no_grad():
                    conv3d.weight.data = torch.stack([conv2d.weight.data] * 1, dim=2)
                layers.append(conv3d)
            else:
                layers.append(nn.Conv3d(out_channels, out_channels, 
                                      kernel_size=(1, 3, 3), padding=(0, 1, 1)))
            layers.append(nn.BatchNorm3d(out_channels))
            layers.append(nn.ReLU(inplace=True))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        """Forward pass with slow and fast pathways"""
        if isinstance(x, list):
            slow_input, fast_input = x
        else:
            slow_input = fast_input = x
        
        slow_features = self.slow_pathway(slow_input)
        fast_features = self.fast_pathway(fast_input)
        
        slow_features = slow_features.view(slow_features.size(0), -1)
        fast_features = fast_features.view(fast_features.size(0), -1)
        
        combined_features = torch.cat([slow_features, fast_features], dim=1)
        
        output = self.classifier(combined_features)
        
        return output

class SlowFastConfig:
    """Configuration class for SlowFast parameters"""
    def __init__(self, activity_classes=None):
        self.CLIP_LENGTH = 32  # Total frames for one prediction
        self.SAMPLE_RATE = 2   # Sample every nth frame
        self.SLOW_FAST_ALPHA = 4  # Ratio between slow and fast pathways
        self.INPUT_SIZE = 224  # Input image size
        self.CONFIDENCE_THRESHOLD = 0.3  # Minimum confidence for predictions
        
        # Dynamically set activity classes from tracked video data
        self.ACTIVITY_CLASSES = activity_classes if activity_classes else self._infer_activity_classes()
    
    def _infer_activity_classes(self):
        """Infer activity classes from the event_type subfolders in tracked_videos"""
        input_folder = "tracked_videos"
        activity_classes = set()
        
        if os.path.exists(input_folder):
            for category in ['anomaly', 'normal']:
                category_path = os.path.join(input_folder, category)
                if os.path.isdir(category_path):
                    for event_type in os.listdir(category_path):
                        event_path = os.path.join(category_path, event_type)
                        if os.path.isdir(event_path):
                            activity_classes.add(event_type.lower())
        
        return sorted(list(activity_classes)) or ["unknown"]

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
    def preprocess_frame_ffmpeg(frame, target_size=(224, 224), transform=None):
        """Preprocess frame for SlowFast input with optional transform"""
        frame_resized = cv2.resize(frame, target_size)
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        if transform:
            frame_tensor = transform(frame_rgb.astype(np.float32) / 255.0)
            return frame_tensor.numpy().transpose(1, 2, 0)  # Convert back to HWC for OpenCV
        return frame_rgb

class VideoDataset(torch.utils.data.Dataset):
    """Custom dataset for tracked videos"""
    def __init__(self, root_dir, config, transform=None, train=True):
        self.root_dir = Path(root_dir)
        self.config = config
        self.transform = transform
        self.train = train
        self.video_files = []
        self.labels = []
        all_files = []
        for category in ['anomaly', 'normal']:
            category_path = self.root_dir / category
            if category_path.exists() and category_path.is_dir():
                for event_type in category_path.iterdir():
                    if event_type.is_dir():
                        label_idx = config.ACTIVITY_CLASSES.index(event_type.name.lower())
                        for video_file in event_type.glob('*.avi'):
                            all_files.append((video_file, label_idx))
        # Split dataset (80% train, 20% test)
        random.shuffle(all_files)
        split_idx = int(0.8 * len(all_files))
        if train:
            self.video_files, self.labels = zip(*all_files[:split_idx])
        else:
            self.video_files, self.labels = zip(*all_files[split_idx:])
        logger.info(f"Loaded {len(self.video_files)} videos for {'train' if train else 'test'} set")

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, idx):
        video_path = self.video_files[idx]
        cap = cv2.VideoCapture(str(video_path))
        frames = []
        for _ in range(self.config.CLIP_LENGTH):
            ret, *frame_data = cap.read()
            if not ret or not frame_data:
                break
            frame = frame_data[0] if frame_data else None
            if frame is not None:
                frame = VideoProcessor.preprocess_frame_ffmpeg(frame, (self.config.INPUT_SIZE, self.config.INPUT_SIZE), self.transform)
                frames.append(frame)
        cap.release()
        if len(frames) < self.config.CLIP_LENGTH:
            frames.extend([frames[-1]] * (self.config.CLIP_LENGTH - len(frames)))
        frames = torch.stack(frames)
        label = self.labels[idx]
        return frames.permute(3, 0, 1, 2).float(), label

class SlowFastActivityRecognizer:
    """SlowFast Network for activity recognition"""
    
    def __init__(self, config=None, device='auto'):
        self.config = config or SlowFastConfig()
        
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        logger.info(f"Using device: {self.device}")
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.model = self._load_model()
        
        self.frame_buffer = deque(maxlen=self.config.CLIP_LENGTH)
        self.activity_history = deque(maxlen=10)
        
    def _load_model(self):
        """Load custom SlowFast model with dynamic number of classes and fine-tuning"""
        try:
            logger.info("Loading custom SlowFast Network...")
            
            model = SlowFastNetwork(num_classes=len(self.config.ACTIVITY_CLASSES))
            model.to(self.device)
            
            model_path = "slowfast_weights.pth"
            if os.path.exists(model_path):
                logger.info(f"Loading pretrained weights from {model_path}")
                checkpoint = torch.load(model_path, map_location=self.device)
                model.load_state_dict(checkpoint)
            else:
                logger.info("No pretrained weights found, performing fine-tuning...")
                # Fine-tuning setup
                train_dataset = VideoDataset("tracked_videos", self.config, transform=self.transform, train=True)
                test_dataset = VideoDataset("tracked_videos", self.config, transform=self.transform, train=False)
                train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=2, shuffle=True)
                test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=2, shuffle=False)
                optimizer = optim.Adam(model.parameters(), lr=0.001)
                criterion = nn.CrossEntropyLoss()
                model.train()
                best_acc = 0.0
                for epoch in range(3):  # 3 epochs
                    running_loss = 0.0
                    for i, (inputs, labels) in enumerate(train_loader):
                        inputs, labels = inputs.to(self.device), labels.to(self.device)
                        optimizer.zero_grad()
                        slow_fast_input = [inputs[:, ::self.config.SLOW_FAST_ALPHA], inputs]
                        outputs = model(slow_fast_input)
                        loss = criterion(outputs, labels)
                        loss.backward()
                        optimizer.step()
                        running_loss += loss.item()
                        if i % 10 == 9:
                            logger.info(f"Epoch {epoch+1}, Batch {i+1}, Loss: {running_loss/10:.3f}")
                            running_loss = 0.0
                    # Validation
                    model.eval()
                    correct = 0
                    total = 0
                    with torch.no_grad():
                        for inputs, labels in test_loader:
                            inputs, labels = inputs.to(self.device), labels.to(self.device)
                            slow_fast_input = [inputs[:, ::self.config.SLOW_FAST_ALPHA], inputs]
                            outputs = model(slow_fast_input)
                            _, predicted = torch.max(outputs.data, 1)
                            total += labels.size(0)
                            correct += (predicted == labels).sum().item()
                    accuracy = 100 * correct / total
                    logger.info(f"Epoch {epoch+1} Validation Accuracy: {accuracy:.2f}%")
                    if accuracy > best_acc:
                        best_acc = accuracy
                        torch.save(model.state_dict(), model_path)
                model.load_state_dict(torch.load(model_path))
                logger.info("Fine-tuning completed and best weights saved.")
            model.eval()
            return model
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def _prepare_slowfast_input(self, frames):
        """Prepare input tensors for SlowFast network"""
        if len(frames) < self.config.CLIP_LENGTH:
            while len(frames) < self.config.CLIP_LENGTH:
                frames.append(frames[-1])
        
        processed_frames = []
        for frame in frames:
            frame_tensor = self.transform(frame)
            processed_frames.append(frame_tensor)
        
        video_tensor = torch.stack(processed_frames)
        
        slow_indices = torch.arange(0, len(processed_frames), self.config.SLOW_FAST_ALPHA)
        if len(slow_indices) > self.config.CLIP_LENGTH // self.config.SLOW_FAST_ALPHA:
            slow_indices = slow_indices[:self.config.CLIP_LENGTH // self.config.SLOW_FAST_ALPHA]
        slow_pathway = video_tensor[slow_indices]
        
        fast_pathway = video_tensor
        
        slow_pathway = slow_pathway.unsqueeze(0).transpose(1, 2).to(self.device)
        fast_pathway = fast_pathway.unsqueeze(0).transpose(1, 2).to(self.device)
        
        return [slow_pathway, fast_pathway]
    
    def predict_activity(self, frames):
        """Predict activity from frame sequence"""
        if len(frames) < 8:
            return "insufficient_data", 0.0, {}
        
        try:
            slow_fast_input = self._prepare_slowfast_input(frames)
            
            with torch.no_grad():
                start_time = time.time()
                outputs = self.model(slow_fast_input)
                inference_time = time.time() - start_time
                
                probabilities = F.softmax(outputs, dim=1)
                
                top_k = min(5, len(self.config.ACTIVITY_CLASSES))
                top_probs, top_indices = torch.topk(probabilities, top_k, dim=1)
                
                predicted_idx = top_indices[0][0].item() % len(self.config.ACTIVITY_CLASSES)
                predicted_class = self.config.ACTIVITY_CLASSES[predicted_idx]
                confidence = top_probs[0][0].item()
                
                # Reset prediction if confidence is too low for too long
                if confidence < self.config.CONFIDENCE_THRESHOLD and len(self.activity_history) > 5:
                    if all(hist_conf < self.config.CONFIDENCE_THRESHOLD for _, hist_conf in self.activity_history):
                        predicted_class = "uncertain"
                        confidence = 0.0
                
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
        processed_frame = VideoProcessor.preprocess_frame_ffmpeg(frame, (self.config.INPUT_SIZE, self.config.INPUT_SIZE), self.transform)
        self.frame_buffer.append(processed_frame)
        
        if len(self.frame_buffer) == self.config.CLIP_LENGTH:
            activity, confidence, details = self.predict_activity(list(self.frame_buffer))
            
            if confidence > self.config.CONFIDENCE_THRESHOLD:
                self.activity_history.append((activity, confidence))
            
            return activity, confidence, details
        
        return None, 0.0, {}
    
    def get_smoothed_prediction(self):
        """Get smoothed prediction based on recent history"""
        if not self.activity_history:
            return "no_activity", 0.0
        
        activity_counts = defaultdict(list)
        for activity, confidence in self.activity_history:
            activity_counts[activity].append(confidence)
        
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
        input_path = Path(input_folder)  # Path for input folder (e.g., tracked_videos)
        output_path = Path(output_folder)  # Path for output folder (e.g., slowfast_activity_videos)
        output_path.mkdir(exist_ok=True)
        
        video_files = []
        for category in ['anomaly', 'normal']:
            category_path = input_path / category  # e.g., tracked_videos/anomaly
            if category_path.exists() and category_path.is_dir():
                for event_type in category_path.iterdir():
                    if event_type.is_dir():
                        video_files.extend(event_type.glob('*.avi'))
        
        if not video_files:
            logger.error(f"No .avi files found in {input_folder}")
            return
        
        logger.info(f"Found {len(video_files)} videos to process")
        
        for video_file in video_files:
            relative_path = video_file.relative_to(input_path)
            output_subfolder = output_path / relative_path.parent  # e.g., slowfast_activity_videos/anomaly/carcrash
            output_subfolder.mkdir(parents=True, exist_ok=True)
            output_file = output_subfolder / f"slowfast_{video_file.name}"
            logger.info(f"Processing: {video_file.name} in {relative_path.parent}")
            
            try:
                self.process_single_video(video_file, output_file)
                logger.info(f"Completed: {video_file.name}")
            except Exception as e:
                logger.error(f"Error processing {video_file.name}: {e}")
    
    def process_single_video(self, input_path, output_path):
        """Process a single tracked video"""
        video_info = self.video_processor.extract_video_info(input_path)
        if not video_info:
            logger.error(f"Could not extract video info for {input_path}")
            return
        
        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            logger.error(f"Could not open video: {input_path}")
            return
        
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        fps = video_info['fps']
        width, height = video_info['width'], video_info['height']
        
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
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
            ret, *frame_data = cap.read()  # Handle extra values
            if not ret or not frame_data:
                break
            
            frame = frame_data[0] if frame_data else None
            if frame is None:
                logger.warning(f"No frame data for frame {frame_count} in {input_path}")
                break
            
            processed_frame = VideoProcessor.preprocess_frame_ffmpeg(
                frame, (self.config.INPUT_SIZE, self.config.INPUT_SIZE), self.recognizer.transform
            )
            
            if frame_count % 4 == 0:
                activity, confidence, details = self.recognizer.update_and_predict(processed_frame)
                
                if activity is not None:
                    current_activity = activity
                    current_confidence = confidence
                    processing_stats['total_predictions'] += 1
                    processing_stats['activities_detected'].add(activity)
                    
                    if 'inference_time' in details:
                        prev_avg = processing_stats['avg_inference_time']
                        count = processing_stats['total_predictions']
                        processing_stats['avg_inference_time'] = (
                            (prev_avg * (count - 1) + details['inference_time']) / count
                        )
                    
                    # Print activity label to terminal
                    print(f"Frame {frame_count}: Activity = {current_activity}, Confidence = {current_confidence:.2f}")
            
            smoothed_activity, smoothed_confidence = self.recognizer.get_smoothed_prediction()
            
            self.draw_activity_overlay(
                frame, smoothed_activity, smoothed_confidence,
                frame_count, total_frames, processing_stats
            )
            
            out.write(frame)
            frame_count += 1
            
            if frame_count % 100 == 0 or frame_count == total_frames:
                progress = (frame_count / total_frames) * 100 if total_frames > 0 else 0
                logger.info(f"Progress: {progress:.1f}% - Current activity: {smoothed_activity}")
        
        cap.release()
        out.release()
        
        logger.info(f"Processing completed for {input_path.name}")
        logger.info(f"Total predictions: {processing_stats['total_predictions']}")
        logger.info(f"Average inference time: {processing_stats['avg_inference_time']:.3f}s")
        logger.info(f"Activities detected: {list(processing_stats['activities_detected'])}")
    
    def draw_activity_overlay(self, frame, activity, confidence, frame_num, total_frames, stats):
        """Draw comprehensive activity information overlay"""
        height, width = frame.shape[:2]
        
        overlay = frame.copy()
        
        panel_height = 120
        cv2.rectangle(overlay, (10, 10), (width - 10, panel_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        activity_display = activity.replace('_', ' ').title()
        cv2.putText(frame, f"Activity: {activity_display}", (20, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        conf_color = (0, 255, 0) if confidence > 0.7 else (0, 255, 255) if confidence > 0.4 else (0, 0, 255)
        cv2.putText(frame, f"Confidence: {confidence:.2f}", (20, 65),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, conf_color, 2)
        
        progress = (frame_num / total_frames) * 100 if total_frames > 0 else 0
        cv2.putText(frame, f"Frame: {frame_num}/{total_frames} ({progress:.1f}%)", (20, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        if stats['total_predictions'] > 0:
            cv2.putText(frame, f"Avg Inference: {stats['avg_inference_time']:.3f}s", (20, 110),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        bar_x = width - 320
        bar_width = int(300 * confidence)
        cv2.rectangle(frame, (bar_x, 30), (bar_x + bar_width, 50), conf_color, -1)
        cv2.rectangle(frame, (bar_x, 30), (bar_x + 300, 50), (255, 255, 255), 2)
        
        if len(self.recognizer.activity_history) > 1:
            y_pos = height - 30
            for i, (hist_activity, hist_conf) in enumerate(list(self.recognizer.activity_history)[-10:]):
                x_pos = 20 + i * 15
                color = (0, 255, 0) if hist_conf > 0.5 else (0, 255, 255)
                cv2.circle(frame, (x_pos, y_pos), 5, color, -1)

def main():
    """Main function to run the complete pipeline"""
    input_folder = "tracked_videos"  # Input folder path
    output_folder = "slowfast_activity_videos"  # Output folder path
    
    if not os.path.exists(input_folder):
        logger.error(f"Input folder '{input_folder}' not found!")
        logger.info("Please ensure your tracked videos are in the 'tracked_videos' folder.")
        return
    
    config = SlowFastConfig()  # Dynamically infers ACTIVITY_CLASSES
    analyzer = TrackedVideoAnalyzer(config)
    
    logger.info("SlowFast Activity Recognition Pipeline Starting")
    logger.info("=" * 60)
    logger.info(f"Input folder: {input_folder}")
    logger.info(f"Output folder: {output_folder}")
    logger.info(f"Device: {analyzer.recognizer.device}")
    logger.info(f"Activity classes: {len(config.ACTIVITY_CLASSES)} - {config.ACTIVITY_CLASSES}")
    
    try:
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