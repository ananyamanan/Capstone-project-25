import os
import cv2
import torch
from torch_geometric.data import Data
import numpy as np
import pytesseract
from pathlib import Path
import logging
from scipy.spatial import KDTree

# Set the path to Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
class GraphConfig:
    def __init__(self):
        self.NUM_CLASSES = 40
        self.NUM_FEATURES = 6  # [x_center, y_center, width, height, speed_x, speed_y]
        self.CLIP_LENGTH = 32  # Frames per graph
        self.DISTANCE_THRESHOLD = 100  # Pixels for edge creation
        self.TRACKED_VIDEOS_DIR = "tracked_videos"
        self.SLOWFAST_VIDEOS_DIR = "slowfast_activity_videos"
        # Activity classes (same as SlowFast)
        self.ACTIVITY_CLASSES = [
            'walking', 'running', 'jogging', 'jumping', 'sitting', 'standing',
            'waving', 'clapping', 'dancing', 'eating', 'drinking', 'talking',
            'reading', 'writing', 'sleeping', 'exercising', 'playing_sports',
            'cooking', 'cleaning', 'driving', 'cycling', 'swimming', 'climbing',
            'carrying', 'pushing', 'pulling', 'throwing', 'catching', 'kicking',
            'punching', 'hugging', 'shaking_hands', 'applauding', 'crawling',
            'falling_down', 'getting_up', 'lying_down', 'bending', 'stretching'
        ]
        self.ACTIVITY_MAP = {name: idx for idx, name in enumerate(self.ACTIVITY_CLASSES)}

# Extract bounding boxes and track IDs (optimized)
def extract_object_data_from_frame(frame):
    """
    Extract bounding boxes and track IDs from a frame using OpenCV.
    DeepSORT draws green boxes with "ID X" labels above them.
    """
    objects = []
    
    # Convert frame to HSV to isolate green boxes
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    green_lower = np.array([35, 100, 100])
    green_upper = np.array([85, 255, 255])
    mask = cv2.inRange(hsv, green_lower, green_upper)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        if cv2.contourArea(contour) < 100:  # Ignore small contours
            continue
        x, y, w, h = cv2.boundingRect(contour)
        # Extract text region above the box
        text_region = frame[max(0, y-30):y, x:x+w]
        if text_region.size == 0:
            continue
        # Preprocess text region for faster OCR
        text_region = cv2.resize(text_region, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
        text = pytesseract.image_to_string(text_region, config='--psm 7').strip()
        track_id = None
        if "ID" in text:
            try:
                track_id = int(text.split("ID")[-1].strip())
            except ValueError:
                continue
        if track_id is not None:
            objects.append({
                'track_id': track_id,
                'bbox': [x, y, x+w, y+h]  # [x1, y1, x2, y2]
            })
    
    return objects

# Extract activity labels (optimized with better OCR and logging)
def extract_activity_from_frame(frame, config):
    """
    Extract activity labels from SlowFast frame using OCR.
    SlowFast overlays text like "Activity: Walking".
    """
    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Try different regions for the activity text
    regions = [
        (10, 50, 20, 300),  # Original: top-left corner
        (0, 40, 0, 200),    # Slightly adjusted top-left
        (frame.shape[0]-50, frame.shape[0], 20, 300),  # Bottom-left corner
    ]
    
    for y1, y2, x1, x2 in regions:
        text_region = gray[y1:y2, x1:x2]
        if text_region.size == 0:
            continue
        # Preprocess for better OCR
        text_region = cv2.resize(text_region, None, fx=3, fy=3, interpolation=cv2.INTER_LINEAR)
        # Increase contrast
        text_region = cv2.convertScaleAbs(text_region, alpha=2, beta=0)
        text = pytesseract.image_to_string(text_region, config='--psm 6').strip()
        logger.info(f"OCR text from region ({y1}:{y2}, {x1}:{x2}): '{text}'")
        
        # Parse activity
        if "Activity:" in text:
            activity_name = text.split("Activity:")[-1].strip().lower().replace(" ", "_")
            # Try partial matches for activity name
            activity_idx = config.ACTIVITY_MAP.get(activity_name, -1)
            if activity_idx == -1:
                # Try matching the first few characters
                for key in config.ACTIVITY_MAP:
                    if key.startswith(activity_name[:3]):
                        activity_idx = config.ACTIVITY_MAP[key]
                        logger.info(f"Matched activity '{activity_name}' to '{key}' (idx: {activity_idx})")
                        break
            if activity_idx != -1:
                return [activity_idx]
    logger.warning("No activity detected in frame")
    return []

# Build relationship graphs (optimized with additional logging)
def create_relationship_graphs(config):
    """
    Create relationship graphs from paired videos, optimized for speed.
    """
    tracked_videos_path = Path(config.TRACKED_VIDEOS_DIR)
    slowfast_videos_path = Path(config.SLOWFAST_VIDEOS_DIR)

    tracked_videos = sorted([f for f in tracked_videos_path.glob("*.avi")])
    slowfast_videos = sorted([f for f in slowfast_videos_path.glob("*.avi")])

    if len(tracked_videos) != 16 or len(slowfast_videos) != 16:
        logger.error(f"Expected 16 videos. Found {len(tracked_videos)} in tracked_videos, {len(slowfast_videos)} in slowfast_activity_videos.")
        return []

    data_list = []
    for tracked_video_path, slowfast_video_path in zip(tracked_videos, slowfast_videos):
        logger.info(f"Processing video pair: {tracked_video_path.name} and {slowfast_video_path.name}")

        tracked_cap = cv2.VideoCapture(str(tracked_video_path))
        slowfast_cap = cv2.VideoCapture(str(slowfast_video_path))

        if not tracked_cap.isOpened() or not slowfast_cap.isOpened():
            logger.error(f"Failed to open video pair: {tracked_video_path.name} or {slowfast_video_path.name}")
            continue

        # Get total frames
        total_frames = min(int(tracked_cap.get(cv2.CAP_PROP_FRAME_COUNT)), 
                          int(slowfast_cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        logger.info(f"Total frames in {tracked_video_path.name}: {total_frames}")

        # Calculate expected number of graphs for this video
        expected_graphs = total_frames // config.CLIP_LENGTH
        logger.info(f"Expected graphs for {tracked_video_path.name}: {expected_graphs}")

        frame_idx = 0
        graphs_for_video = 0
        while frame_idx < total_frames:
            # Skip to the middle frame of the clip
            clip_start = frame_idx
            clip_end = min(clip_start + config.CLIP_LENGTH, total_frames)
            middle_frame_idx = clip_start + (clip_end - clip_start) // 2

            # Set frame position
            tracked_cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame_idx)
            slowfast_cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame_idx)

            # Read the middle frame
            ret_tracked, tracked_frame = tracked_cap.read()
            ret_slowfast, slowfast_frame = slowfast_cap.read()

            if not ret_tracked or not ret_slowfast:
                logger.warning(f"Failed to read frame {middle_frame_idx} in {tracked_video_path.name}")
                break

            # Extract data from the middle frame
            objects = extract_object_data_from_frame(tracked_frame)
            activities = extract_activity_from_frame(slowfast_frame, config)

            if not objects:
                logger.warning(f"No objects detected in frame {middle_frame_idx} of {tracked_video_path.name}")
                frame_idx += config.CLIP_LENGTH
                continue

            # Process objects
            frame_data = []
            positions = {}  # For speed calculation
            for i, obj in enumerate(objects):
                track_id = obj['track_id']
                x1, y1, x2, y2 = obj['bbox']
                width = x2 - x1
                height = y2 - y1
                x_center = x1 + width / 2
                y_center = y1 + height / 2
                activity_idx = activities[i % len(activities)] if activities else np.random.randint(0, config.NUM_CLASSES)  # Random activity if none detected
                logger.info(f"Assigned activity index {activity_idx} to object {track_id} in frame {middle_frame_idx}")
                frame_data.append({
                    'track_id': track_id,
                    'features': [x_center, y_center, width, height],
                    'activity': activity_idx,
                    'position': [x_center, y_center]
                })
                positions[track_id] = [x_center, y_center]

            # Create graph for this clip
            if frame_data:
                unique_objects = {}
                for obj in frame_data:
                    track_id = obj['track_id']
                    if track_id not in unique_objects:
                        unique_objects[track_id] = {
                            'features': obj['features'],
                            'activity': obj['activity'],
                            'position': obj['position']
                        }

                if unique_objects:
                    # Prepare node features and labels
                    node_features = []
                    node_labels = []
                    node_positions = []
                    track_id_to_idx = {}
                    for idx, (track_id, data) in enumerate(unique_objects.items()):
                        features = data['features']
                        # Approximate speed (since we only have one frame, set to 0)
                        speed = [0, 0]  # Simplified: no speed since we process one frame
                        features = np.concatenate([features, speed])
                        node_features.append(features)
                        node_labels.append(data['activity'])
                        node_positions.append(data['position'])
                        track_id_to_idx[track_id] = idx

                    # Use KDTree for efficient edge creation
                    node_positions = np.array(node_positions)
                    kdtree = KDTree(node_positions)
                    pairs = kdtree.query_pairs(config.DISTANCE_THRESHOLD)
                    edge_index = []
                    for i, j in pairs:
                        edge_index.append([i, j])
                        edge_index.append([j, i])

                    # Create PyG Data object
                    x = torch.from_numpy(np.array(node_features, dtype=np.float32))
                    y = torch.from_numpy(np.array(node_labels, dtype=np.int64))
                    edge_index = torch.from_numpy(np.array(edge_index, dtype=np.int64)).t().contiguous()
                    data = Data(x=x, edge_index=edge_index, y=y)
                    data_list.append(data)
                    graphs_for_video += 1
                else:
                    logger.warning(f"No unique objects in clip starting at frame {clip_start} of {tracked_video_path.name}")

            # Move to the next clip
            frame_idx += config.CLIP_LENGTH

        logger.info(f"Created {graphs_for_video} graphs for {tracked_video_path.name}")
        tracked_cap.release()
        slowfast_cap.release()

    logger.info(f"Total graphs created: {len(data_list)}")
    return data_list

def main():
    config = GraphConfig()
    graphs = create_relationship_graphs(config)
    torch.save(graphs, "relationship_graphs.pt")
    logger.info("Graphs saved to relationship_graphs.pt")

if __name__ == "__main__":
    main()