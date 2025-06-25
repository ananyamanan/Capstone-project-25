#!/usr/bin/env python3
"""
Setup script for SlowFast Activity Recognition Integration
Works with existing YOLOv8 + DeepSORT pipeline
"""

import subprocess
import sys
import os
import shutil
from pathlib import Path

def check_system_requirements():
    """Check if all system tools are available"""
    print("🔍 Checking system requirements...")
    
    required_tools = {
        'ffmpeg': 'FFmpeg (for video processing)',
        'ffprobe': 'FFprobe (for video info extraction)'
    }
    
    missing_tools = []
    
    for tool, description in required_tools.items():
        if shutil.which(tool) is None:
            missing_tools.append((tool, description))
            print(f"❌ {tool} not found")
        else:
            print(f"✅ {tool} found")
    
    if missing_tools:
        print("\n⚠️  Missing system tools:")
        for tool, desc in missing_tools:
            print(f"   - {tool}: {desc}")
        
        print("\n📝 Installation instructions for macOS:")
        print("   brew install ffmpeg")
        print("\n   Or download from: https://ffmpeg.org/download.html")
        return False
    
    return True

def check_python_packages():
    """Check and install required Python packages"""
    print("\n🐍 Checking Python packages...")
    
    existing_packages = [
        'torch', 'torchvision', 'opencv-python', 'numpy', 
        'pillow', 'ultralytics', 'deep-sort-realtime'
    ]
    
    additional_packages = []
    
    for package in existing_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - should be in your requirements.txt")
            additional_packages.append(package)
    
    if additional_packages:
        print(f"\n📦 Installing missing packages: {', '.join(additional_packages)}")
        for package in additional_packages:
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                print(f"✅ Installed {package}")
            except subprocess.CalledProcessError:
                print(f"❌ Failed to install {package}")
                return False
    
    return True

def verify_torch_installation():
    """Verify PyTorch installation and check for GPU support"""
    print("\n🔥 Verifying PyTorch installation...")
    
    try:
        import torch
        import torchvision
        
        print(f"✅ PyTorch version: {torch.__version__}")
        print(f"✅ TorchVision version: {torchvision.__version__}")
        
        if torch.cuda.is_available():
            print(f"🚀 CUDA available - GPU: {torch.cuda.get_device_name(0)}")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            print("🚀 Apple Metal Performance Shaders (MPS) available")
        else:
            print("💻 Using CPU (no GPU acceleration)")
        
        print("🧪 Testing SlowFast model creation...")
        
        import torch.nn as nn
        
        class TestSlowFast(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv3d = nn.Conv3d(3, 64, kernel_size=(1, 7, 7))
                self.classifier = nn.Linear(64, 10)
                
            def forward(self, x):
                if isinstance(x, list):
                    x = x[0]
                b, c, t, h, w = x.shape
                x = self.conv3d(x)
                x = torch.mean(x, dim=[2, 3, 4])
                return self.classifier(x)
        
        model = TestSlowFast()
        print("✅ Custom SlowFast model structure created successfully")
        
        device = torch.device('cuda' if torch.cuda.is_available() else 
                            'mps' if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else 'cpu')
        
        model = model.to(device)
        dummy_input = torch.randn(1, 3, 8, 224, 224).to(device)
        
        with torch.no_grad():
            output = model([dummy_input, dummy_input])
            print(f"✅ Model forward pass successful - Output shape: {output.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ PyTorch verification failed: {e}")
        return False

def setup_project_structure():
    """Setup the required folder structure"""
    print("\n📁 Setting up project structure...")
    
    folders = {
        'tracked_videos': 'Input folder for tracked videos (.avi files) with subfolders anomaly/normal/event_type',
        'slowfast_activity_videos': 'Output folder for activity recognition results',
        'logs': 'Folder for log files'
    }
    
    for folder, description in folders.items():
        folder_path = Path(folder)
        if not folder_path.exists():
            folder_path.mkdir(exist_ok=True)
            print(f"✅ Created {folder} - {description}")
        else:
            print(f"✅ Found {folder}")
    
    # Check for input videos in nested structure
    tracked_videos_count = 0
    for category in ['anomaly', 'normal']:
        category_path = Path('tracked_videos') / category
        if category_path.exists():
            for event_type in category_path.iterdir():
                if event_type.is_dir():
                    tracked_videos = list(event_type.glob('*.avi'))
                    tracked_videos_count += len(tracked_videos)
    
    print(f"📹 Found {tracked_videos_count} .avi files in tracked_videos/{category}/<event_type>/")
    
    if tracked_videos_count == 0:
        print("⚠️  No .avi files found in tracked_videos/<category>/<event_type>/ folders")
        print("   Please add your tracked videos from deepsort_yolo.py output before running the pipeline")

def create_config_file():
    """Create a configuration file for easy customization"""
    config_content = '''# SlowFast Activity Recognition Configuration
# Modify these parameters as needed

[MODEL]
CLIP_LENGTH = 32          # Number of frames for one prediction
SAMPLE_RATE = 2           # Sample every nth frame
CONFIDENCE_THRESHOLD = 0.3 # Minimum confidence for predictions
INPUT_SIZE = 224          # Input image size
SLOW_FAST_ALPHA = 4       # Ratio between slow and fast pathways

[PROCESSING]
PROCESS_EVERY_N_FRAMES = 4  # Process every nth frame to optimize performance
USE_GPU = auto              # 'auto', 'cuda', 'mps', or 'cpu'
BATCH_SIZE = 1              # Batch size for processing

[OUTPUT]
DRAW_CONFIDENCE_BAR = true   # Show confidence bar in output
DRAW_ACTIVITY_HISTORY = true # Show activity history dots
OVERLAY_TRANSPARENCY = 0.7   # Overlay transparency (0.0 to 1.0)

[ACTIVITIES]
# Add or remove activities as needed based on SPHAR Dataset
CUSTOM_ACTIVITIES = [
    "sitting", "stealing", "vandalizing", "running",
    "carcrash", "kicking", "neutral"
]
'''
    
    config_path = Path('slowfast_config.ini')
    if not config_path.exists():
        config_path.write_text(config_content)
        print("✅ Created slowfast_config.ini")
    else:
        print("✅ Found existing slowfast_config.ini")

def run_quick_test():
    """Run a quick test to ensure everything works"""
    print("\n🧪 Running quick functionality test...")
    
    try:
        import torch
        import torchvision
        import cv2
        import numpy as np
        
        device = torch.device('cuda' if torch.cuda.is_available() else 
                            'mps' if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else 'cpu')
        print(f"✅ Using device: {device}")
        
        dummy_frame = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        
        print("✅ All components working correctly!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def main():
    """Main setup function"""
    print("🚀 SlowFast Activity Recognition Setup")
    print("=" * 50)
    print("This script will set up SlowFast integration for your existing")
    print("YOLOv8 + DeepSORT tracked videos pipeline.\n")
    
    if not check_system_requirements():
        print("\n❌ Setup failed: Missing system requirements")
        return False
    
    if not check_python_packages():
        print("\n❌ Setup failed: Python package issues")
        return False
    
    if not verify_torch_installation():
        print("\n❌ Setup failed: PyTorch verification failed")
        return False
    
    setup_project_structure()
    create_config_file()
    
    if not run_quick_test():
        print("\n❌ Setup failed: Functionality test failed")
        return False
    
    print("\n" + "=" * 50)
    print("🎉 Setup completed successfully!")
    print("\n📋 Next steps:")
    print("1. Add your tracked .avi videos to the 'tracked_videos/<category>/<event_type>/' folders")
    print("2. Run the activity recognition:")
    print("   python slowfast_activity.py")
    print("3. Check results in 'slowfast_activity_videos/' folder")
    print("\n⚙️  Configuration:")
    print("- Edit 'slowfast_config.ini' to customize settings")
    print("- Logs will be saved in 'logs/' folder")
    print("\n💡 Tips:")
    print("- GPU acceleration will be used automatically if available")
    print("- Processing time depends on video length and hardware")
    print("- Use smaller CLIP_LENGTH for faster processing")
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)