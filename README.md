# Monocular 3D Scene Understanding for ADAS

This project implements a real-time monocular vision pipeline for Advanced Driver Assistance Systems (ADAS). It performs depth estimation, 3D bounding box inference, lane detection, BEV rendering, and human mesh recovery.

## Features

- Monocular depth estimation (Depth Anything v2)
- Real-time object detection and tracking (YOLOv11)
- 3D bounding box inference using camera projection + Kalman filtering
- Classical lane detection (Canny + Hough)
- Bird's Eye View (BEV) overlay with class-specific icons
- Human mesh recovery using VIBE + SMPL

---

## Setup Instructions

```bash
# Create and activate virtual environment
python -m venv env
source $PWD/env/bin/activate

# Install PyTorch for CUDA 11.8
pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install -r requirements.txt

# Give execution permission
chmod +x run.sh

# Execute the pipeline
./run.sh

Requirements:
Python 3.8+
CUDA 11.8
NVIDIA GPU with compatible driver
