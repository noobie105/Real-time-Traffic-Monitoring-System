# Real-Time Traffic Monitoring System

This project implements a high-performance computer vision pipeline for real-time vehicle detection, multi-object tracking, and traffic monitoring with a functional GUI.

## Objective
Build and deploy a low-latency object detection + tracking system that:
- Detects and classifies vehicles in real time
- Assigns consistent tracking IDs across frames (handles brief occlusions)
- Displays annotated live feed in a user-friendly GUI with performance metrics

## Features
- **Model**: Fine-tuned **YOLOv8m** on the provided 462-image dataset + class names
- **Tracking**: Built-in Ultralytics tracker (BoT-SORT / ByteTrack) with persistent IDs
- **GUI** (Tkinter):
  - Live video feed with bounding boxes, class labels, tracking IDs, confidence scores
  - Start / Stop buttons
  - Real-time dashboard: FPS, CPU%, GPU%, object count
- **Video Processing**: One-click generation of fully annotated output video
- **Hardware Awareness**: GPU support (CUDA) with CPU fallback; metrics via psutil & GPUtil

## Project Structure
real-time-traffic-monitoring-system
├── GUI.py                  # Main application with Tkinter GUI and YOLO inference
├── Real-time Traffic Monitoring System.ipynb     # Fine-tuned YOLOv8m model
└── requirements.txt        # Dependencies

## Setup & Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/YOUR_USERNAME/real-time-traffic-monitoring-system.git
   cd real-time-traffic-monitoring-system
   ```
2. Create & activate virtual environment (recommended
   python -m venv venv
   ```bash
   # Windows
   venv\Scripts\activate
   # macOS/Linux
   source venv/bin/activate
   ```
3. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```
4. Update paths in GUI.py
5. Run the application:
   ```bash
   python GUI.py
   ```

