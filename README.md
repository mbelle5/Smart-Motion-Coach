
# Smart Motion Coach

Smart Motion Coach is a Python-based exercise tracking application that uses computer vision to monitor body movements and provide real-time feedback through a webcam.

The project uses OpenCV and Google's MediaPipe Pose Landmarker to detect body landmarks, calculate joint angles, track exercise repetitions, and identify potential posture alignment issues.

## Features

### 1. Squat Form Checker
- Tracks squat repetitions.
- Calculates knee angles using 3D body landmarks.
- Detects squat depth and standing position.
- Provides feedback on torso leaning.

### 2. Arm Raise Tracker
- Tracks arm raise repetitions.
- Calculates shoulder angles.
- Detects when both arms are raised and lowered.
- Provides real-time exercise feedback.

### 3. Bicep Curl Tracker
- Tracks bicep curl repetitions.
- Calculates elbow angles.
- Detects completed repetitions when both arms curl and return to the extended position.
- Displays joint angles and movement feedback.

### 4. Posture Monitor
- Measures shoulder alignment.
- Calculates upper-body lean.
- Detects potential posture issues.
- Displays warnings when poor alignment continues for more than 1.5 seconds.

## Technologies Used

- Python
- OpenCV
- MediaPipe
- NumPy
- Git and GitHub

## How It Works

Smart Motion Coach uses a webcam to capture live video.

1. OpenCV captures and displays video frames.
2. MediaPipe detects body landmarks.
3. The application uses 3D landmark coordinates to calculate joint angles.
4. Each exercise tracker compares those angles against predefined thresholds.
5. The application counts repetitions or provides posture feedback based on the detected movement.

The application provides visual feedback directly on the webcam display.

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/mbelle5/Smart-Motion-Coach.git
```

### 2. Navigate to the Project Folder

```bash
cd Smart-Motion-Coach
```

### 3. Install Dependencies

Make sure Python is installed, then run:

```bash
python3 -m pip install -r requirements.txt
```

### 4. Download the MediaPipe Model

Download the MediaPipe Pose Landmarker Lite model:

https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task

Place the downloaded file in the main project folder with the filename:

`pose_landmarker_lite.task`

### 5. Run the Application

```bash
python3 smart_motion_coach.py
```

## Application Menu

When the program starts, users can select:

1. Squat Form Checker
2. Arm Raise Tracker
3. Bicep Curl Tracker
4. Posture Monitor
5. Exit

Press **Q** in the webcam window to stop an exercise and return to the main menu.

## Project Structure

```text
Smart-Motion-Coach/
├── smart_motion_coach.py
├── squat_form_checker.py
├── arm_raise_tracker.py
├── bicep_curl_tracker.py
├── posture_monitor.py
├── core/
│   ├── __init__.py
│   ├── geometry.py
│   └── pose.py
├── tests/
│   └── test_geometry.py
├── pose_landmarker_lite.task
├── requirements.txt
└── README.md
```

## Testing

The project includes unit tests for shared geometry functions.

Run the tests with:

```bash
python3 -m unittest discover -s tests -v
```

## Future Improvements

- Support for additional exercises.
- Individual arm repetition tracking.
- Exercise history and progress tracking.
- More advanced form analysis.
- A graphical user interface.
- Improved movement accuracy and calibration.

## Project Purpose

Smart Motion Coach was developed to explore how computer vision and pose estimation can be applied to exercise tracking and movement analysis.

The project demonstrates Python programming, mathematical calculations, real-time video processing, and modular software design.

## Limitations

The application uses experimental angle thresholds and webcam-based pose estimation. Tracking accuracy can vary with camera position, lighting, and landmark visibility.

Smart Motion Coach is intended as an educational project and is not a substitute for professional fitness or medical guidance.
