# Yoga Pose Detection and Report Generator

This project uses computer vision and machine learning to detect and provide real-time feedback on the yoga Tree Pose (Vrikshasana). It also generates comprehensive reports to help users track their progress, receive personalized recommendations, and gain health insights based on their practice.

## Features

- **Real-time Pose Detection:** Utilizes MediaPipe and a trained ML model to identify the Tree Pose from webcam input.
- **Audio Feedback:** Provides instant voice feedback to help users correct their pose.
- **Detailed Pose Analysis:** Evaluates hand position, leg alignment, and overall posture, offering specific corrections.
- **Comprehensive Reporting:** Generates PDF reports summarizing session performance, progress trends, and common issues.
- **Health Analysis:** Identifies potential health concerns based on recurring practice patterns and user profile.
- **Personalized Recommendations:** Offers tailored advice considering user fitness level, medical history, and performance.

## Requirements

- Python 3.7+
- Webcam
- Required Python libraries (see `requirements.txt`)

## Installation

1. **Clone the repository:**
   ```
   git clone https://github.com/aditya13504/yoga-pose-detection-and-report-generator.git
   cd yoga-pose-detection-and-report-generator
   ```
2. **Install dependencies:**
   ```
   pip install -r requirements.txt
   ```
3. **Prepare data:**
   - Place a reference image named `reference_tree_pose.jpg` in the project directory.
   - Ensure `tree/` and `other/` directories contain labeled training images if you wish to retrain the model.

## Usage

1. **Run the main application:**
   ```
   python yoga_detector.py
   ```
2. **Workflow:**
   - The app displays a reference pose.
   - Activates your webcam for real-time pose analysis.
   - Provides audio feedback for corrections.
   - Saves images of correctly performed poses.
   - Generates a detailed PDF report after your session.

## Report Generation

After each session, the application generates a PDF report including:
- **Session Analysis:** Details of each practice session.
- **Progress Tracking:** Charts showing improvement over time.
- **Accuracy Metrics:** Quantitative pose accuracy.
- **Common Issues:** Recurring problems in your practice.
- **Health Considerations:** Potential health impacts based on your data.
- **Personalized Recommendations:** Custom advice based on your profile and performance.

Reports are saved in the `reports/` directory.

## User Profile

On first use, you'll be prompted to create a user profile including:
- Personal information (name, age, height, weight, fitness level)
- Medical history (conditions, injuries, limitations)
- Yoga experience (years practicing, frequency, favorite/challenging poses)

This information is used to generate personalized recommendations and health insights.

## Project Structure

- `yoga_detector.py`: Main application for pose detection and feedback.
- `yoga_report_generator.py`: Handles report generation and user/session data.
- `yoga_pose_mlp.joblib`: Pre-trained ML model for pose classification.
- `pose_scaler.joblib`: Feature scaler for the ML model.
- `tree/` & `other/`: Directories with training images for model training.
- `correct_poses/`: Stores images of correctly performed poses.
- `reports/`: Stores generated PDF reports and trend graphs.
- `session_data/`: Stores session data as JSON files.
- `user_data.json`: Stores user profile information.
- `requirements.txt`: List of required Python packages.

## Example Session Data

A session JSON file in `session_data/` looks like:
```json
{
   "timestamp": "20250502_210741",
    "pose_type": "tree",
    "duration": 52,
    "attempts": 0,
    "correct_poses": [],
    "feedback_history": [
       "Try to form the Tree Pose shape."
    ]
}
```

## Notes

- Currently supports only the Tree Pose (Vrikshasana).
- For best results, ensure good lighting and a clear camera view.
- If you have medical conditions or injuries, consult a healthcare professional before practicing yoga.

## Future Enhancements that can be done

- Support for additional yoga poses
- Cloud synchronization of practice data
- Comparison with professional yoga practitioners
- Integration with fitness tracking apps

 ## License
 
 MIT License
 
 ## Contributions
 
 Always open for all.