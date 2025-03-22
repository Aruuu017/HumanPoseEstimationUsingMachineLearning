🎯 Human Pose Estimation Using Machine Learning
This project implements a Human Pose Estimation system using machine learning models, specifically leveraging OpenCV and MediaPipe for pose detection and classification.

📚 Project Overview
Human Pose Estimation is the task of detecting key points (joints) in the human body and predicting their precise positions. This project uses OpenCV and MediaPipe libraries to track and estimate body landmarks in real time.

🛠️ Technologies Used
Python: Core programming language

OpenCV: For image and video processing

MediaPipe: To detect human body landmarks

NumPy: To process landmark coordinates

Matplotlib: For visualization and analysis

Scikit-learn: For machine learning model training and evaluation

Streamlit (optional): For creating a web-based interface

📁 Project Structure
bash
Copy
Edit
📂 HumanPoseEstimationUsingMachineLearning
├── 📂 Dataset
│   └── landmarks.csv          # Preprocessed dataset with landmark coordinates
├── 📂 Models
│   └── pose_model.pkl         # Trained ML model for classification
├── 📂 Utils
│   └── preprocess.py          # Script for data preprocessing
├── 📂 Images
│   └── sample_pose.png        # Sample output images
├── 📂 App
│   └── app.py                 # Streamlit-based web application
├── main.py                    # Main script for pose estimation
├── requirements.txt           # Required dependencies
└── README.md                  # Project documentation
📊 Dataset Information
The dataset contains pose landmarks captured from multiple frames.

Keypoints include landmarks such as shoulders, elbows, wrists, knees, and ankles.

CSV file landmarks.csv stores landmark coordinates and corresponding labels for classification.

🧩 Model Information
Model Type: Random Forest / Decision Tree (or other models if applicable)

Input: 33 Landmark points from MediaPipe Pose Model

Output: Predicted pose class (e.g., standing, sitting, jumping)


⚙️ Usage
1. Run Pose Estimation Script
bash
Copy
Edit
python main.py
This will load the trained model and perform real-time pose estimation.

Displays pose keypoints overlaid on the live webcam feed.

2. Run Streamlit Web Application (Optional)
bash
Copy
Edit
streamlit run App/app.py
Access the app at: http://localhost:8501/

Provides an interactive interface to upload videos and analyze pose predictions.

🔬 Model Training Workflow
1. Data Preprocessing
Extract pose landmarks using MediaPipe.

Normalize and save data to landmarks.csv.

2. Model Training
Load preprocessed data.

Split data into training and testing sets.

Train the machine learning model and save it as pose_model.pkl.

📊 Evaluation Metrics
Accuracy: Measures the percentage of correct predictions.

Confusion Matrix: Visualizes model performance across multiple classes.

Precision & Recall: Assesses the reliability of predictions.

🖼️ Sample Output

📝 Future Scope
Integration with Deep Learning models such as CNNs.

Real-time multi-person pose estimation.

Mobile application for fitness and sports analysis.

Gesture recognition for interactive applications.


📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

📧 Contact
Name : Aryan Zende
Email: aryanzende017@gmail.com
GitHub: Aruuu017

