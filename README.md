Face recognition and Attendance System

An automated facial recognition system built to identify known students in classroom images and log attendance to a local CSV file

Consent and Privacy Notice

Explicit Consent: All student portrait images in the training and testing datasets are collected with explicit user consent.
No Public Distribution: Real student facial photographs and personal biometric assets are kept strictly private and will never be shared publicly or published to online repositories.
Academic & Ethical Standards: This project is developed solely for educational and research evaluation under privacy-compliant data guidelines.

Data Handling and Scrunity Rules

Local & Offline Processing: All facial detection, embedding extraction, and attendance logging operations are performed 100% locally and offline on the host computer.
Repository Isolation (.gitignore):
The local dataset folder containing student images (e.g., dataset/ or lfw/) is strictly excluded from version control via .gitignore .
No raw image files or private student photos are committed or pushed to GitHub.
Log Handling: Attendance results are generated locally into attendance.csv containing only Name, Date, and Time stamps.

Tech Stack & Dependencies

Python 3.x
InsightFace: Pretrained face-analysis models for face detection and facial embedding extraction, using the buffalo_l model pack.
ONNX Runtime: Executes the pretrained InsightFace models for face detection and recognition.
OpenCV (cv2): Image loading and computer vision operations
Pandas: Data handling and attendance CSV management
Matplotlib: Visualizing Exploratory Data Analysis (EDA) charts