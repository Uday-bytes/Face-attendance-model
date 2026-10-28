Face Recognition Attendance System

A desktop app that marks class attendance from a webcam. Students are registered once; when a registered face appears, attendance is written to a CSV file. Unknown faces are saved to a dated folder so a teacher can check them manually.

It uses InsightFace to turn each face into an embedding (a list of 512 numbers) and compares faces by cosine similarity. A face is accepted when similarity is at least 0.45, and detections below 0.6 confidence are ignored.

Privacy
Only people who gave consent, or a public dataset (LFW), are used.
Everything runs locally. Nothing is uploaded.
Photos, attendance.csv and the unknown-faces folder are not included in this repository.
Files
face_attendance_gui.py - the app (Take Attendance / Register New Faces)
EDA.py - exploratory data analysis of the image dataset
Face_Attendance_System.ipynb - notebook with each part explained
attendance.csv, registered_faces/, unknown_faces/<date>/ - created by the app
Setup and run

Requires Python 3.9+ and a webcam.

pip install insightface onnxruntime opencv-python numpy pillow pandas matplotlib seaborn
python face_attendance_gui.py

The first run downloads the InsightFace model, so it needs internet once.

Run the EDA on a folder of images:

python EDA.py --data path/to/images --min-images 5 --anonymize

Results are saved in eda_output/.

Limitations
A photo or video of a face can fool the system (no liveness check).
Poor light, blur and side angles lower accuracy.
Detection runs on the CPU, so it is slow.