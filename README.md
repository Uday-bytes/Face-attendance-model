# Face Recognition Attendance System

This project is a face recognition attendance prototype. We built a desktop app that marks class attendance from a webcam, saves unrecognised faces for a teacher to check, and wrote an exploratory data analysis (EDA) script to study the image data.

**Detailed explanations of every code block, the concepts used and the design choices are in the Jupyter notebook (`Face_Attendance_System.ipynb`).** This README is only a short summary of what was done.

## What we built

- A desktop app with a graphical interface (Tkinter) and two options: **Take Attendance** and **Register New Faces**.
- Registration: a student's face is captured from the webcam and saved with their name or ID. The app accepts only one face at a time and warns about possible duplicates before saving.
- Attendance: the app recognises registered faces and writes the name and time to `attendance.csv`. A student is marked only once per day.
- Unknown faces: faces that do not match anyone are saved as images in a dated folder (`unknown_faces/<date>/`) so a teacher can verify them manually. The same unknown person is not saved repeatedly within a short time.
- After a face is handled, the camera closes and the main menu stays available.
- Face detection runs in a background thread so the interface and live video stay responsive.

## How recognition works

We used InsightFace's pretrained `buffalo_l` models, so no model was trained from scratch. The detector finds faces, and the recognition model turns each face into an embedding, a list of 512 numbers. A live face is compared with all registered embeddings using cosine similarity. If the best match scores at least 0.45, the student is marked present. Otherwise the face is treated as unknown. Detections with a confidence below 0.6 are ignored.

## EDA

We wrote `EDA.py` using plot types from the Data Visualization lecture. It analyses:

- Image counts per person and how balanced the data is
- Lighting and image quality (brightness, contrast, sharpness)
- Failed detections and how they relate to image quality
- Face size and head angle of detected faces
- Correlations between the measurements
- Similarity between images of the same person and of different people, used to estimate false accepts and false rejects and to examine the 0.45 threshold

The script saves its plots, a text summary and CSV files in an output folder. It can anonymise names in its outputs. We practiced on the LFW (Labeled Faces in the Wild) dataset, obtained through Kaggle.

## Privacy

- Only consented images or a public dataset are used.
- Everything is processed locally on the computer and nothing is uploaded.
- Only a name and ID, one photo and the embedding are stored per student.
- Photos, `attendance.csv` and the unknown-faces folder are kept out of the repository through `.gitignore`.
- Unknown faces are reviewed by a person, and the system does not decide who they are.

## Files

- `face_attendance_gui.py` - the attendance app
- `EDA.py` - the EDA script
- `Face_Attendance_System.ipynb` - notebook with detailed explanations
- `requirements.txt` - Python packages used

The app creates `registered_faces/`, `attendance.csv` and `unknown_faces/` while it runs. These are not part of the repository.

## Limitations

- A photo or video of a face can fool the system, because there is no liveness check.
- Poor light, blur and side angles lower accuracy.
- Detection runs on the CPU, so it is slow.
- The 0.45 threshold is a common starting value and still needs checking on our own data.

## Credits

- InsightFace for face detection and recognition. Its pretrained models are intended for non-commercial research use, so check its license before any other use.
- LFW dataset, obtained through Kaggle. https://www.kaggle.com/datasets/jessicali9530/lfw-dataset
