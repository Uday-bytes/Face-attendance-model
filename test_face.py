
import os
if not os.path.exists('registered_faces'):
    os.makedirs('registered_faces')

import cv2
import os
import csv
import numpy as np
import warnings
import tkinter as tk
from datetime import datetime
from numpy.linalg import norm
from insightface.app import FaceAnalysis
from tkinter import simpledialog

# Suppress InsightFace future warnings
warnings.filterwarnings("ignore", category=FutureWarning)

def compute_similarity(feat1, feat2):
    return np.dot(feat1, feat2) / (norm(feat1) * norm(feat2))

# 1. Setup Registration Directory
if not os.path.exists('registered_faces'):
    os.makedirs('registered_faces')

app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
app.prepare(ctx_id=0, det_size=(320, 320))

# 2. Load Existing Faces into Memory
known_embeddings = {}
print("Loading registered faces...")
for filename in os.listdir('registered_faces'):
    if filename.endswith(('.png', '.jpg', '.jpeg')):
        name = os.path.splitext(filename)[0]
        img_path = os.path.join('registered_faces', filename)
        img = cv2.imread(img_path)
        det_faces = app.get(img)
        if det_faces:
            known_embeddings[name] = det_faces[0].normed_embedding

# 3. Setup Attendance CSV
marked_students = set()
with open('attendance.csv', mode='a', newline='') as file:
    writer = csv.writer(file)
    if file.tell() == 0:
        writer.writerow(["Name/ID", "Time"])

camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

frame_count = 0
faces = []
threshold = 0.45 # Recognition threshold

root = tk.Tk()
root.withdraw()

print("System ready. Press 'r' to register a new face. Press 'q' to quit.")

while True:
    success, frame = camera.read()
    if not success:
        print("Could not access webcam")
        break

    frame_count += 1
    key = cv2.waitKey(1) & 0xFF

    # --- REGISTRATION MODE ---
    if key == ord('r'):
        reg_faces = app.get(frame)
        if len(reg_faces) == 1:
            name = simpledialog.askstring(
                "Register Face", 
                "Face detected! Enter the student's name/ID:",
                parent=root
            )
            if name and name.strip():
                name = name.strip()
                known_embeddings[name] = reg_faces[0].normed_embedding
                cv2.imwrite(os.path.join('registered_faces', f"{name}.jpg"), frame)
                print(f"Successfully registered {name}!")
            else:
                print("\nRegistration cancelled.")
        elif len(reg_faces) > 1:
            print("\nRegistration failed: Multiple faces detected.")
        else:
            print("\nRegistration failed: No face detected.")
        continue

    # --- DETECTION MODE (Optimized) ---
    if frame_count % 10 == 0:
        small_frame = cv2.resize(frame, (320, 240))
        faces = app.get(small_frame)

    display_frame = cv2.resize(frame, (640, 480))
    known_face_found = False

    # --- RECOGNITION & DRAWING ---
    for face in faces:
        x1, y1, x2, y2 = face.bbox.astype(int)
        live_embedding = face.normed_embedding
        
        best_match_name = "Unknown"
        highest_similarity = -1

        for name, known_emb in known_embeddings.items():
            similarity = compute_similarity(live_embedding, known_emb)
            if similarity > highest_similarity:
                highest_similarity = similarity
                best_match_name = name

        if highest_similarity >= threshold:
            # KNOWN FACE DETECTED: Log and trigger shutdown
            if best_match_name not in marked_students:
                current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                with open('attendance.csv', mode='a', newline='') as file:
                    csv.writer(file).writerow([best_match_name, current_time])
                marked_students.add(best_match_name)
                
            print(f"\n✅ Known face detected: {best_match_name}")
            print(f"Attendance registered at {current_time}. Closing camera.")
            known_face_found = True
            break  # Break out of the face-checking loop
            
        else:
            # UNKNOWN FACE: Draw red box and prompt
            cv2.rectangle(display_frame, (x1 * 2, y1 * 2), (x2 * 2, y2 * 2), (0, 0, 255), 2)
            cv2.putText(display_frame, "Unknown: Try again or press 'r'", (x1 * 2, y1 * 2 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # If a known face was found, break the main camera loop to exit
    if known_face_found:
        break

    cv2.imshow("Face Detection Test", display_frame)

    if key == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()
