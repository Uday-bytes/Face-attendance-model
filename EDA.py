import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from insightface.app import FaceAnalysis
import warnings

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Initialize the model to test for failed detections
app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
app.prepare(ctx_id=0, det_size=(320, 320))

dataset_path = "dataset"
results = []

image_counts = {}
failed_detections = 0
total_images = 0

print("Starting Exploratory Data Analysis (EDA)...\n")

for person_name in os.listdir(dataset_path):
    person_folder = os.path.join(dataset_path, person_name)
    
    if os.path.isdir(person_folder):
        images = [f for f in os.listdir(person_folder) if f.endswith(('.jpg', '.png', '.jpeg'))]
        image_counts[person_name] = len(images)
        
        for img_name in images:
            total_images += 1
            img_path = os.path.join(person_folder, img_name)
            img = cv2.imread(img_path)
            
            # 1. Calculate Lighting Variation (Brightness)
            # Convert to HSV (Hue, Saturation, Value) and take the mean of the Value (Brightness) channel
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            brightness = hsv[:, :, 2].mean()
            
            # 2. Check for Failed Detections
            faces = app.get(img)
            status = "Success"
            if len(faces) == 0:
                failed_detections += 1
                status = "Failed (No face found)"
            elif len(faces) > 1:
                status = "Failed (Multiple faces found)"
                failed_detections += 1
                
            results.append({
                "person": person_name,
                "image": img_name,
                "brightness": brightness,
                "status": status
            })

# --- PRINT EDA REPORT ---
print("-" * 40)
print("EDA SUMMARY REPORT")
print("-" * 40)
print(f"Total people registered: {len(image_counts)}")
print(f"Total images analyzed: {total_images}")
print(f"Failed Detections: {failed_detections} out of {total_images} images")
print("\nImage Counts per Person:")
for person, count in image_counts.items():
    print(f" - {person}: {count} images")

# --- PLOT THE DATA ---
plt.figure(figsize=(10, 5))

# Plot 1: Image counts per person
plt.subplot(1, 2, 1)
plt.bar(image_counts.keys(), image_counts.values(), color='skyblue')
plt.title('Image Counts per Person')
plt.xlabel('Person')
plt.ylabel('Number of Images')

# Plot 2: Lighting Variation (Brightness Distribution)
brightness_values = [res['brightness'] for res in results]
plt.subplot(1, 2, 2)
plt.hist(brightness_values, bins=10, color='orange', edgecolor='black')
plt.title('Lighting Variation (Brightness)')
plt.xlabel('Brightness Level (0=Dark, 255=Bright)')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()