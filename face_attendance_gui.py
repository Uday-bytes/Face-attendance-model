
import csv
import os
import queue
import re
import subprocess
import sys
import threading
import time
import tkinter as tk
import warnings
import cv2
import numpy as np
from numpy.linalg import norm
from PIL import Image, ImageTk
from insightface.app import FaceAnalysis
from datetime import datetime
from tkinter import messagebox, ttk

# Settings 
REGISTERED_DIR = "registered_faces"
UNKNOWN_DIR = "unknown_faces"       # unknown_faces/<YYYY-MM-DD>/unknown_<HH-MM-SS>.jpg
CSV_FILE = "attendance.csv"

THRESHOLD = 0.45          # cosine similarity needed to count as a registered face
MIN_DET_SCORE = 0.6       # ignore low-confidence detections
DETECT_INTERVAL = 0.3     # seconds between detection passes
UNKNOWN_COOLDOWN = 60     # don't re-save the same unknown person for this many seconds
SAME_PERSON_SIM = 0.5     # similarity above which two unknown faces are the same person

GREEN = (0, 200, 0)
RED = (0, 0, 255)
ORANGE = (0, 165, 255)


#  Helpers 
def compute_similarity(a, b):
    return float(np.dot(a, b) / (norm(a) * norm(b)))


def safe_name(name):
    """Remove characters that are illegal in filenames."""
    return re.sub(r'[\\/:*?"<>|]', "_", name).strip()


def largest_face(faces):
    return max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))


def ensure_csv():
    if not os.path.exists(CSV_FILE) or os.path.getsize(CSV_FILE) == 0:
        with open(CSV_FILE, "w", newline="") as f:
            csv.writer(f).writerow(["Name/ID", "Time"])


def read_today_records():
    """Return [(name, 'HH:MM:SS'), ...] for today, oldest first."""
    today = datetime.now().strftime("%Y-%m-%d")
    records = []
    if not os.path.exists(CSV_FILE):
        return records
    with open(CSV_FILE, newline="") as f:
        reader = csv.reader(f)
        next(reader, None)  # header
        for row in reader:
            if len(row) >= 2 and row[1].startswith(today):
                records.append((row[0], row[1][11:]))
    return records


def open_folder(path):
    os.makedirs(path, exist_ok=True)
    if sys.platform.startswith("win"):
        os.startfile(os.path.abspath(path))
    elif sys.platform == "darwin":
        subprocess.Popen(["open", path])
    else:
        subprocess.Popen(["xdg-open", path])


#  App 
class AttendanceApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Face Attendance System")
        self.resizable(False, False)
        self.protocol("WM_DELETE_WINDOW", self.on_close)

        os.makedirs(REGISTERED_DIR, exist_ok=True)
        os.makedirs(UNKNOWN_DIR, exist_ok=True)
        ensure_csv()

        # model / data
        self.fa = None
        self.known = {}                 # {name: embedding}
        self.model_state = "loading"    # "loading" | "ready" | "error"
        self.model_error = ""
        self.model_lock = threading.Lock()
        self.events = queue.Queue()

        # runtime state
        self.mode = None
        self.cap = None
        self.running = False
        self.busy = False
        self.finishing = False
        self.current_frame = None
        self.latest_results = []
        self.last_detect = 0.0
        self.marked = set()
        self.unknown_cache = []
        self.unknown_count = 0
        self.tick_job = None
        self.photo = None

        self._build_ui()
        threading.Thread(target=self._load_model, daemon=True).start()
        self.after(200, self._check_model)

    #  UI construction 
    def _build_ui(self):
        style = ttk.Style(self)
        style.configure("Title.TLabel", font=("Segoe UI", 20, "bold"))
        style.configure("Sub.TLabel", font=("Segoe UI", 14, "bold"))
        style.configure("Big.TButton", font=("Segoe UI", 14), padding=14)
        style.configure("Stop.TButton", font=("Segoe UI", 12, "bold"), padding=8)

        self.container = ttk.Frame(self, padding=16)
        self.container.pack()
        self.video_labels = {}

        self._build_home()
        self._build_attendance()
        self._build_register()
        self._show(self.home)

    def _show(self, frame):
        for f in (self.home, self.att, self.reg):
            f.pack_forget()
        frame.pack()

    def _build_home(self):
        self.home = ttk.Frame(self.container)
        ttk.Label(self.home, text="Face Attendance System", style="Title.TLabel").pack(pady=(10, 10))
        self.home_status = ttk.Label(self.home, text="Loading face model, please wait...")
        self.home_status.pack(pady=(0, 20))

        self.btn_attendance = ttk.Button(
            self.home, text="1.  Take Attendance", style="Big.TButton", width=28,
            state="disabled", command=lambda: self.start_mode("attendance"))
        self.btn_attendance.pack(pady=8)

        self.btn_register = ttk.Button(
            self.home, text="2.  Register New Faces", style="Big.TButton", width=28,
            state="disabled", command=lambda: self.start_mode("register"))
        self.btn_register.pack(pady=8)

        ttk.Button(self.home, text="Quit", command=self.on_close).pack(pady=(25, 10))

    def _build_attendance(self):
        self.att = ttk.Frame(self.container)
        video = ttk.Label(self.att)
        video.grid(row=0, column=0, sticky="n")
        self.video_labels["attendance"] = video

        side = ttk.Frame(self.att)
        side.grid(row=0, column=1, sticky="ns", padx=(14, 0))

        ttk.Label(side, text="Taking Attendance", style="Sub.TLabel").pack(anchor="w")
        self.att_status = ttk.Label(side, text="", wraplength=280, justify="left")
        self.att_status.pack(anchor="w", pady=(6, 6))

        self.present_var = tk.StringVar(value="Present today: 0")
        self.unknown_var = tk.StringVar(value="Unknown faces saved this session: 0")
        ttk.Label(side, textvariable=self.present_var).pack(anchor="w")
        ttk.Label(side, textvariable=self.unknown_var).pack(anchor="w", pady=(0, 8))

        ttk.Label(side, text="Marked today (newest first):").pack(anchor="w")
        list_frame = ttk.Frame(side)
        list_frame.pack(fill="both", expand=True)
        scroll = ttk.Scrollbar(list_frame)
        scroll.pack(side="right", fill="y")
        self.att_list = tk.Listbox(list_frame, width=34, height=13, yscrollcommand=scroll.set)
        self.att_list.pack(side="left", fill="both", expand=True)
        scroll.config(command=self.att_list.yview)

        ttk.Button(side, text="Open Unknown Faces Folder",
                   command=lambda: open_folder(UNKNOWN_DIR)).pack(fill="x", pady=(10, 4))
        ttk.Button(side, text="STOP Attendance", style="Stop.TButton",
                   command=self.stop_mode).pack(fill="x", pady=(4, 0))

    def _build_register(self):
        self.reg = ttk.Frame(self.container)
        video = ttk.Label(self.reg)
        video.grid(row=0, column=0, sticky="n")
        self.video_labels["register"] = video

        side = ttk.Frame(self.reg)
        side.grid(row=0, column=1, sticky="ns", padx=(14, 0))

        ttk.Label(side, text="Register New Face", style="Sub.TLabel").pack(anchor="w")
        ttk.Label(side, text="Student name / ID:").pack(anchor="w", pady=(10, 2))
        self.name_var = tk.StringVar()
        entry = ttk.Entry(side, textvariable=self.name_var, width=32)
        entry.pack(anchor="w")
        entry.bind("<Return>", lambda e: self.capture_register())

        ttk.Button(side, text="Capture && Register", command=self.capture_register
                   ).pack(fill="x", pady=(10, 4))
        self.reg_status = ttk.Label(side, text="", wraplength=280, justify="left")
        self.reg_status.pack(anchor="w", pady=(4, 8))

        ttk.Label(side, text="Registered students:").pack(anchor="w")
        list_frame = ttk.Frame(side)
        list_frame.pack(fill="both", expand=True)
        scroll = ttk.Scrollbar(list_frame)
        scroll.pack(side="right", fill="y")
        self.reg_list = tk.Listbox(list_frame, width=34, height=11, yscrollcommand=scroll.set)
        self.reg_list.pack(side="left", fill="both", expand=True)
        scroll.config(command=self.reg_list.yview)

        ttk.Button(side, text="Back to Menu", style="Stop.TButton",
                   command=self.stop_mode).pack(fill="x", pady=(10, 0))

    #  Model loading 
    def _load_model(self):
        try:
            fa = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
            fa.prepare(ctx_id=-1, det_size=(640, 640))  # -1 = CPU

            known = {}
            for filename in sorted(os.listdir(REGISTERED_DIR)):
                if filename.lower().endswith((".png", ".jpg", ".jpeg")):
                    img = cv2.imread(os.path.join(REGISTERED_DIR, filename))
                    if img is None:
                        continue
                    faces = fa.get(img)
                    if faces:
                        known[os.path.splitext(filename)[0]] = largest_face(faces).normed_embedding

            self.fa = fa
            self.known = known
            self.model_state = "ready"
        except Exception as e:
            self.model_error = str(e)
            self.model_state = "error"

    def _check_model(self):
        if self.model_state == "ready":
            self.btn_attendance.config(state="normal")
            self.btn_register.config(state="normal")
            self._refresh_home()
        elif self.model_state == "error":
            self.home_status.config(text=f"Failed to load face model:\n{self.model_error}")
        else:
            self.after(200, self._check_model)

    def _refresh_home(self):
        self.home_status.config(text=f"Ready.  Registered students: {len(self.known)}")

    #  Start / stop 
    def start_mode(self, mode):
        if self.model_state != "ready" or self.running:
            return

        backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY
        cap = cv2.VideoCapture(0, backend)
        if not cap.isOpened():
            messagebox.showerror("Camera", "Could not open the webcam.")
            return
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.cap = cap
        self.mode = mode
        self.latest_results = []
        self.last_detect = 0.0
        self.video_labels[mode].configure(image="")

        if mode == "attendance":
            records = read_today_records()
            self.marked = {name for name, _ in records}
            self.att_list.delete(0, "end")
            for name, t in records:
                self.att_list.insert(0, f"{t}   {name}")
            self.unknown_cache = []
            self.unknown_count = 0
            self.present_var.set(f"Present today: {len(self.marked)}")
            self.unknown_var.set("Unknown faces saved this session: 0")
            self.att_status.config(text="Camera running. Press STOP when finished.")
            self._show(self.att)
        else:
            self._refresh_registered_list()
            self.name_var.set("")
            self.reg_status.config(text="Face the camera, type a name, then press Capture.")
            self._show(self.reg)

        # discard leftover messages from a previous session
        while not self.events.empty():
            try:
                self.events.get_nowait()
            except queue.Empty:
                break

        self.finishing = False
        self.running = True
        self._tick()

    def stop_mode(self, message=None):
        self.running = False
        self.mode = None
        if self.tick_job is not None:
            self.after_cancel(self.tick_job)
            self.tick_job = None
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.latest_results = []
        self.current_frame = None
        self._refresh_home()
        if message:
            self.home_status.config(
                text=f"{message}\nRegistered students: {len(self.known)}")
        self._show(self.home)

    def on_close(self):
        if self.running:
            self.stop_mode()
        self.destroy()

    #  Video loop 
    def _tick(self):
        if not self.running:
            return

        ok, frame = self.cap.read()
        if not ok:
            self.stop_mode()
            messagebox.showerror("Camera", "Lost connection to the webcam.")
            return
        self.current_frame = frame
        self._drain_events()
        if not self.running:   # a "finished" event closed the camera
            return

        now = time.time()
        if (not self.busy and not self.finishing
                and now - self.last_detect >= DETECT_INTERVAL):
            self.busy = True
            self.last_detect = now
            threading.Thread(target=self._detect_worker,
                             args=(frame.copy(), self.mode), daemon=True).start()

        display = frame.copy()
        for (x1, y1, x2, y2), label, color in self.latest_results:
            cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
            if label:
                cv2.putText(display, label, (x1, max(y1 - 10, 15)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        self.photo = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.video_labels[self.mode].configure(image=self.photo)

        self.tick_job = self.after(15, self._tick)

    def _drain_events(self):
        while True:
            try:
                event = self.events.get_nowait()
            except queue.Empty:
                break
            kind = event[0]
            if kind == "marked":
                _, name, t = event
                self.att_list.insert(0, f"{t}   {name}")
                self.present_var.set(f"Present today: {len(self.marked)}")
                self.att_status.config(text=f"Attendance marked: {name}")
            elif kind == "unknown":
                self.unknown_count += 1
                self.unknown_var.set(f"Unknown faces saved this session: {self.unknown_count}")
                self.att_status.config(text=f"Unknown face saved: {os.path.basename(event[1])}")
            elif kind == "finished":
                if self.running and self.mode == "attendance":
                    self.stop_mode(event[1])
            elif kind == "error":
                label = self.att_status if self.mode == "attendance" else self.reg_status
                label.config(text=f"Error: {event[1]}")

    #  Detection worker (background thread) 
    def _detect_worker(self, frame, mode):
        try:
            with self.model_lock:
                faces = [f for f in self.fa.get(frame) if f.det_score >= MIN_DET_SCORE]

            if mode != self.mode or not self.running:
                return  # user already left this mode

            if mode == "attendance":
                self.latest_results = self._process_attendance(frame, faces)
            else:
                color = GREEN if len(faces) == 1 else ORANGE
                self.latest_results = [(f.bbox.astype(int), "", color) for f in faces]
        except Exception as e:
            self.events.put(("error", str(e)))
        finally:
            self.busy = False

    def _match(self, embedding):
        best_name, best_sim = None, -1.0
        for name, known_emb in self.known.items():
            sim = compute_similarity(embedding, known_emb)
            if sim > best_sim:
                best_name, best_sim = name, sim
        if best_sim >= THRESHOLD:
            return best_name, best_sim
        return None, best_sim

    def _process_attendance(self, frame, faces):
        results = []
        messages = []
        now = datetime.now()
        for face in faces:
            box = face.bbox.astype(int)
            name, sim = self._match(face.normed_embedding)

            if name is None:
                results.append((box, "Unknown", RED))
                self._handle_unknown(frame, face, box, now)
                messages.append("Unknown face saved for review")
            else:
                results.append((box, f"{name} ({sim:.2f})", GREEN))
                if name not in self.marked:
                    stamp = now.strftime("%Y-%m-%d %H:%M:%S")
                    with open(CSV_FILE, "a", newline="") as f:
                        csv.writer(f).writerow([name, stamp])
                    self.marked.add(name)
                    self.events.put(("marked", name, stamp[11:]))
                    messages.append(f"Attendance marked: {name}")
                else:
                    messages.append(f"{name} was already marked today")

        # A face was handled: tell the GUI thread to close the camera
        if messages:
            self.finishing = True
            self.events.put(("finished", "; ".join(messages)))
        return results

    def _handle_unknown(self, frame, face, box, now):
        """Save the frame for manual review, once per unknown person per cooldown."""
        t = time.time()
        emb = face.normed_embedding

        self.unknown_cache = [c for c in self.unknown_cache if t - c[1] < UNKNOWN_COOLDOWN]
        for cached in self.unknown_cache:
            if compute_similarity(emb, cached[0]) >= SAME_PERSON_SIM:
                cached[1] = t   # still the same person in front of the camera
                return
        self.unknown_cache.append([emb, t])

        folder = os.path.join(UNKNOWN_DIR, now.strftime("%Y-%m-%d"))
        os.makedirs(folder, exist_ok=True)
        base = now.strftime("unknown_%H-%M-%S")
        path = os.path.join(folder, base + ".jpg")
        i = 1
        while os.path.exists(path):
            path = os.path.join(folder, f"{base}_{i}.jpg")
            i += 1

        img = frame.copy()
        x1, y1, x2, y2 = box
        cv2.rectangle(img, (x1, y1), (x2, y2), RED, 2)
        cv2.imwrite(path, img)
        self.events.put(("unknown", path))

    #  Registration
    def _refresh_registered_list(self):
        self.reg_list.delete(0, "end")
        for name in sorted(self.known, key=str.lower):
            self.reg_list.insert("end", name)

    def capture_register(self):
        if self.mode != "register" or self.current_frame is None:
            return

        name = safe_name(self.name_var.get())
        if not name:
            self.reg_status.config(text="Please enter a name / ID first.")
            return

        frame = self.current_frame.copy()
        with self.model_lock:
            faces = [f for f in self.fa.get(frame) if f.det_score >= MIN_DET_SCORE]

        if len(faces) == 0:
            self.reg_status.config(text="No face detected. Try again.")
            return
        if len(faces) > 1:
            self.reg_status.config(text="Multiple faces detected. Only one person at a time.")
            return

        embedding = faces[0].normed_embedding

        match_name, sim = self._match(embedding)
        if match_name and match_name != name:
            if not messagebox.askyesno(
                    "Possible duplicate",
                    f"This face looks like the already registered '{match_name}' "
                    f"(similarity {sim:.2f}).\n\nRegister as '{name}' anyway?"):
                return
        if name in self.known:
            if not messagebox.askyesno("Overwrite", f"'{name}' is already registered. Replace?"):
                return

        if not cv2.imwrite(os.path.join(REGISTERED_DIR, f"{name}.jpg"), frame):
            self.reg_status.config(text="Could not save the image. Try a different name.")
            return

        self.known[name] = embedding
        self.name_var.set("")
        self._refresh_registered_list()
        self.reg_status.config(text=f"Successfully registered {name}!")


if __name__ == "__main__":
    AttendanceApp().mainloop()