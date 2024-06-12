import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.collections import LineCollection
import csv
import zipfile
import threading
import glob

# Load the models
modelDanger = torch.hub.load('ultralytics/yolov5', 'custom', path='Yolo/DangerBest.pt')
modelRoad = torch.hub.load('ultralytics/yolov5', 'custom', path='Yolo/RoadBest.pt')

def extract_zip():
    zip_path = filedialog.askopenfilename(filetypes=[("ZIP", "*.zip")])
    if not zip_path:
        return

    extract_to = "ZIP_ex"
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    
    video_files = glob.glob(os.path.join(extract_to, "*.mp4"))
    csv_files = glob.glob(os.path.join(extract_to, "*.csv"))

    if video_files and csv_files:
        # Create threads for video and CSV processing
        video_thread = threading.Thread(target=open_video, args=(1, video_files[0]))
        csv_thread = threading.Thread(target=open_csv, args=(1, csv_files[0]))

        # Start the threads
        video_thread.start()
        csv_thread.start()
    else:
        messagebox.showinfo("Error", "No video or CSV file found in the ZIP.")

def resize_image(image, max_size=(800, 600)):
    h, w = image.shape[:2]
    ratio = min(max_size[0] / w, max_size[1] / h)
    new_size = (int(w * ratio), int(h * ratio))
    resized_image = cv2.resize(image, new_size, interpolation=cv2.INTER_LANCZOS4)
    return resized_image

def open_video(openMethod: int = 0, file_path = ""):
    global cap
    if openMethod == 0:     
        file_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4;*.avi;*.mov;*.mkv")])
    if file_path:
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            status_bar.config(text="Failed to load video")
            messagebox.showinfo("Error", "Failed to load video.")
            return
        
        start_event.wait()  

        window.title(f"Danger on the Road Detection - {file_path}")
        status_bar.config(text="Video loaded: " + file_path)
        detect_objects()
    else:
        status_bar.config(text="No video selected")
        messagebox.showinfo("Information", "No video file selected.")

def detect_objects():
    global cap, canvas, window, photo
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results_danger = modelDanger(frame)
            results_road = modelRoad(frame)

            if var1.get():
                results_danger.render()
            if var2.get(): 
                results_road.render()

            detected_classes = []
            if results_danger.pred[0] is not None:
                detected_classes.extend(results_danger.names[int(cls)] for cls in results_danger.pred[0][:, -1])
            if results_road.pred[0] is not None:
                detected_classes.extend(results_road.names[int(cls)] for cls in results_road.pred[0][:, -1])
            detection_label.config(text="Detected: " + ", ".join(set(detected_classes)) if detected_classes else "Detected: None")

            frame_image = Image.fromarray(frame)
            frame_resized = resize_image(np.array(frame_image))
            photo = ImageTk.PhotoImage(image=Image.fromarray(frame_resized))
            canvas.create_image(20, 20, anchor='nw', image=photo)
            window.after(64, detect_objects)
        else:
            cap.release()
            status_bar.config(text="Video ended")

def show_about():
    messagebox.showinfo("About", "This application detects dangers on the road and type of the road using YOLOv5 models.")

def open_csv(openMethod: int = 0, file_path = ""):
    if openMethod == 0:
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    elif openMethod == 1:
        file_path = file_path
        
    if file_path:
        # Create a thread for loading the CSV file
        csv_thread = threading.Thread(target=load_csv, args=(file_path,))
        csv_thread.start()

def load_csv(file_path):
    global gyroscope_y_values, timestamps, num_segments, ani
    gyroscope_y_values = []
    timestamps = []
    with open(file_path, "r") as csv_file:
        csv_reader = csv.DictReader(csv_file, delimiter=';')
        start_event.set()
        for row in csv_reader:
            timestamps.append(float(row['Timestamp']))
            gyroscope_y_values.append(float(row['GyroscopeY']))
    gyroscope_y_values = np.array(gyroscope_y_values)
    timestamps = np.array(timestamps)
    num_segments = len(gyroscope_y_values) // 20

    ani = FuncAnimation(fig, update, frames=range(0, num_segments), interval=1000)
    window.after(0, canvas_plot.draw)

def update(frame):
    if frame is not None:
        if frame == 0:
            start_idx = 0
        else:
            start_idx = frame * 20 - 1
        end_idx = (frame + 1) * 20
        segment_data = gyroscope_y_values[start_idx:end_idx]
        segment_time = timestamps[start_idx:end_idx]

        color = 'green'
        if np.any(np.abs(segment_data) >= 2):
            color = 'red'
        elif np.any(np.abs(segment_data) >= 0.25):
            color = 'orange'

        points = np.array([segment_time, segment_data]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = LineCollection(segments, colors=color, linewidths=3, linestyle='solid')
        ax.add_collection(lc)

        ax.set_ylim(-5, 5)

        current_time = timestamps[end_idx - 1]
        ax.set_xlim(max(0, current_time - 10), current_time)
        ax.set_xticks(np.arange(max(0, current_time - 10), current_time + 1, 1))
        ax.set_xticklabels(np.arange(max(0, current_time - 10), current_time + 1, 1).astype(int))
    window.after(0, canvas_plot.draw)

# Create the main window
window = tk.Tk()
window.title("Danger on the Road Detection")
window.geometry("1200x900")
window.minsize(1200, 900)

start_event = threading.Event()

# Create a menu bar
menu_bar = tk.Menu(window)

# Variables to store the state of checkboxes
var1 = tk.BooleanVar()
var2 = tk.BooleanVar()

# Create checkboxes
checkbox1 = tk.Checkbutton(window, text="Show possible danger", variable=var1)
checkbox2 = tk.Checkbutton(window, text="Show type of the road", variable=var2)

# Create the File menu
file_menu = tk.Menu(menu_bar, tearoff=0)
file_menu.add_command(label="Open Video", command=open_video)
file_menu.add_command(label="Open CSV", command=open_csv)
file_menu.add_command(label="Import ZIP", command=extract_zip)
file_menu.add_separator()
file_menu.add_command(label="Exit", command=window.quit)
menu_bar.add_cascade(label="File", menu=file_menu)

# Create the Help menu
help_menu = tk.Menu(menu_bar, tearoff=0)
help_menu.add_command(label="About", command=show_about)
menu_bar.add_cascade(label="Help", menu=help_menu)

# Display the menu bar
window.config(menu=menu_bar)

# Create a frame for the video and controls
left_frame = tk.Frame(window)
left_frame.grid(row=0, column=0, sticky="nsew")

# Create a canvas to display the video
canvas = tk.Canvas(left_frame, bg="black", width=800, height=600)
canvas.grid(row=0, column=0, rowspan=2, padx=10, pady=10)

# Create checkboxes for the detection options
checkbox1.grid(row=1, column=0, sticky="nw", padx=10, pady=5)
checkbox2.grid(row=2, column=0, sticky="nw", padx=10, pady=5)

# Create a label to display the detected classes
detection_label = tk.Label(left_frame, text="Detected: None", font=("Helvetica", 12))
detection_label.grid(row=3, column=0, sticky="nw", padx=10, pady=10)

# Create a status bar
status_bar = tk.Label(window, text="Select a video file", bd=1, relief=tk.SUNKEN, anchor=tk.W)
status_bar.grid(row=2, column=0, columnspan=2, sticky="ew")

# Add the plot area for the graph
fig, ax = plt.subplots()
fig.patch.set_facecolor('white')

canvas_plot = FigureCanvasTkAgg(fig, master=window)
canvas_plot.get_tk_widget().grid(row=0, column=1, sticky="nsew", padx=10, pady=10)

window.grid_columnconfigure(0, weight=1)
window.grid_columnconfigure(1, weight=1)
window.grid_rowconfigure(0, weight=1)

# Start the Tkinter event loop
window.mainloop()
