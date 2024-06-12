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

# Load the models
modelDanger = torch.hub.load('ultralytics/yolov5', 'custom', path='Yolo/DangerBest.pt')
modelRoad = torch.hub.load('ultralytics/yolov5', 'custom', path='Yolo/RoadBest.pt')

def resize_image(image, max_size=(800, 600)):
    h, w = image.shape[:2]
    ratio = min(max_size[0] / w, max_size[1] / h)
    new_size = (int(w * ratio), int(h * ratio))
    resized_image = cv2.resize(image, new_size, interpolation=cv2.INTER_LANCZOS4)
    return resized_image

def open_video():
    global cap
    file_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4;*.avi;*.mov;*.mkv")])
    if file_path:
        cap = cv2.VideoCapture(file_path)
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

def open_csv():
    file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    if file_path:
        load_csv(file_path)

def load_csv(file_path):
    global gyroscope_y_values, timestamps, num_segments, ani, start_timestamp
    gyroscope_y_values = []
    timestamps = []
    with open(file_path, "r") as csv_file:
        csv_reader = csv.DictReader(csv_file, delimiter=';')
        for row in csv_reader:
            timestamps.append(float(row['Timestamp']))
            gyroscope_y_values.append(float(row['GyroscopeY']))
    gyroscope_y_values = np.array(gyroscope_y_values)
    timestamps = np.array(timestamps)
    start_timestamp = timestamps[0]  # Record the initial timestamp
    timestamps -= start_timestamp  # Convert timestamps to relative time
    num_segments = len(gyroscope_y_values) // 20

    ani = FuncAnimation(fig, update, frames=range(0, num_segments), interval=1000)
    canvas_plot.draw()

def update(frame):
    global red_lines, orange_lines
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

        current_time = segment_time[-1]
        ax.set_xlim(max(0, current_time - 10), current_time)
        ax.set_xticks(np.arange(max(0, current_time - 10), current_time + 1, 1))
        ax.set_xticklabels(np.arange(max(0, current_time - 10), current_time + 1, 1).astype(int))

        if var3.get():
            if red_lines is not None:
                for line in red_lines:
                    line.remove()
            if orange_lines is not None:
                for line in orange_lines:
                    line.remove()
                    
            red_lines = [ax.axhline(2, color='red', linestyle='--', alpha=0.8),
                         ax.axhline(-2, color='red', linestyle='--', alpha=0.8)]
            orange_lines = [ax.axhline(0.25, color='orange', linestyle='--', alpha=0.8),
                            ax.axhline(-0.25, color='orange', linestyle='--', alpha=0.8)]
        else:
            if red_lines is not None:
                for line in red_lines:
                    line.remove()
                red_lines = None
            if orange_lines is not None:
                for line in orange_lines:
                    line.remove()
                orange_lines = None

    canvas_plot.draw()

# Create the main window
window = tk.Tk()
window.title("Danger on the Road Detection")
window.geometry("1600x800")
window.minsize(1600, 800)

# Create a menu bar
menu_bar = tk.Menu(window)

# Variables to store the state of checkboxes
var1 = tk.BooleanVar()
var2 = tk.BooleanVar()
var3 = tk.BooleanVar()

# Create the File menu
file_menu = tk.Menu(menu_bar, tearoff=0)
file_menu.add_command(label="Open Video", command=open_video)
file_menu.add_command(label="Open CSV", command=open_csv)
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
left_frame = tk.Frame(window, bg="white", padx=10, pady=10)
left_frame.grid(row=0, column=0, sticky="nsew")

# Create a canvas to display the video
canvas = tk.Canvas(left_frame, bg="black", width=800, height=600)
canvas.grid(row=0, column=0, columnspan=3, padx=10, pady=10)

# Create checkboxes for the detection options
checkbox1 = tk.Checkbutton(left_frame, text="Show possible danger", variable=var1, bg="white")
checkbox2 = tk.Checkbutton(left_frame, text="Show type of the road", variable=var2, bg="white")
checkbox3 = tk.Checkbutton(left_frame, text="Show thresholds for the graph", variable=var3, bg="white", command=lambda: update(None))
checkbox1.grid(row=1, column=0, sticky="nw", padx=10, pady=5)
checkbox2.grid(row=1, column=1, sticky="nw", padx=10, pady=5)
checkbox3.grid(row=1, column=2, sticky="nw", padx=10, pady=5)

# Create a label to display the detected classes
detection_label = tk.Label(left_frame, text="Detected: None", font=("Helvetica", 12), bg="white")
detection_label.grid(row=2, column=0, columnspan=3, sticky="nw", padx=10, pady=10)

# Create a status bar
status_bar = tk.Label(window, text="Select a video file", bd=1, relief=tk.SUNKEN, anchor=tk.W)
status_bar.grid(row=2, column=0, columnspan=2, sticky="ew")

# Add the plot area for the graph
fig, ax = plt.subplots()
fig.patch.set_facecolor('white')

ax.set_ylabel('Gyroscope Y')
ax.set_xlabel('Seconds')

canvas_plot = FigureCanvasTkAgg(fig, master=window)
canvas_plot.get_tk_widget().grid(row=0, column=1, rowspan=2, sticky="nsew", padx=10, pady=10)

window.grid_rowconfigure(0, weight=1)
window.grid_columnconfigure(1, weight=1)

# Variables to store gyroscope data and animation
gyroscope_y_values = []
timestamps = []
num_segments = 0
red_lines = None
orange_lines = None
ani = None

# Run the application
window.mainloop()
