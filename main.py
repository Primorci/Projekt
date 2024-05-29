import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import torch
import cv2
import numpy as np

# Load the models
modelDanger = torch.hub.load('ultralytics/yolov5', 'custom', path='Yolo\DangerBest.pt')
modelRoad = torch.hub.load('ultralytics/yolov5', 'custom', path='Yolo\RoadBest.pt')

def resize_image(image, max_size=(800, 600)):
    """
    Resize the image to fit within a specific size while maintaining the aspect ratio.
    """
    h, w = image.shape[:2]
    ratio = min(max_size[0] / w, max_size[1] / h)
    new_size = (int(w * ratio), int(h * ratio))
    resized_image = cv2.resize(image, new_size, interpolation=cv2.INTER_LANCZOS4)
    return resized_image

def open_video():
    """
    Open a video file using a file dialog and start object detection.
    """
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
    """
    Detect objects in the video frame by frame and display the results on the canvas.
    """
    global cap, canvas, window, photo
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            # Convert the color from BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Run detection using both models
            results_danger = modelDanger(frame)
            results_road = modelRoad(frame)

            # Render detections
            if var1.get():
                results_danger.render()

            if var2.get(): 
                results_road.render()

            # Update the detection results label
            detected_classes = []
            if results_danger.pred[0] is not None:
                detected_classes.extend(results_danger.names[int(cls)] for cls in results_danger.pred[0][:, -1])
            if results_road.pred[0] is not None:
                detected_classes.extend(results_road.names[int(cls)] for cls in results_road.pred[0][:, -1])
            detection_label.config(text="Detected: " + ", ".join(set(detected_classes)) if detected_classes else "Detected: None")

            # Convert array to Image
            frame_image = Image.fromarray(frame)
            frame_resized = resize_image(np.array(frame_image))
            photo = ImageTk.PhotoImage(image=Image.fromarray(frame_resized))
            canvas.create_image(20, 20, anchor='nw', image=photo)
            window.after(64, detect_objects)
        else:
            cap.release()
            status_bar.config(text="Video ended")

def show_about():
    """
    Show an about message box.
    """
    messagebox.showinfo("About", "This application detects dangers on the road and type of the road using YOLOv5 models.")

# Create the main window
window = tk.Tk()
window.title("Danger on the Road Detection")
window.geometry("820x680")  # Adjust window size to include status bar

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
file_menu.add_separator()
file_menu.add_command(label="Exit", command=window.quit)
menu_bar.add_cascade(label="File", menu=file_menu)

# Create the Help menu
help_menu = tk.Menu(menu_bar, tearoff=0)
help_menu.add_command(label="About", command=show_about)
menu_bar.add_cascade(label="Help", menu=help_menu)

# Display the menu bar
window.config(menu=menu_bar)

# Create a canvas to show the video frames
canvas = tk.Canvas(window, width=800, height=600)
canvas.pack()

# Place checkboxes in the window
checkbox1.pack(anchor='w', padx=10)
checkbox2.pack(anchor='w', padx=10)

# Label to display detected objects
detection_label = tk.Label(window, text="Detected: None", bd=1, relief=tk.SUNKEN, anchor=tk.W)
detection_label.pack(side=tk.TOP, fill=tk.X)

# Create a status bar to display information
status_bar = tk.Label(window, text="Status: Ready", bd=1, relief=tk.SUNKEN, anchor=tk.W)
status_bar.pack(side=tk.BOTTOM, fill=tk.X)

window.mainloop()
