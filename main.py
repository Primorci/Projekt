import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import torch
import cv2
import numpy as np

# Load the model
# model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model = torch.hub.load('ultralytics/yolov5', 'custom', path='C:/Users/mihap/yolov5/runs/train/exp9/weights/best.pt')


def resize_image(image, max_size=(800, 600)):
    """
    Resize the image to fit within a specific size while maintaining aspect ratio.
    """
    h, w = image.shape[:2]
    ratio = min(max_size[0] / w, max_size[1] / h)
    new_size = (int(w * ratio), int(h * ratio))
    resized_image = cv2.resize(image, new_size, interpolation=cv2.INTER_LANCZOS4)
    return resized_image


def open_video():
    global cap
    file_path = filedialog.askopenfilename()
    if file_path:
        cap = cv2.VideoCapture(file_path)
        window.title(f"Object Detection - {file_path}")
        detect_objects()


def detect_objects():
    global cap, canvas, window, photo
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            # Convert the color from BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Run detection
            results = model(frame)

            # Render detections
            results.render()

            # Convert array to Image
            frame_image = Image.fromarray(frame)
            frame_resized = resize_image(np.array(frame_image))
            photo = ImageTk.PhotoImage(image=Image.fromarray(frame_resized))
            canvas.create_image(20, 20, anchor='nw', image=photo)
            window.after(64, detect_objects)
        else:
            cap.release()


# Create the main window
window = tk.Tk()
window.title("Danger on the road detection")

# Create a canvas to show the image
canvas = tk.Canvas(window, width=800, height=600)
canvas.pack()

# Buttons for loading video and running detection
btn_load = tk.Button(window, text="Open Video", command=open_video)
btn_load.pack(side='left')

window.mainloop()