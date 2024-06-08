import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import torch
import cv2
import numpy as np
import threading

# MQTT lib
import time
import psutil
import paho.mqtt.client as mqtt
from prometheus_client import start_http_server, Counter, Summary, Gauge

# Prometheus metrics
frame_processing_rate = Counter('frame_processing_rate', 'Number of frames processed per second')
detection_count = Counter('detection_count', 'Number of objects detected')
detection_latency = Summary('detection_latency_seconds', 'Time taken to process each frame')
average_confidence = Summary('average_confidence', 'Average confidence score of detections')
memory_usage = Gauge('memory_usage_percent', 'Memory usage of the YOLO algorithm')
cpu_usage = Gauge('cpu_usage_percent', 'CPU usage of the YOLO algorithm')
error_count = Counter('error_count', 'Number of frames that failed to process')
connection_status = Gauge('mqtt_connection_status', 'MQTT connection status (1 for connected, 0 for disconnected)')

# Start Prometheus metrics server
start_http_server(5555)

# IP address and port of MQTT Broker (Mosquitto MQTT)
broker = "10.8.1.6"
port = 1883
topic = "/data"

def on_connect(client, userdata, flags, reasonCode, properties=None):
    if reasonCode == 0:
        print("Connected to MQTT Broker successfully.")
        connection_status.set(1)  # Set connection status to connected
    else:
        print(f"Failed to connect to MQTT Broker. Reason: {reasonCode}")
        connection_status.set(0)  # Set connection status to disconnected

def on_disconnect(client, userdata, rc):
    print(f"Disconnected from MQTT Broker. Reason: {rc}")
    connection_status.set(0)  # Set connection status to disconnected

producer = mqtt.Client(client_id="producer_1", callback_api_version=mqtt.CallbackAPIVersion.VERSION2)

# Connect to MQTT broker
producer.connect(broker, port, 60)
producer.loop_start()  # Start a new thread to handle network traffic and dispatching callbacks

# Setup MQTT client
producer.on_connect = on_connect
producer.on_disconnect = on_disconnect

# Load the models
modelDanger = torch.hub.load('ultralytics/yolov5', 'custom', path='Yolo/DangerBest.pt')
modelRoad = torch.hub.load('ultralytics/yolov5', 'custom', path='Yolo/RoadBest.pt')

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
            start_time = time.time()
            try:
                # Convert the color from BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Run detection using both models
                results_danger = modelDanger(frame)
                results_road = modelRoad(frame)

                # Render detections
                results_danger.render()
                results_road.render()

                # Update the detection results label
                detected_classes = []
                total_detections = 0

                if results_danger.pred[0] is not None:
                    detected_classes.extend(results_danger.names[int(cls)] for cls in results_danger.pred[0][:, -1])
                    total_detections += len(results_danger.pred[0])
                    for detection in results_danger.pred[0]:
                        average_confidence.observe(detection[-2])
                    detection_count.inc(len(results_danger.pred[0]))

                if results_road.pred[0] is not None:
                    detected_classes.extend(results_road.names[int(cls)] for cls in results_road.pred[0][:, -1])
                    total_detections += len(results_road.pred[0])
                    for detection in results_road.pred[0]:
                        average_confidence.observe(detection[-2])
                    detection_count.inc(len(results_road.pred[0]))

                detection_label.config(text="Detected: " + ", ".join(set(detected_classes)) if detected_classes else "Detected: None")
                
                # Convert array to Image
                frame_image = Image.fromarray(frame)
                frame_resized = resize_image(np.array(frame_image))
                photo = ImageTk.PhotoImage(image=Image.fromarray(frame_resized))
                canvas.create_image(20, 20, anchor='nw', image=photo)

                # Prometheus metrics
                frame_processing_rate.inc()
                detection_latency.observe(time.time() - start_time)

            except Exception as e:
                error_count.inc()
                print(f"Error processing frame: {e}")

            window.after(64, detect_objects)
        else:
            cap.release()
            status_bar.config(text="Video ended")

def update_system_metrics():
    """
    Updates system metrics on prometheus (CPU and memory)
    """
    while True:
        mem_percent = psutil.virtual_memory().percent
        memory_usage.set(mem_percent)

        cpu_percent = psutil.cpu_percent(interval=1)
        cpu_usage.set(cpu_percent)

def show_about():
    """
    Show an about message box.
    """
    messagebox.showinfo("About", "This application detects dangers on the road using YOLOv5 models.")

# Create the main window
window = tk.Tk()
window.title("Danger on the Road Detection")
window.geometry("820x680")  # Adjust window size to include status bar

# Create a menu bar
menu_bar = tk.Menu(window)

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

# Label to display detected objects
detection_label = tk.Label(window, text="Detected: None", bd=1, relief=tk.SUNKEN, anchor=tk.W)
detection_label.pack(side=tk.TOP, fill=tk.X)

# Create a status bar to display information
status_bar = tk.Label(window, text="Status: Ready", bd=1, relief=tk.SUNKEN, anchor=tk.W)
status_bar.pack(side=tk.BOTTOM, fill=tk.X)

# Start a separate thread to continuously update system metrics
metrics_thread = threading.Thread(target=update_system_metrics)
metrics_thread.daemon = True
metrics_thread.start()

window.mainloop()