# Danger on the Road Detection

This application detects dangers on the road using YOLOv5 models. The program loads video files and processes them frame by frame to identify and display detected objects related to road safety.

## Features

- Load and display video files.
- Perform object detection using pretrained YOLOv5 models for specific dangers and road objects.
- Display detected objects and their names in the video frames.
- User-friendly interface with a menu bar and status bar.

## Requirements

- Python 3.6 or higher
- Tkinter
- OpenCV
- PIL (Pillow)
- PyTorch
- YOLOv5 models

## Installation

1. **Clone the repository:**

    ```sh
    git clone https://github.com/Primorci/Projekt.git
    cd Projekt
    ```

2. **Install the required packages:**

    ```sh
    pip install tkinter pillow opencv-python torch
    ```

3. **Download the YOLOv5 models:**
    - Ensure you have the pretrained YOLOv5 model weights saved in the appropriate paths specified in the code:
      - `C:/Users/mihap/yolov5/runs/train/exp9/weights/best.pt`
      - `C:/Users/mihap/yolov5/runs/train/exp11/weights/best.pt`

## Usage

1. **Run the application:**

    ```sh
    python main.py
    ```

2. **Open a video file:**
    - Go to `File > Open Video` in the menu bar and select a video file (`.mp4`, `.avi`, `.mov`, `.mkv`).

3. **View detections:**
    - The application will process the video and display detected objects frame by frame.
    - Detected object names will be shown at the top of the window.

4. **About:**
    - Go to `Help > About` in the menu bar to see information about the application.

## Code Overview

- **Model Loading:**
    ```python
    modelDanger = torch.hub.load('ultralytics/yolov5', 'custom', path='C:/Users/mihap/yolov5/runs/train/exp9/weights/best.pt')
    modelRoad = torch.hub.load('ultralytics/yolov5', 'custom', path='C:/Users/mihap/yolov5/runs/train/exp11/weights/best.pt')
    ```

- **Resize Image Function:**
    ```python
    def resize_image(image, max_size=(800, 600)):
        ...
    ```

- **Open Video Function:**
    ```python
    def open_video():
        ...
    ```

- **Detect Objects Function:**
    ```python
    def detect_objects():
        ...
    ```

- **Show About Function:**
    ```python
    def show_about():
        ...
    ```

- **GUI Setup:**
    ```python
    window = tk.Tk()
    ...
    ```
    
## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
