Here’s a sample `README.md` file for your **Video Analysis with YOLOv8 and DeepSORT** Streamlit application.

# Video Analysis with YOLOv8 and DeepSORT

This application allows users to analyze Videos by detecting and tracking objects in the video using **YOLOv8** and **DeepSORT**. It provides an intuitive web-based interface built with **Streamlit**, where users can upload video files, start the analysis, and download a PDF report summarizing the detected objects.

## Features

- Upload a video file for analysis (supports `.mp4`, `.avi`, `.mov` formats).
- Detect and track objects in the video using **YOLOv8** for detection and **DeepSORT** for tracking.
- Draw bounding boxes and object labels on the detected objects in the video.
- Generate a **PDF report** summarizing the detected objects and their counts.
- Download the PDF report after analysis.

## Installation

### Requirements

- Python 3.7+
- Libraries:
  - Streamlit
  - OpenCV
  - Ultralytics (YOLOv8)
  - DeepSORT (deep-sort-realtime)
  - ReportLab

### Step-by-Step Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/kpj307/videoanalysis.git
   ```

2. Create a virtual environment:

   ```bash
   python -m venv venv
   ```

3. Activate the virtual environment:

- On **Windows powershell**:
    ```bash
    venv\Scripts\activate.ps1
    ```
- On **macOS/Linux**:
    ```bash
    source venv/bin/activate
    ```

4. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

5. Download the YOLOv8 model:

   You will need to download the YOLOv8 weights manually from Ultralytics:

   ```bash
   yolo task=detect model=yolov8m.pt
   ```

   Place the `yolov8m.pt` model file in the root directory of the project.

## Usage

1. Run the Streamlit application:

   ```bash
   streamlit run app.py
   ```

2. In your browser, you should see the Streamlit app open.

3. Upload a video file in `.mp4`, `.avi`, or `.mov` format.

4. Click the **Analyze Video** button to start the analysis. This process will detect objects in the video, track them, and draw bounding boxes on the detected objects.

5. Once the analysis is complete, you will be able to download the **PDF report**.

## Example

After analyzing a video file, the app will:

- Display the uploaded video.
- Generate a PDF report like this:

```plaintext
Video Analysis Report

Object Counts:
person: 3
car: 5

Detected Events:
Object: person | Time: 2024-09-24 14:32:12 | Confidence: 0.94 | ID: 1
Object: car    | Time: 2024-09-24 14:32:20 | Confidence: 0.88 | ID: 2
```

## Folder Structure

```
streamlit/
├── app.py   # Main application script
├── requirements.txt             # Required dependencies
├── yolov8m.pt                   # YOLOv8 weights (to be downloaded)
└── README.md                    # Documentation file
```

## Dependencies

The `requirements.txt` file includes the following dependencies:

```
chardet==5.2.0
numpy==2.1.1
opencv-contrib-python==4.10.0.84
opencv-python==4.10.0.84
pillow==10.4.0
reportlab==4.2.2
ultralytics 
deep-sort-realtime
streamlit
```

You can install them using:

```bash
pip install -r requirements.txt
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments
```
- [Ultralytics YOLOv8](https://github.com/ultralytics/yolov8) for object detection.
- [DeepSORT](https://github.com/nwojke/deep_sort) for object tracking.
- [Streamlit](https://streamlit.io/) for building the web interface.
```

### Optional Enhancements:

1. Improve tracking with less ID switching for a more accurate report.
2. Detect suspecious activities.

This `README.md` should guide users on how to install, run, and understand your project.