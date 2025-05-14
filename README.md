# Video Processing App with ML

A simple Gradio-based application for processing videos with ML techniques.

## Overview

This application provides a user-friendly web interface for:

- Uploading videos
- Processing them with a customizable ML model
- Displaying a progress bar during processing
- Saving the processed result
- Showing the processed video in the interface

## Setup Instructions

### Prerequisites

- Python 3.6+
- pip package manager

### Installation

1. **Clone the repository:**

  ```bash
  git clone https://github.com/yourusername/video-processing-ml.git
  cd video-processing-ml
  ```

2. **Create and activate a virtual environment (recommended):**

  ```bash
  python -m venv venv

  # On Windows:
  venv\Scripts\activate

  # On macOS/Linux:
  source venv/bin/activate
  ```

3. **Install dependencies:**

  ```bash
  pip install gradio==3.50.2 opencv-python numpy tqdm
  ```

  Note: If you're running an older version of Gradio, make sure to specify the version.


### Running the Application

1. **Launch the app:**

  ```bash
  python main.py
  ```

2. **Access the interface:**

  - Open your web browser and go to http://127.0.0.1:7860 (or the URL shown in the terminal)

## Directory Structure

```
├── main.py          # Main application code
├── uploaded_videos/ # Directory for storing uploaded videos
└── processed_videos/ # Directory for storing processed videos
```

## Customizing the ML Model

To replace the sample edge detection with your own ML model:

1. Modify the `apply_ml_processing()` function in `main.py`
2. Replace the sample processing code with your model inference code
3. Make sure your model properly handles video frames and maintains the progress reporting

Example modification:

```python
# Import your model
from my_model import VideoProcessor

# Initialize your model
model = VideoProcessor()

# In the apply_ml_processing function:
processed_frame = model.process_frame(frame)
```

## Troubleshooting

- **OpenCV Errors:** If you encounter dimension mismatch errors, check that your video processing operations maintain the correct dimensions and channel count
- **Memory Issues:** For processing large videos, consider chunking the processing or reducing the resolution
- **Codec Problems:** If videos don't play in the interface, try changing the fourcc codec in the code (e.g., 'mp4v', 'avc1', 'XVID')

## License

[MIT License](LICENSE)