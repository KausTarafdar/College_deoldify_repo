import gradio as gr
import os
import time
import cv2
import numpy as np
from tqdm import tqdm
import tempfile
import shutil

# Create directories for uploaded and processed videos
UPLOAD_DIR = "uploaded_videos"
PROCESSED_DIR = "processed_videos"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

def apply_ml_processing(video_path, progress=None):
    """
    Example ML function that applies a simple effect to the video.
    In a real application, you would replace this with your actual ML model.

    Args:
        video_path (str): Path to the input video
        progress (gr.Progress, optional): Progress bar

    Returns:
        str: Path to the processed video
    """
    # Open the video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Could not open the video")

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Create output video filename
    video_name = os.path.basename(video_path)
    output_path = os.path.join(PROCESSED_DIR, f"processed_{video_name}")

    # Create VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use appropriate codec
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Process the video frame by frame
    if progress is not None:
        progress(0, desc="Starting video processing...")

    for i in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break

        # Apply a simple effect (example ML processing)
        # Edge detection as a simple visual effect
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        # Convert back to 3-channel to match original frame dimensions
        edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

        # Ensure dimensions match before blending
        if frame.shape == edges_colored.shape:
            # Blend the original frame with the edge detection
            processed_frame = cv2.addWeighted(frame, 0.7, edges_colored, 0.3, 0)
        else:
            # If dimensions don't match, just use the original frame
            print(f"Warning: Frame dimensions don't match. Using original frame.")
            processed_frame = frame

        # Write the frame
        out.write(processed_frame)

        # Update progress
        if progress is not None:
            progress((i + 1) / total_frames, desc=f"Processing frame {i+1}/{total_frames}")

        # Simulate some additional processing time
        time.sleep(0.01)

    # Release resources
    cap.release()
    out.release()

    return output_path

def process_video(video, progress=gr.Progress()):
    """
    Process the uploaded video

    Args:
        video (str): Path to the uploaded video
        progress (gr.Progress): Gradio progress bar

    Returns:
        str: Path to the processed video
    """
    if video is None:
        return None

    # Save the uploaded video
    video_name = os.path.basename(video)
    saved_path = os.path.join(UPLOAD_DIR, video_name)

    # Copy to our upload directory
    shutil.copy(video, saved_path)
    progress(0.1, desc="Video uploaded and saved")

    # Process the video
    try:
        processed_path = apply_ml_processing(saved_path, progress)
        progress(1.0, desc="Processing complete!")
        return processed_path
    except Exception as e:
        print(f"Error processing video: {e}")
        return None

# Create the Gradio interface
with gr.Blocks() as app:
    gr.Markdown("# Video Processing with ML")

    with gr.Row():
        with gr.Column():
            # Input components
            video_input = gr.Video(label="Upload a video")
            process_btn = gr.Button("Process Video", variant="primary")

        with gr.Column():
            # Output components
            output_video = gr.Video(label="Processed Video")
            status = gr.Textbox(label="Status", value="Upload a video and click 'Process Video'")

    # Define the processing flow
    def on_process_click(video):
        if video is None:
            return None, "Please upload a video first"
        return None, "Processing started... please wait"

    # Define callbacks for different stages
    def on_process_complete(output_video):
        if output_video is not None:
            return "Processing complete! The processed video is displayed above."
        else:
            return "An error occurred during processing."

    # Set up the processing pipeline without using .error()
    process_btn.click(
        fn=on_process_click,
        inputs=[video_input],
        outputs=[output_video, status],
        queue=False
    ).then(
        fn=process_video,
        inputs=[video_input],
        outputs=[output_video],
        queue=True
    ).then(
        fn=on_process_complete,
        inputs=[output_video],
        outputs=[status]
    )

    gr.Markdown("""
    ## How it works
    1. Upload your video using the upload component
    2. Click the "Process Video" button
    3. The video will be processed with our ML algorithm
    4. The progress bar will show the processing status
    5. Once complete, the processed video will be displayed
    """)

# Launch the app
if __name__ == "__main__":
    app.launch()