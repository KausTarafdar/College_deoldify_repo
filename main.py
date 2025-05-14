import gradio as gr
import os
import time
import cv2
import numpy as np
import shutil

# Create directories for uploaded and processed videos
UPLOAD_DIR = "uploaded_videos"
PROCESSED_DIR = "processed_videos"
DEMO_DIR = "demo_videos"  # Directory for pre-processed demo videos

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(DEMO_DIR, exist_ok=True)

def apply_ml_processing(video_path, progress=None):
    """
    Example ML function that applies a simple effect to the video.
    In a real application, you would replace this with your actual ML model.
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
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Process the video frame by frame
    if progress is not None:
        progress(0, desc="Starting video processing...")

    for i in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break

        # Apply a simple effect (example ML processing)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

        # Ensure dimensions match before blending
        if frame.shape == edges_colored.shape:
            processed_frame = cv2.addWeighted(frame, 0.7, edges_colored, 0.3, 0)
        else:
            processed_frame = frame

        # Write the frame
        out.write(processed_frame)

        # Update progress
        if progress is not None:
            progress((i + 1) / total_frames, desc=f"Processing frame {i+1}/{total_frames}")

        time.sleep(0.01)  # Simulate processing time

    # Release resources
    cap.release()
    out.release()

    return output_path

def demo_processing(video_path, progress=None):
    """
    Demo function that pretends to process the video but actually returns
    a pre-processed demo video from the demo directory
    """
    # For demo, we'll use a pre-processed video that's already in the demo directory
    # In a real app, you would have pre-processed demo videos ready

    # Find a demo video to use (or use a default one)
    demo_files = os.listdir(DEMO_DIR)

    # If no demo files exist yet, create a simple one (grayscale version)
    if not demo_files:
        print("No demo files found. Creating a default demo video...")
        # Create a simple grayscale demo video as the default
        video_name = os.path.basename(video_path)
        demo_output_path = os.path.join(DEMO_DIR, f"demo_processed_{video_name}")

        # Create a basic demo video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError("Could not open the video")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(demo_output_path, fourcc, fps, (width, height))

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Just make it grayscale for demo
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            colored_gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            out.write(colored_gray)

        cap.release()
        out.release()

        # Add to our demo files list
        demo_files = [f"demo_processed_{video_name}"]

    # Choose a demo file to use
    demo_file = demo_files[0]  # Just use the first one
    demo_output_path = os.path.join(DEMO_DIR, demo_file)

    # Simulate a processing delay with progress bar
    if progress is not None:
        # Simulate processing steps
        num_steps = 10
        for i in range(num_steps):
            progress((i + 1) / num_steps, desc=f"Demo Processing Step {i+1}/{num_steps}")
            time.sleep(0.5)  # Slow enough to see the progress

    return demo_output_path

def process_video(video, mode="test", progress=gr.Progress()):
    """
    Process the uploaded video based on selected mode
    """
    if video is None:
        return None

    # Save the uploaded video
    video_name = os.path.basename(video)
    saved_path = os.path.join(UPLOAD_DIR, video_name)

    # Copy to our upload directory
    shutil.copy(video, saved_path)
    progress(0.1, desc="Video uploaded and saved")

    # Process based on mode
    try:
        if mode == "demo":
            processed_path = demo_processing(saved_path, progress)
        else:  # test mode
            processed_path = apply_ml_processing(saved_path, progress)

        progress(1.0, desc="Processing complete!")
        return processed_path
    except Exception as e:
        print(f"Error processing video: {e}")
        return None

# Create the Gradio interface
with gr.Blocks() as app:
    gr.Markdown("# Video Processing with ML")

    # Store the current mode
    mode = gr.State("test")

    with gr.Row():
        with gr.Column():
            # Input components
            video_input = gr.Video(label="Upload a video")

            with gr.Row():
                demo_btn = gr.Button("Demo Mode", variant="secondary")
                test_btn = gr.Button("Test Mode", variant="primary")

        with gr.Column():
            # Output components
            output_video = gr.Video(label="Processed Video")
            status = gr.Textbox(label="Status", value="Upload a video and select a processing mode")

    # Mode selection functions
    def set_demo_mode():
        return "demo", "Demo mode selected. Upload a video and it will be quickly processed with a pre-made effect."

    def set_test_mode():
        return "test", "Test mode selected. Upload a video for real processing."

    # Define callbacks for processing
    def on_process_start(video, current_mode):
        if video is None:
            return None, f"Please upload a video first"

        mode_name = "demo" if current_mode == "demo" else "test"
        return None, f"{mode_name.capitalize()} processing started... please wait"

    def on_process_complete(output_video):
        if output_video is not None:
            return "Processing complete! The processed video is displayed above."
        else:
            return "An error occurred during processing."

    # Set up mode buttons
    demo_btn.click(
        fn=set_demo_mode,
        inputs=[],
        outputs=[mode, status]
    )

    test_btn.click(
        fn=set_test_mode,
        inputs=[],
        outputs=[mode, status]
    )

    # Set up processing for both buttons
    demo_btn.click(
        fn=on_process_start,
        inputs=[video_input, mode],
        outputs=[output_video, status],
        queue=False
    ).then(
        fn=process_video,
        inputs=[video_input, mode],
        outputs=[output_video],
        queue=True
    ).then(
        fn=on_process_complete,
        inputs=[output_video],
        outputs=[status]
    )

    test_btn.click(
        fn=on_process_start,
        inputs=[video_input, mode],
        outputs=[output_video, status],
        queue=False
    ).then(
        fn=process_video,
        inputs=[video_input, mode],
        outputs=[output_video],
        queue=True
    ).then(
        fn=on_process_complete,
        inputs=[output_video],
        outputs=[status]
    )

# Launch the app
if __name__ == "__main__":
    app.launch()