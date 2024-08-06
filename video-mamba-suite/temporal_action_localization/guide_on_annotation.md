# Quick Guide to Annotate Your Own Tennis Videos

## Step 1: Running the Inference Pipeline

To start annotating your tennis videos, follow these steps:

1. Run the `inference_pipeline_video_tennis` script with the following options:
   - `--create_json`: Generates a JSON file with annotations.
   - `--create_english_video` or `--create_french_video`: Creates a video with annotations in English or French.
   - `--add_frame_number`: Adds frame numbers to the annotations.

   ```bash
   python inference_pipeline_video_tennis --create_json --create_english_video --add_frame_number

## Step 2: Reviewing and Modifying Annotations
Use the VLC Media Player to review the videos frame by frame. Inspect the annotations and manually modify them if necessary.

## Step 3: Correcting Frame Numbers
After reviewing and modifying the annotations,Run the Correct_segments.py script to correct the frame numbers automatically, eliminating the need for manual calculations