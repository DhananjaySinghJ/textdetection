import cv2
import easyocr
import numpy as np
import streamlit as st
from datetime import timedelta

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])

## Function to detect text in an image
def detect_text(image, region="full"):
    """
    This function detects text in an image based on the specified region.

    Args:
        image: The input image as a NumPy array.
        region (str, optional): The region of the image to analyze. 
            Can be "full" for the entire image or "subtitle" for the lower portion. 
            Defaults to "full".

    Returns:
        tuple: A tuple containing the extracted text and a flag indicating 
               if any text was detected.

    Raises:
        ValueError: If an invalid region is specified.
    """
    if region == "full":
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif region == "subtitle":
        height, _ = image.shape[:2]
        subtitle_region = image[int(3*height/4):height, :]
        gray = cv2.cvtColor(subtitle_region, cv2.COLOR_BGR2GRAY)
    else:
        raise ValueError("Invalid region specified")
    
    # Detect text using EasyOCR
    result = reader.readtext(gray, detail=0)
    text = " ".join(result)
    return text.strip(), len(text.strip()) > 0

## Function to analyze a single video frame
def analyze_frame(frame):
    """
    This function analyzes a single video frame to identify scene type based on text presence.

    Args:
        frame: The video frame as a NumPy array.

    Returns:
        tuple: A tuple containing the scene type ("Texted", "Semi-Textless", "Textless") 
               and the detected text (empty string if none).
    """
    full_text, has_full_text = detect_text(frame, "full")
    subtitle_text, has_subtitle = detect_text(frame, "subtitle")

    if has_full_text and not has_subtitle:
        return "Texted", full_text
    elif has_subtitle and not has_full_text:
        return "Semi-Textless", subtitle_text
    elif has_full_text and has_subtitle:
        return "Texted", full_text
    else:
        return "Textless", ""

## Function to analyze an entire video
def analyze_video(video_path, interval=1):
    """
    This function analyzes a video and returns a list of results for each frame at a specified interval.

    Args:
        video_path (str): The path to the video file.
        interval (int, optional): The interval (in seconds) between analyzed frames. Defaults to 1.

    Returns:
        list: A list of tuples containing (start_timestamp, end_timestamp, scene type, detected text) for each detected text segment.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Error opening video file")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    results = []
    text_start = None
    text_end = None
    text_type = None
    text_content = ""

    for frame_number in range(0, frame_count, int(fps * interval)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()

        if not ret:
            break

        timestamp = frame_number / fps
        scene_type, detected_text = analyze_frame(frame)

        if scene_type != "Textless":
            if text_start is None:
                text_start = timestamp
                text_type = scene_type
                text_content = detected_text
            else:
                text_content += " " + detected_text
            text_end = timestamp
        else:
            if text_start is not None:
                results.append((text_start, text_end, text_type, text_content.strip()))
                text_start = None
                text_type = None
                text_content = ""

        # Progress feedback
        st.progress(frame_number / frame_count)

    if text_start is not None:
        results.append((text_start, text_end, text_type, text_content.strip()))

    cap.release()
    return results

# Streamlit dashboard
st.title("Video Text Detection")

# Upload video file
uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov", "mkv"])

if uploaded_file is not None:
    video_path = f"/tmp/{uploaded_file.name}"
    with open(video_path, mode='wb') as f:
        f.write(uploaded_file.read())

    if st.button("Process Video"):
        st.write("Processing...")
        results = analyze_video(video_path)

        st.write(f"Results for {uploaded_file.name}:")
        for start_timestamp, end_timestamp, scene_type, text in results:
            start_time_str = str(timedelta(seconds=start_timestamp))
            end_time_str = str(timedelta(seconds=end_timestamp))
            st.write(f"From {start_time_str} to {end_time_str}, Scene Type: {scene_type}")
            if scene_type != "Textless":
                st.write(f"Detected Text: {text[:100]}...")
