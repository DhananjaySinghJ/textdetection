import streamlit as st
import cv2
import pytesseract
from pytesseract import Output
import tempfile
import os
from concurrent.futures import ThreadPoolExecutor
import numpy as np

# Path to Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'/opt/homebrew/bin/tesseract'

def preprocess_frame(frame):
    resized_frame = cv2.resize(frame, (320, 180))  # Further reduce frame size
    gray_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
    _, thresh_frame = cv2.threshold(gray_frame, 150, 255, cv2.THRESH_BINARY)
    return thresh_frame

def extract_text_from_frame(frame):
    preprocessing_frame = preprocess_frame(frame)
    data = pytesseract.image_to_data(preprocessing_frame, config='--oem 1 --psm 6', output_type=Output.DICT)
    text = ' '.join([word for i, word in enumerate(data['text']) if float(data['conf'][i]) >= 70])
    return text.strip()

def classify_frame(text):
    word_count = len(text.split())
    if word_count == 0:
        return "textless"
    elif word_count < 5:
        return "semi-textless"
    else:
        return "texted"

@st.cache_data
def process_video(video_path, start_time_ms, target_fps):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error(f"Error: Could not open video {video_path}.")
        return
    
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = max(1, int(original_fps / target_fps))
    
    results = []
    frame_count = 0
    
    progress_bar = st.progress(0)
    progress_text = st.empty()
    
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_position_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            if frame_position_ms < start_time_ms:
                continue
            
            if frame_count % frame_interval == 0:
                future = executor.submit(extract_text_from_frame, frame)
                futures.append((frame_count, frame_position_ms, future))
            
            frame_count += 1
            
            # Update progress every 10 frames
            if frame_count % 10 == 0:
                progress = frame_count / total_frames
                progress_bar.progress(progress)
                progress_text.text(f"Processing frame {frame_count} of {total_frames} ({progress * 100:.2f}%)")
    
        for frame_count, frame_position_ms, future in futures:
            text = future.result()
            classification = classify_frame(text)
            
            minutes, seconds = divmod(frame_position_ms // 1000, 60)
            milliseconds = int(frame_position_ms % 1000)
            time_str = f"{int(minutes):02d}:{int(seconds):02d}.{milliseconds:03d}"
            
            results.append((frame_count, text, time_str, classification))
    
    cap.release()
    progress_bar.empty()
    progress_text.empty()
    return results

def main():
    st.set_page_config(page_title="Video Text Extraction and Classification", layout="wide")
    st.title("Video Text Extraction and Classification")
    
    with st.sidebar:
        st.subheader("Video Settings")
        video_file = st.file_uploader("Upload Video", type=["mp4", "mov", "avi"])
        start_time_ms = st.slider("Start Time (ms)", 0, 600000, 0)
        target_fps = st.slider("Target FPS", 1, 30, 10)
    
    if video_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(video_file.read())
            temp_file_path = tmp_file.name
        
        if st.button("Process Video"):
            results = process_video(temp_file_path, start_time_ms, target_fps)
            
            if results:
                st.subheader("Extracted Text and Classification")
                for frame_count, text, time_str, classification in results:
                    with st.expander(f"Frame {frame_count} (Time: {time_str}) - {classification.capitalize()}"):
                        st.write(text)
                
                classifications = [r[3] for r in results]
                texted_count = classifications.count("texted")
                semi_textless_count = classifications.count("semi-textless")
                textless_count = classifications.count("textless")
                
                st.subheader("Classification Summary")
                st.write(f"Texted frames: {texted_count}")
                st.write(f"Semi-textless frames: {semi_textless_count}")
                st.write(f"Textless frames: {textless_count}")
                
                st.bar_chart({
                    "Texted": texted_count,
                    "Semi-textless": semi_textless_count,
                    "Textless": textless_count
                })
            else:
                st.write("No text detected or video processing failed.")
        
        os.remove(temp_file_path)

if __name__ == "__main__":
    main()


# from flask import Flask, render_template, request
# import cv2
# import pytesseract
# from pytesseract import Output as PyOutput
# import tempfile
# import os
# import base64
# from concurrent.futures import ThreadPoolExecutor

# app = Flask(__name__)

# # Path to Tesseract executable
# pytesseract.pytesseract.tesseract_cmd = r'/opt/homebrew/bin/tesseract'

# def preprocess_frame(frame):
#     resized_frame = cv2.resize(frame, (640, 360))  # Resize the frame to reduce processing time
#     gray_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
#     blur_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)
#     _, thresh_frame = cv2.threshold(blur_frame, 150, 255, cv2.THRESH_BINARY)
#     return thresh_frame

# def extract_text_from_frame(frame):
#     preprocessing_frame = preprocess_frame(frame)
#     data = pytesseract.image_to_data(preprocessing_frame, config='--oem 3 --psm 6', output_type=PyOutput.DICT)
#     text = ''
#     confidence_threshold = 70
#     for i in range(len(data['text'])):
#         if float(data['conf'][i]) >= confidence_threshold:
#             text += data['text'][i] + ' '
#     return text.strip()

# def process_video(video_path, start_time_ms, target_fps):
#     cap = cv2.VideoCapture(video_path)
#     if not cap.isOpened():
#         return None

#     frame_count = 0
#     original_fps = cap.get(cv2.CAP_PROP_FPS)
#     frame_interval = int(original_fps / target_fps)

#     results = []
#     with ThreadPoolExecutor() as executor:
#         while cap.isOpened():
#             ret, frame = cap.read()
#             if not ret:
#                 break

#             frame_position_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
#             if frame_position_ms < start_time_ms:
#                 continue

#             future = executor.submit(extract_text_from_frame, frame)
#             text = future.result()

#             minutes, seconds = divmod(frame_position_ms // 1000, 60)
#             milliseconds = int(frame_position_ms % 1000)
#             time_str = f"{int(minutes):02d}:{int(seconds):02d}.{milliseconds:03d}"

#             results.append((frame_count, text, time_str))
#             frame_count += 1

#     cap.release()
#     return results

# @app.route("/", methods=["GET", "POST"])
# def index():
#     extracted_text = None
#     if request.method == "POST":
#         video_file = request.files.get("video_file")
#         start_time_ms = int(request.form.get("start_time_ms", 0))
#         target_fps = int(request.form.get("target_fps", 30))

#         if video_file:
#             video_data = video_file.read()
#             with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
#                 tmp_file.write(video_data)
#                 temp_file_path = tmp_file.name

#             results = process_video(temp_file_path, start_time_ms, target_fps)

#             if results:
#                 extracted_text = [
#                     {"frame": frame_count, "text": text, "time": time_str}
#                     for frame_count, text, time_str in results
#                 ]
#             os.remove(temp_file_path)

#     return render_template("index.html", extracted_text=extracted_text)

# if __name__ == "__main__":
#     app.run(debug=True)


