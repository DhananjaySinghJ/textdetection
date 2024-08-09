import streamlit as st
import cv2
import pytesseract
from pytesseract import Output
import tempfile
import os
from concurrent.futures import ThreadPoolExecutor

# Path to Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'/opt/homebrew/bin/tesseract'

def preprocess_frame(frame):
    resized_frame = cv2.resize(frame, (640, 360))  # Resize the frame to reduce processing time
    gray_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
    blur_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)
    _, thresh_frame = cv2.threshold(blur_frame, 150, 255, cv2.THRESH_BINARY)
    return thresh_frame

def extract_text_from_frame(frame):
    preprocessing_frame = preprocess_frame(frame)
    data = pytesseract.image_to_data(preprocessing_frame, config='--oem 3 --psm 6', output_type=Output.DICT)
    text = ''
    confidence_threshold = 70
    for i in range(len(data['text'])):
        if float(data['conf'][i]) >= confidence_threshold:
            text += data['text'][i] + ' '
    return text.strip()

def process_video(video_path, start_time_ms, target_fps):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error(f"Error: Could not open video {video_path}.")
        return

    frame_count = 0
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = int(original_fps / target_fps)

    results = []
    progress_bar = st.progress(0)
    progress_text = st.empty()

    with ThreadPoolExecutor() as executor:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_position_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            if frame_position_ms < start_time_ms:
                continue

            future = executor.submit(extract_text_from_frame, frame)
            text = future.result()

            minutes, seconds = divmod(frame_position_ms // 1000, 60)
            milliseconds = int(frame_position_ms % 1000)
            time_str = f"{int(minutes):02d}:{int(seconds):02d}.{milliseconds:03d}"

            results.append((frame_count, text, time_str))
            frame_count += 1

            # Update the progress bar
            progress = frame_count / total_frames
            progress_bar.progress(progress)
            progress_text.text(f"Processing frame {frame_count} of {total_frames} ({progress * 100:.2f}%)")

    cap.release()
    progress_bar.empty()  # Remove the progress bar when done
    progress_text.empty()  # Remove the progress text when done

    return results

def main():
    st.set_page_config(page_title="Video Text Extraction and Classification", layout="wide")

    st.title("Video Text Extraction and Classification")

    with st.sidebar:
        st.subheader("Video Settings")
        video_file = st.file_uploader("Upload Video", type=["mp4", "mov", "avi"])
        start_time_ms = st.slider("Start Time (ms)", 0, 600000, 0)
        target_fps = st.slider("Target FPS", 1, 60, 30)

    if video_file is not None:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(video_file.read())
            temp_file_path = tmp_file.name

        if st.button("Process Video"):
            results = process_video(temp_file_path, start_time_ms, target_fps)

            if results:
                st.subheader("Extracted Text")
                for frame_count, text, time_str in results:
                    with st.expander(f"Frame {frame_count} (Time: {time_str})"):
                        st.write(text)
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


