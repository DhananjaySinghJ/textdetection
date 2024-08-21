# import streamlit as st
# import cv2
# import pytesseract
# from pytesseract import Output
# import tempfile
# import os
# from concurrent.futures import ThreadPoolExecutor
# import numpy as np

# # Path to Tesseract executable
# pytesseract.pytesseract.tesseract_cmd = r'/opt/homebrew/bin/tesseract'

# def preprocess_frame(frame):
#     resized_frame = cv2.resize(frame, (320, 180))  # Resize frame
#     gray_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
#     _, thresh_frame = cv2.threshold(gray_frame, 150, 255, cv2.THRESH_BINARY)
#     return thresh_frame

# def extract_text_from_frame(frame, is_subtitle=False):
#     preprocessing_frame = preprocess_frame(frame)
    
#     if is_subtitle:
#         height = preprocessing_frame.shape[0]
#         subtitle_region = preprocessing_frame[int(2*height/3):, :]
#         data = pytesseract.image_to_data(subtitle_region, config='--oem 1 --psm 6', output_type=Output.DICT)
#     else:
#         data = pytesseract.image_to_data(preprocessing_frame, config='--oem 1 --psm 6', output_type=Output.DICT)
    
#     text = ' '.join([word for i, word in enumerate(data['text']) if float(data['conf'][i]) >= 70])
#     return text.strip()

# def classify_frame(text, subtitle_text):
#     word_count = len(text.split())
#     if word_count == 0:
#         frame_class = "textless"
#     elif word_count < 5:
#         frame_class = "semi-textless"
#     else:
#         frame_class = "texted"
    
#     has_subtitle = len(subtitle_text) > 0
#     return frame_class, has_subtitle

# @st.cache_data
# def process_video(video_path, start_time_ms, target_fps):
#     cap = cv2.VideoCapture(video_path)
#     if not cap.isOpened():
#         st.error(f"Error: Could not open video {video_path}.")
#         return []
    
#     original_fps = cap.get(cv2.CAP_PROP_FPS)
#     total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     frame_interval = max(1, int(original_fps / target_fps))
    
#     results = []
#     frame_count = 0
    
#     progress_bar = st.progress(0)
#     progress_text = st.empty()
    
#     with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
#         futures = []
#         while cap.isOpened():
#             ret, frame = cap.read()
#             if not ret:
#                 break
            
#             frame_position_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
#             if frame_position_ms < start_time_ms:
#                 continue
            
#             if frame_count % frame_interval == 0:
#                 future_text = executor.submit(extract_text_from_frame, frame)
#                 future_subtitle = executor.submit(extract_text_from_frame, frame, is_subtitle=True)
#                 futures.append((frame_count, frame_position_ms, future_text, future_subtitle))
            
#             frame_count += 1
            
#             if frame_count % 10 == 0:
#                 progress = frame_count / total_frames
#                 progress_bar.progress(progress)
#                 progress_text.text(f"Processing frame {frame_count} of {total_frames} ({progress * 100:.2f}%)")
    
#         for frame_count, frame_position_ms, future_text, future_subtitle in futures:
#             text = future_text.result()
#             subtitle_text = future_subtitle.result()
#             classification, has_subtitle = classify_frame(text, subtitle_text)
            
#             minutes, seconds = divmod(frame_position_ms // 1000, 60)
#             milliseconds = int(frame_position_ms % 1000)
#             time_str = f"{int(minutes):02d}:{int(seconds):02d}.{milliseconds:03d}"
            
#             results.append((frame_count, text, subtitle_text, time_str, classification, has_subtitle))
    
#     cap.release()
#     progress_bar.empty()
#     progress_text.empty()
#     return results

# def main():
#     st.set_page_config(page_title="Video Text Extraction and Classification", layout="wide")
#     st.title("📹 Video Text Extraction and Classification")
#     st.markdown("This app processes a video to extract text from frames, classify frames based on text content, and detect subtitles.")

#     with st.sidebar:
#         st.subheader("⚙️ Video Settings")
#         video_file = st.file_uploader("📂 Upload Video", type=["mp4", "mov", "avi"], help="Select a video file to process.")
#         start_time_ms = st.slider("⏱️ Start Time (ms)", 0, 600000, 0, help="Specify the start time in milliseconds.")
#         target_fps = st.slider("🎞️ Target FPS", 1, 30, 10, help="Set the frames per second to process.")
        
#         st.subheader("📊 Classification Filter")
#         classification_filter = st.multiselect(
#             "Select frame classifications to display:",
#             options=["texted", "semi-textless", "textless"],
#             default=["texted", "semi-textless", "textless"]
#         )
        
#     if video_file is not None:
#         with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
#             tmp_file.write(video_file.read())
#             temp_file_path = tmp_file.name
        
#         if st.button("🚀 Process Video") or 'processed_results' not in st.session_state:
#             with st.spinner("Processing video..."):
#                 st.session_state['processed_results'] = process_video(temp_file_path, start_time_ms, target_fps)
        
#         results = st.session_state.get('processed_results', [])
        
#         if results:
#             # Classification Summary
#             classifications = [r[4] for r in results]
#             subtitle_count = sum(r[5] for r in results)
#             texted_count = classifications.count("texted")
#             semi_textless_count = classifications.count("semi-textless")
#             textless_count = classifications.count("textless")
            
#             st.subheader("📊 Classification Summary")
#             col1, col2, col3, col4 = st.columns(4)
#             col1.metric("Texted Frames", texted_count, delta=None)
#             col2.metric("Semi-Textless Frames", semi_textless_count, delta=None)
#             col3.metric("Textless Frames", textless_count, delta=None)
#             col4.metric("Frames with Subtitles", subtitle_count, delta=None)
            
#             st.bar_chart({
#                 "Texted": texted_count,
#                 "Semi-textless": semi_textless_count,
#                 "Textless": textless_count,
#                 "With Subtitles": subtitle_count
#             })
            
#             st.subheader("📑 Extracted Text and Classification")
#             for frame_count, text, subtitle_text, time_str, classification, has_subtitle in results:
#                 if classification in classification_filter:
#                     subtitle_status = "Subtitle detected" if has_subtitle else "No subtitle"
#                     badge_color = {
#                         "texted": "🟢",
#                         "semi-textless": "🟡",
#                         "textless": "🔴"
#                     }[classification]
#                     with st.expander(f"{badge_color} Frame {frame_count} (Time: {time_str}) - {classification.capitalize()} - {subtitle_status}"):
#                         st.write("Main text:", text)
#                         if has_subtitle:
#                             st.write("Subtitle:", subtitle_text)
#         else:
#             st.warning("No text detected or video processing failed.")
        
#         os.remove(temp_file_path)

# if __name__ == "__main__":
#     main()

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
    resized_frame = cv2.resize(frame, (320, 180))  # Resize frame
    gray_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
    _, thresh_frame = cv2.threshold(gray_frame, 150, 255, cv2.THRESH_BINARY)
    return thresh_frame

def extract_text_from_frame(frame, is_subtitle=False):
    preprocessing_frame = preprocess_frame(frame)
    
    if is_subtitle:
        height = preprocessing_frame.shape[0]
        subtitle_region = preprocessing_frame[int(2*height/3):, :]
        data = pytesseract.image_to_data(subtitle_region, config='--oem 1 --psm 6', output_type=Output.DICT)
    else:
        data = pytesseract.image_to_data(preprocessing_frame, config='--oem 1 --psm 6', output_type=Output.DICT)
    
    text = ' '.join([word for i, word in enumerate(data['text']) if float(data['conf'][i]) >= 70])
    return text.strip()

def classify_frame(text, subtitle_text):
    word_count = len(text.split())
    if word_count == 0:
        frame_class = "textless"
    elif word_count < 5:
        frame_class = "semi-textless"
    else:
        frame_class = "texted"
    
    has_subtitle = len(subtitle_text) > 0
    return frame_class, has_subtitle

@st.cache_data
def process_video(video_path, start_time_ms, target_fps):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error(f"Error: Could not open video {video_path}.")
        return []
    
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
                future_text = executor.submit(extract_text_from_frame, frame)
                future_subtitle = executor.submit(extract_text_from_frame, frame, is_subtitle=True)
                futures.append((frame_count, frame_position_ms, frame, future_text, future_subtitle))
            
            frame_count += 1
            
            if frame_count % 10 == 0:
                progress = frame_count / total_frames
                progress_bar.progress(progress)
                progress_text.text(f"Processing frame {frame_count} of {total_frames} ({progress * 100:.2f}%)")
    
        for frame_count, frame_position_ms, frame, future_text, future_subtitle in futures:
            text = future_text.result()
            subtitle_text = future_subtitle.result()
            classification, has_subtitle = classify_frame(text, subtitle_text)
            
            minutes, seconds = divmod(frame_position_ms // 1000, 60)
            milliseconds = int(frame_position_ms % 1000)
            time_str = f"{int(minutes):02d}:{int(seconds):02d}.{milliseconds:03d}"
            
            results.append((frame_count, text, subtitle_text, time_str, classification, has_subtitle, frame))
    
    cap.release()
    progress_bar.empty()
    progress_text.empty()
    return results

def main():
    st.set_page_config(page_title="Video Text Extraction and Classification", layout="wide")
    st.title("📹 Video Text Extraction and Classification")
    st.markdown("This app processes a video to extract text from frames, classify frames based on text content, and detect subtitles.")

    with st.sidebar:
        st.subheader("⚙️ Video Settings")
        video_file = st.file_uploader("📂 Upload Video", type=["mp4", "mov", "avi"], help="Select a video file to process.")
        start_time_ms = st.slider("⏱️ Start Time (ms)", 0, 600000, 0, help="Specify the start time in milliseconds.")
        target_fps = st.slider("🎞️ Target FPS", 1, 30, 10, help="Set the frames per second to process.")
        
        st.subheader("📊 Classification Filter")
        classification_filter = st.multiselect(
            "Select frame classifications to display:",
            options=["texted", "semi-textless", "textless"],
            default=["texted", "semi-textless", "textless"]
        )
        
    if video_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(video_file.read())
            temp_file_path = tmp_file.name
        
        if st.button("🚀 Process Video") or 'processed_results' not in st.session_state:
            with st.spinner("Processing video..."):
                st.session_state['processed_results'] = process_video(temp_file_path, start_time_ms, target_fps)
        
        results = st.session_state.get('processed_results', [])
        
        if results:
            # Classification Summary
            classifications = [r[4] for r in results]
            subtitle_count = sum(r[5] for r in results)
            texted_count = classifications.count("texted")
            semi_textless_count = classifications.count("semi-textless")
            textless_count = classifications.count("textless")
            
            st.subheader("📊 Classification Summary")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Texted Frames", texted_count, delta=None)
            col2.metric("Semi-Textless Frames", semi_textless_count, delta=None)
            col3.metric("Textless Frames", textless_count, delta=None)
            col4.metric("Frames with Subtitles", subtitle_count, delta=None)
            
            st.bar_chart({
                "Texted": texted_count,
                "Semi-textless": semi_textless_count,
                "Textless": textless_count,
                "With Subtitles": subtitle_count
            })
            
            st.subheader("📑 Extracted Text and Classification")
            for frame_count, text, subtitle_text, time_str, classification, has_subtitle, frame in results:
                if classification in classification_filter:
                    subtitle_status = "Subtitle detected" if has_subtitle else "No subtitle"
                    badge_color = {
                        "texted": "🟢",
                        "semi-textless": "🟡",
                        "textless": "🔴"
                    }[classification]
                    
                    with st.expander(f"{badge_color} Frame {frame_count} (Time: {time_str}) - {classification.capitalize()} - {subtitle_status}"):
                        st.write("Main text:", text)
                        if has_subtitle:
                            st.write("Subtitle:", subtitle_text)
                        st.image(frame, channels="BGR", caption=f"Frame {frame_count}")
        else:
            st.warning("No text detected or video processing failed.")
        
        os.remove(temp_file_path)

if __name__ == "__main__":
    main()

