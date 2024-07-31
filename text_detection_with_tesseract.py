import cv2
import pytesseract
import numpy as np
import streamlit as st
from spellchecker import SpellChecker
from transformers import BertTokenizer, BertForMaskedLM
import torch
import tempfile
import os

class VideoTextDetector:
    def __init__(self):
        self.spell = SpellChecker()
        self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert_model = BertForMaskedLM.from_pretrained('bert-base-uncased')

    def preprocess_image(self, image, region="full"):
        if region == "full":
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif region == "subtitle":
            height, _ = image.shape[:2]
            subtitle_region = image[int(3 * height / 4):height, :]
            gray = cv2.cvtColor(subtitle_region, cv2.COLOR_BGR2GRAY)
        else:
            raise ValueError("Invalid region specified")

        gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        return gray

    def detect_text(self, image, region="full", confidence_threshold=0.5):
        gray = self.preprocess_image(image, region)
        config = f'--oem 3 --psm 6'
        result = pytesseract.image_to_data(gray, config=config, output_type=pytesseract.Output.DICT)
        filtered_result = [result['text'][i] for i in range(len(result['text'])) if int(result['conf'][i]) > confidence_threshold * 100]
        return " ".join(filtered_result).strip()

    def correct_text_with_bert(self, text):
        tokens = self.bert_tokenizer.tokenize(text)
        for i, token in enumerate(tokens):
            if token not in self.spell:
                inputs = self.bert_tokenizer(text, return_tensors="pt")
                token_ids = inputs.input_ids[0]
                masked_position = token_ids[i + 1]
                token_ids[i + 1] = self.bert_tokenizer.mask_token_id
                with torch.no_grad():
                    outputs = self.bert_model(input_ids=token_ids.unsqueeze(0))
                predicted_token_id = outputs.logits[0, i + 1].argmax()
                predicted_token = self.bert_tokenizer.convert_ids_to_tokens([predicted_token_id])[0]
                tokens[i] = predicted_token
        return self.bert_tokenizer.convert_tokens_to_string(tokens)

    def analyze_frame(self, frame, confidence_threshold):
        full_text = self.detect_text(frame, "full", confidence_threshold)
        subtitle_text = self.detect_text(frame, "subtitle", confidence_threshold)

        if full_text and not subtitle_text:
            return "Texted", full_text
        elif subtitle_text and not full_text:
            return "Semi-Textless", subtitle_text
        elif full_text and subtitle_text:
            return "Texted", full_text
        else:
            return "Textless", ""

    def analyze_video(self, video_path, interval=1, confidence_threshold=0.5):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError("Error opening video file")

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        results = []
        for frame_number in range(0, frame_count, int(fps * interval)):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = cap.read()
            if not ret:
                break
            
            scene_type, detected_text = self.analyze_frame(frame, confidence_threshold)
            if scene_type != "Textless":
                corrected_text = self.correct_text_with_bert(detected_text)
                results.append((frame_number, scene_type, corrected_text))
            
            # Update progress
            progress = (frame_number + 1) / frame_count
            st.progress(progress)

        cap.release()
        return results

def main():
    st.title("Enhanced Video Text Detection")

    detector = VideoTextDetector()

    uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov", "mkv"])
    
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(uploaded_file.read())
            video_path = tmp_file.name

        interval = st.slider("Frame interval (seconds)", 1, 10, 1)
        confidence_threshold = st.slider("Confidence threshold", 0.0, 1.0, 0.5)

        if st.button("Process Video"):
            st.write("Processing...")
            results = detector.analyze_video(video_path, interval, confidence_threshold)

            st.write(f"Results for {uploaded_file.name}:")
            for frame_number, scene_type, text in results:
                st.write(f"Frame {frame_number}: {scene_type}")
                if scene_type != "Textless":
                    st.write(f"Detected Text: {text[:100]}...")

        # Clean up the temporary file
        os.unlink(video_path)

if __name__ == "__main__":
    main()
