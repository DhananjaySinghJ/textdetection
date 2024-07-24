# import cv2
# import pytesseract
# import numpy as np
# import pysrt

# # Set the path to Tesseract executable
# pytesseract.pytesseract.tesseract_cmd = r'/opt/homebrew/bin/tesseract'

# ## Function to detect text in an image
# def detect_text(image, region="full"):
#     """
#     This function detects text in an image based on the specified region.

#     Args:
#         image: The input image as a NumPy array.
#         region (str, optional): The region of the image to analyze. 
#             Can be "full" for the entire image or "subtitle" for the lower portion. 
#             Defaults to "full".

#     Returns:
#         tuple: A tuple containing the extracted text and a flag indicating 
#                if any text was detected.

#     Raises:
#         ValueError: If an invalid region is specified.
#     """

#     if region == "full":
#         gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     elif region == "subtitle":
#         height, _ = image.shape[:2]
#         subtitle_region = image[int(3*height/4):height, :]
#         gray = cv2.cvtColor(subtitle_region, cv2.COLOR_BGR2GRAY)
#     else:
#         raise ValueError("Invalid region specified")
    
#     text = pytesseract.image_to_string(gray)
#     return text.strip(), len(text.strip()) > 0

# ## Function to parse subtitle file
# def parse_subtitles(subtitle_path):
#     """
#     This function parses subtitle files and returns a list of subtitle entries.

#     Args:
#         subtitle_path (str): The path to the subtitle file.

#     Returns:
#         list: A list of tuples containing (start_time, end_time, text) for each subtitle entry.
#     """
#     subs = pysrt.open(subtitle_path)
#     subtitles = [(sub.start.seconds, sub.end.seconds, sub.text) for sub in subs]
#     return subtitles

# ## Function to analyze a single video frame
# def analyze_frame(frame, subtitles, fps):
#     """
#     This function analyzes a single video frame to identify scene type based on text presence.

#     Args:
#         frame: The video frame as a NumPy array.
#         subtitles: List of subtitles to check against.
#         fps: Frames per second of the video.

#     Returns:
#         tuple: A tuple containing the scene type ("Texted", "Semi-Textless", "Textless") 
#                and the detected text (empty string if none).
#     """
#     full_text, has_full_text = detect_text(frame, "full")
#     subtitle_text, has_subtitle = detect_text(frame, "subtitle")

#     timestamp = frame_number / fps
#     subtitle_match = any(start <= timestamp <= end for start, end, _ in subtitles)

#     if has_full_text and not has_subtitle:
#         return "Texted", full_text
#     elif subtitle_match and not has_full_text:
#         return "Semi-Textless", subtitle_text
#     elif has_full_text and subtitle_match:
#         return "Texted", full_text
#     else:
#         return "Textless", ""

# ## Function to analyze an entire video
# def analyze_video(video_path, subtitle_path=None, interval=1):
#     """
#     This function analyzes a video and returns a list of results for each frame at a specified interval.

#     Args:
#         video_path (str): The path to the video file.
#         subtitle_path (str, optional): The path to the subtitle file. Defaults to None.
#         interval (int, optional): The interval (in seconds) between analyzed frames. Defaults to 1.

#     Returns:
#         list: A list of tuples containing (timestamp, scene type, detected text) for each analyzed frame.
#     """
#     cap = cv2.VideoCapture(video_path)
#     fps = cap.get(cv2.CAP_PROP_FPS)
#     frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

#     subtitles = parse_subtitles(subtitle_path) if subtitle_path else []

#     results = []

#     for frame_number in range(0, frame_count, int(fps * interval)):
#         cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
#         ret, frame = cap.read()

#         if not ret:
#             break

#         scene_type, detected_text = analyze_frame(frame, subtitles, fps)

#         timestamp = frame_number / fps
#         results.append((timestamp, scene_type, detected_text))

#     cap.release()
#     return results

# ## Function to analyze video and print results
# def analyze_and_print_results(video_path, video_name, subtitle_path=None):
#     """
#     This function analyzes a video and prints a summary of the scene types and detected text.

#     Args:
#         video_path (str): The path to the video file.
#         video_name (str): The name of the video for printing results.
#         subtitle_path (str, optional): The path to the subtitle file. Defaults to None.
#     """
#     results = analyze_video(video_path, subtitle_path)

#     print(f"\nResults for {video_name}:")
#     for timestamp, scene_type, text in results:
#         print(f"Timestamp: {timestamp:.2f}s, Scene Type: {scene_type}")
#         if scene_type != "Textless":
#             print(f"Detected Text: {text[:100]}...")

# # Analyze Video A
# video_a_path = "/Users/dhananjay/Downloads/test1.mp4"
# subtitle_a_path = "/Users/dhananjay/Downloads/test1.srt"  # Path to subtitle file
# analyze_and_print_results(video_a_path, "Video A", subtitle_path=subtitle_a_path)

# # Analyze Video B
# # video_b_path = "path/to/video_b.mp4"
# # subtitle_b_path = "path/to/video_b.srt"  # Path to subtitle file
# # analyze_and_print_results(video_b_path, "Video B", subtitle_path=subtitle_b_path)





# ---------------- WITHOUT SUBTITLES --------------------
import cv2
import pytesseract
import numpy as np

# Set the path to Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'/opt/homebrew/bin/tesseract'

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
    
    text = pytesseract.image_to_string(gray)
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
        list: A list of tuples containing (timestamp, scene type, detected text) for each analyzed frame.
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    results = []

    for frame_number in range(0, frame_count, int(fps * interval)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()

        if not ret:
            break

        timestamp = frame_number / fps
        scene_type, detected_text = analyze_frame(frame)

        results.append((timestamp, scene_type, detected_text))

    cap.release()
    return results

## Function to analyze video and print results
def analyze_and_print_results(video_path, video_name):
    """
    This function analyzes a video and prints a summary of the scene types and detected text.

    Args:
        video_path (str): The path to the video file.
        video_name (str): The name of the video for printing results.
    """
    results = analyze_video(video_path)

    print(f"\nResults for {video_name}:")
    for timestamp, scene_type, text in results:
        print(f"Timestamp: {timestamp:.2f}s, Scene Type: {scene_type}")
        if scene_type != "Textless":
            print(f"Detected Text: {text[:100]}...")

# Analyze Video A
video_a_path = "/Users/dhananjay/Downloads/test2.mp4"
analyze_and_print_results(video_a_path, "Video A")

# Analyze Video B
# video_b_path = "path/to/video_b.mp4"
# analyze_and_print_results(video_b_path, "Video B")