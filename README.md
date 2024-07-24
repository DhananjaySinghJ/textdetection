# textdetection 
# Video Text Detection

This project analyzes video frames to detect and categorize text presence using the Tesseract OCR engine. The script can analyze entire videos and print summaries of scene types and detected text.

## Features

- Detects text in video frames.
- Categorizes frames as "Texted," "Semi-Textless," or "Textless".
- Analyzes entire videos at specified intervals.

## Requirements

- Python 3.x
- `opencv-python` library
- `pytesseract` library
- `numpy` library
- Tesseract OCR installed

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/DhananjaySinghJ/textdetection.git
   cd textdetection
   
2. Install the required packages:
- pip install opencv-python pytesseract numpy

3. Install tesseract ocr
- brew install tesseract

4. Set the path to the Tesseract executable in your script:
- pytesseract.pytesseract.tesseract_cmd = r'/path/to/tesseract'

