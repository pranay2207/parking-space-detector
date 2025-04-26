# Parking Space Detector - User Guide

## Introduction

The Parking Space Detector is a user-friendly application that helps identify and count occupied and vacant parking spaces in parking lot images. This guide explains how to use the application effectively.

## Getting Started

### Installation

Before using the application, make sure you have Python 3.8 or higher installed on your system. Then follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/pranit1812/parking-space-detector.git
   cd parking-space-detector
   ```

2. Create and activate a virtual environment (recommended):
   ```bash
   # Windows
   python -m venv venv
   venv\Scripts\activate

   # macOS/Linux
   python -m venv venv
   source venv/bin/activate
   ```

3. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Starting the Application

Launch the application by running:
```bash
python run.py
```

This will start a Streamlit web server and automatically open the application in your default web browser. If it doesn't open automatically, navigate to `http://localhost:8501` in your browser.

## Using the Application

### Main Interface

The application interface consists of:
- An upload area for parking lot images
- A results display section showing the analyzed image
- Parking statistics (occupied, available, and total spaces)

### Analyzing Parking Lot Images

1. **Upload an Image**:
   - Click the "Upload a parking lot image" button in the main content area
   - Select a JPG, JPEG, or PNG image of a parking lot
   - The application will automatically process the image once uploaded

2. **View Results**:
   - The processed image will appear with bounding boxes
   - Red boxes: Occupied parking spaces
   - Green boxes: Available parking spaces

3. **Understand Statistics**:
   - On the right side, you'll see statistics about:
     - Number of occupied spaces
     - Number of available spaces
     - Total spaces
     - Availability percentage
     - Types of vehicles detected

### Adjusting Settings

You can customize the application's behavior using the settings in the left sidebar:

1. **Manual Total Spaces**:
   - If you know the exact number of parking spaces in the lot, check "Manually set total parking spaces"
   - Enter the correct number in the field that appears
   - This improves accuracy when some spaces are not visible or not correctly detected

2. **Available Space Detection**:
   - Toggle "Show available parking spaces" to show or hide the detection of empty spaces
   - When enabled, the application will highlight available spaces with green boxes

## Example Workflow

1. Start the application using `python run.py`
2. Upload a parking lot image using the upload button
3. Wait a few seconds for the image to be processed
4. Review the detected spaces (red for occupied, green for available)
5. Check the statistics panel for occupancy information
6. If needed, adjust settings in the sidebar for better results

## Tips for Best Results

- Use clear, well-lit images of parking lots
- Images taken from an elevated angle work best
- For more accurate space counting, use the manual total spaces option
- Higher resolution images generally provide better detection results
- The model works well in various lighting conditions but performs best in daylight

## Troubleshooting

If you encounter issues:

- **No spaces detected**: Try adjusting the image or using a clearer image of the parking lot
- **Incorrect detections**: Enable manual space counting and provide the correct total
- **Application crashes**: Check your Python version (3.8+ required) and ensure all dependencies are installed
- **Slow performance**: Processing time depends on your hardware; larger images take longer to process

## Technical Support

For technical issues or questions, please:
- Submit an issue on GitHub: https://github.com/pranit1812/parking-space-detector/issues
- Include a description of the problem and steps to reproduce it
- Attach the problematic image if possible (with personal data removed) 