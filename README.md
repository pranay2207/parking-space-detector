# Parking Space Detector

A computer vision application that detects and counts occupied and vacant parking spaces in parking lot images and video streams.

## Features

- Detect occupied and vacant parking spaces using a custom-trained YOLOv8 model
- Process static images or live camera feeds
- Calculate and display parking lot occupancy statistics
- Real-time visual indicators (red: occupied, green: vacant)
- Mobile-friendly web interface

## Model Training Results

The parking space detection model was trained on a dataset of parking lot images with the following results:

- mAP50: 0.968 (96.8%)
- mAP50-95: 0.883 (88.3%)
- Class detection:
  - Occupied spaces: 0.970 mAP50 (97.0%)
  - Vacant spaces: 0.965 mAP50 (96.5%)

## Requirements

- Python 3.8+
- YOLOv8 (ultralytics)
- Streamlit
- OpenCV
- See requirements.txt for complete dependencies

## Installation

```bash
# Clone the repository
git clone https://github.com/pranit1812/parking-space-detector.git
cd parking-space-detector

# Create and activate a virtual environment (optional but recommended)
python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

```bash
# Run the Streamlit app
python run.py
```

Then open your browser at http://localhost:8501

## Directory Structure

```
├── data.yaml            # Dataset configuration
├── models/              # Trained models
├── requirements.txt     # Project dependencies
├── run.py               # Main entry point
└── src/                 # Source code
    ├── app.py           # Streamlit application
    ├── train_model.py   # Model training script
    └── update_app.py    # Update script for model paths
```

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. 