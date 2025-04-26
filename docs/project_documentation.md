# Parking Space Detector: Comprehensive Project Documentation

## Project Overview

The Parking Space Detector is a computer vision application designed to accurately identify and count occupied and vacant parking spaces in parking lot images and video streams. It uses a custom-trained YOLOv8 model optimized specifically for parking space detection.

## Model Training Details

### Dataset

The model was trained on a specialized parking space dataset with the following characteristics:

- **Classes**: 2 classes (`occupied` and `vacant` parking spaces)
- **Training data**: Located in `train/images` and `train/labels`
- **Validation data**: Located in `valid/images` and `valid/labels`
- **Test data**: Located in `test/images` and `test/labels`
- **Source**: Roboflow dataset "parking-space-4-wucim" (version 1)

### Training Process

The training process leveraged transfer learning by starting with a pre-trained YOLOv8 model (`yolov8n.pt`) and fine-tuning it on our parking space dataset:

- **Framework**: Ultralytics YOLOv8
- **Base model**: YOLOv8n (nano)
- **Epochs**: 50
- **Image size**: 640×640 pixels
- **Batch size**: 16
- **Early stopping patience**: 20
- **Training time**: 2.395 hours (8621 seconds)

### Training Strategies

Several key training strategies were employed:

1. **Transfer Learning**: Started with a pre-trained model to leverage general object detection capabilities
2. **Data Augmentation**: Used mosaic augmentation for the first 40 epochs
3. **Adaptive Learning Rate**: Implemented learning rate scheduling, peaking at 0.00153 and gradually reducing to 4.97e-05
4. **Early Stopping**: Set to halt training if no improvement after 20 epochs (though the model continued improving)

## Model Performance Analysis

### Final Metrics

The trained model achieved exceptional performance on the validation dataset:

| Metric | Value | Description |
|--------|-------|-------------|
| mAP50 | 0.968 (96.8%) | Mean Average Precision at IoU threshold of 0.5 |
| mAP50-95 | 0.883 (88.3%) | Mean Average Precision across IoU thresholds from 0.5 to 0.95 |
| Precision | 0.952 | Proportion of correct detections among all detections |
| Recall | 0.921 | Proportion of actual spaces correctly detected |

### Class-Specific Performance

| Class | mAP50 | mAP50-95 | Precision | Recall |
|-------|-------|----------|-----------|--------|
| Occupied | 0.970 | 0.910 | 0.964 | 0.938 |
| Vacant | 0.965 | 0.856 | 0.940 | 0.912 |

### Training Progress

The training progression showed consistent improvement throughout the 50 epochs:

- **Early epochs (1-10)**: Rapid improvement from 0.540 to 0.943 mAP50
- **Middle epochs (11-30)**: Continued refinement from 0.953 to 0.969 mAP50
- **Final epochs (31-50)**: Fine-tuning with slight improvements from 0.966 to 0.968 mAP50

## Performance Analysis Visualizations

Key visualizations were generated during training to analyze model performance:

- **Precision-Recall Curve**: Shows the trade-off between precision and recall
- **Confusion Matrix**: Highlights correct classifications and misclassifications
- **F1 Curve**: Displays the F1 score (harmonic mean of precision and recall) across different confidence thresholds

## Model Architecture

The YOLOv8n (nano) architecture provides an excellent balance between performance and efficiency:

- **Parameters**: 3,006,038
- **GFLOPs**: 8.1
- **Layers**: 72 (after fusion)
- **Inference Speed**: ~42.3ms per image on CPU

## Inference Performance

The model demonstrates excellent real-world performance:

- **Processing Time**:
  - 0.7ms for preprocessing
  - 42.3ms for inference
  - 1.6ms for postprocessing
- **Total Inference Time**: ~45ms per image (22 FPS) on CPU

## Application Integration

The model is seamlessly integrated into a user-friendly Streamlit web application that:

1. Accepts uploaded parking lot images or connects to video streams
2. Processes the images using the trained YOLOv8 model
3. Identifies and marks occupied spaces with red bounding boxes
4. Identifies and marks vacant spaces with green bounding boxes
5. Calculates and displays occupancy statistics

## Deployment Instructions

The application can be deployed using the following steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/pranit1812/parking-space-detector.git
   cd parking-space-detector
   ```

2. **Set up the environment**:
   ```bash
   python -m venv venv
   # On Windows
   venv\Scripts\activate
   # On macOS/Linux
   source venv/bin/activate
   
   # Install dependencies
   pip install -r requirements.txt
   ```

3. **Run the application**:
   ```bash
   python run.py
   ```

## Future Enhancements

Potential areas for further improvement include:

1. **Real-time CCTV Integration**: Adding direct support for RTSP streams from parking lot CCTV cameras
2. **Time-Series Analysis**: Tracking occupancy changes over time to predict peak periods
3. **Mobile App**: Developing a companion mobile application for quick access
4. **Model Optimization**: Further optimizing the model for edge deployment using quantization
5. **Counting System Integration**: Connecting with parking barrier systems for automatic entry/exit management

## Technical Challenges and Solutions

During development, several technical challenges were addressed:

1. **Class Imbalance**: Addressed through careful dataset curation and class weighting
2. **Environmental Variability**: Improved through data augmentation strategies
3. **Detection Accuracy**: Enhanced through transfer learning and hyperparameter optimization
4. **Real-time Performance**: Achieved by selecting an optimal model size and export format

## Conclusion

The Parking Space Detector project successfully demonstrates the application of modern computer vision techniques to solve a practical urban problem. The high accuracy achieved (96.8% mAP50) makes this system suitable for real-world deployment in parking management scenarios, providing valuable information for both parking operators and drivers.

By leveraging the power of YOLOv8 and optimizing the model specifically for parking space detection, we've created a system that balances accuracy, speed, and usability, making it a viable solution for smart parking infrastructure. 