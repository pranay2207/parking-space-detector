#!/usr/bin/env python3
import cv2
import numpy as np
import time
from pathlib import Path
from ultralytics import YOLO
import pandas as pd
import datetime

from utils.parking_spaces import ParkingSpaceManager

class ParkingDetector:
    def __init__(self, model_path=None, conf_threshold=0.25, iou_threshold=0.45):
        """
        Initialize the parking space detector.
        
        Args:
            model_path (str, optional): Path to YOLOv8 model file
            conf_threshold (float): Confidence threshold for detections
            iou_threshold (float): IoU threshold for NMS
        """
        # Find the best model if not specified
        if model_path is None:
            model_path = self._find_best_model()
        
        # Load the model
        self.model = YOLO(model_path)
        
        # Detection parameters
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        
        # Parking space manager
        self.space_manager = ParkingSpaceManager()
        
        # DataFrame for storing occupancy data
        self.occupancy_data = pd.DataFrame(columns=['timestamp', 'space_id', 'occupied'])
        
        # Cache for visualization
        self.last_results = None
        self.last_occupancy = {}
        self.last_processed_frame = None
    
    def _find_best_model(self):
        """Find the best available model in the models directory."""
        models_dir = Path.cwd() / 'models' / 'parking_detector' / 'weights'
        
        # Check for exported ONNX model first (faster inference)
        onnx_model = models_dir / 'best.onnx'
        if onnx_model.exists():
            return str(onnx_model)
        
        # Check for best PT model
        pt_model = models_dir / 'best.pt'
        if pt_model.exists():
            return str(pt_model)
        
        # Fall back to a pre-trained model
        print("Warning: No trained model found, using pre-trained YOLOv8n")
        return 'yolov8n.pt'
    
    def process_frame(self, frame):
        """
        Process a frame for parking space detection.
        
        Args:
            frame (numpy.ndarray): Input frame
            
        Returns:
            dict: Occupancy status by space_id
        """
        if frame is None:
            return {}
            
        # Detect objects in the frame
        results = self.model(
            frame, 
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=False
        )
        
        self.last_results = results
        
        # Convert results to the format expected by the space manager
        detections = []
        for result in results:
            boxes = result.boxes
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = box.conf[0].cpu().numpy()
                cls = int(box.cls[0].cpu().numpy())
                
                detections.append({
                    'bbox': [x1, y1, x2, y2],
                    'class': cls,  # 0=occupied, 1=vacant
                    'confidence': confidence
                })
        
        # Check occupancy
        occupancy = self.space_manager.check_occupancy(detections)
        self.last_occupancy = occupancy
        
        # Update occupancy data
        self._update_occupancy_data(occupancy)
        
        # Create visualization
        self.last_processed_frame = self._draw_results(frame, detections, occupancy)
        
        return occupancy
    
    def _update_occupancy_data(self, occupancy):
        """Update occupancy dataframe with new data."""
        timestamp = datetime.datetime.now()
        new_data = []
        
        for space_id, occupied in occupancy.items():
            new_data.append({
                'timestamp': timestamp,
                'space_id': space_id,
                'occupied': occupied
            })
        
        # Append to dataframe
        if new_data:
            df = pd.DataFrame(new_data)
            self.occupancy_data = pd.concat([self.occupancy_data, df], ignore_index=True)
    
    def save_occupancy_data(self, path=None):
        """
        Save occupancy data to CSV.
        
        Args:
            path (str, optional): Path to save CSV file
            
        Returns:
            str: Path to saved file
        """
        if path is None:
            path = Path.cwd() / 'data' / 'occupancy'
            Path(path).mkdir(parents=True, exist_ok=True)
            
            # Generate filename with timestamp
            filename = f"occupancy_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            path = path / filename
        
        self.occupancy_data.to_csv(path, index=False)
        return str(path)
    
    def _draw_results(self, frame, detections, occupancy):
        """
        Draw detection and occupancy results on the frame.
        
        Args:
            frame (numpy.ndarray): Input frame
            detections (list): List of detections
            occupancy (dict): Occupancy status by space_id
            
        Returns:
            numpy.ndarray: Annotated frame
        """
        # Create a copy of the frame
        annotated = frame.copy()
        
        # Draw parking spaces
        spaces = self.space_manager.get_all_spaces()
        for space_id, space_data in spaces.items():
            # Get polygon points
            polygon = np.array(space_data['polygon'], np.int32)
            polygon = polygon.reshape((-1, 1, 2))
            
            # Determine color based on occupancy
            color = (0, 255, 0)  # Green for vacant
            if occupancy.get(space_id, False):
                color = (0, 0, 255)  # Red for occupied
                
            # Draw filled polygon with transparency
            overlay = annotated.copy()
            cv2.fillPoly(overlay, [polygon], color)
            cv2.polylines(overlay, [polygon], True, (255, 255, 255), 2)
            cv2.addWeighted(overlay, 0.4, annotated, 0.6, 0, annotated)
            
            # Add space ID
            if polygon.size > 0:
                center = np.mean(polygon, axis=0).astype(np.int32)
                cv2.putText(
                    annotated, 
                    space_id, 
                    (center[0][0], center[0][1]), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.6, 
                    (255, 255, 255), 
                    2, 
                    cv2.LINE_AA
                )
        
        # Draw detections
        for detection in detections:
            x1, y1, x2, y2 = [int(c) for c in detection['bbox']]
            cls = detection['class']
            conf = detection['confidence']
            
            # Determine color based on class
            color = (0, 0, 255) if cls == 0 else (0, 255, 0)
            
            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            
            # Add label
            label = f"{'occupied' if cls == 0 else 'vacant'} {conf:.2f}"
            cv2.putText(
                annotated, 
                label, 
                (x1, y1 - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, 
                color, 
                2, 
                cv2.LINE_AA
            )
        
        # Draw summary
        vacant_count = list(occupancy.values()).count(False)
        total_count = len(occupancy)
        cv2.putText(
            annotated,
            f"Available: {vacant_count}/{total_count}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )
        
        return annotated
    
    def get_space_count(self):
        """Get count of total and available parking spaces."""
        if not self.last_occupancy:
            return {
                'total': 0,
                'available': 0,
                'occupied': 0
            }
            
        total = len(self.last_occupancy)
        occupied = list(self.last_occupancy.values()).count(True)
        available = total - occupied
        
        return {
            'total': total,
            'available': available,
            'occupied': occupied
        }
    
    def get_processed_frame(self):
        """Get the last processed frame with visualizations."""
        return self.last_processed_frame 