#!/usr/bin/env python3
import cv2
import numpy as np
import argparse
import os
from pathlib import Path
import json

from utils.camera import CameraStream
from utils.parking_spaces import ParkingSpaceManager

class ParkingSpaceConfigurator:
    def __init__(self, image_path=None, camera_source=None, output_path=None):
        """
        Initialize the parking space configurator.
        
        Args:
            image_path (str, optional): Path to an image file
            camera_source (str, optional): Camera source (URL or device index)
            output_path (str, optional): Path to save the space coordinates JSON
        """
        self.image_path = image_path
        self.camera_source = camera_source
        
        if output_path is None:
            output_path = Path.cwd() / 'data' / 'spaces.json'
        self.output_path = Path(output_path)
        
        # Initialize space manager
        self.space_manager = ParkingSpaceManager(self.output_path)
        
        # Drawing parameters
        self.image = None
        self.drawing = False
        self.current_polygon = []
        self.current_id = None
        self.polygons = {}
        
        # Load existing spaces
        self.load_spaces()
    
    def load_spaces(self):
        """Load existing spaces from the space manager."""
        spaces = self.space_manager.get_all_spaces()
        self.polygons = {}
        
        for space_id, space_data in spaces.items():
            self.polygons[space_id] = space_data.get("polygon", [])
    
    def load_image(self):
        """Load an image from file or camera."""
        if self.image_path:
            # Load from file
            if not os.path.exists(self.image_path):
                print(f"Error: Image file not found: {self.image_path}")
                return False
                
            self.image = cv2.imread(self.image_path)
            return self.image is not None
            
        elif self.camera_source:
            # Capture from camera
            camera = CameraStream(self.camera_source)
            if not camera.start():
                print(f"Error: Could not connect to camera: {self.camera_source}")
                return False
                
            # Wait for a frame
            for _ in range(10):
                frame = camera.read()
                if frame is not None:
                    self.image = frame.copy()
                    camera.stop()
                    return True
                    
            camera.stop()
            print("Error: Could not capture frame from camera")
            return False
        else:
            print("Error: No image source specified")
            return False
    
    def draw_polygons(self, image):
        """Draw all polygons on the image."""
        result = image.copy()
        
        for space_id, points in self.polygons.items():
            if not points:
                continue
                
            # Convert points to numpy array
            points_array = np.array(points, np.int32)
            points_array = points_array.reshape((-1, 1, 2))
            
            # Draw polygon
            cv2.polylines(result, [points_array], True, (0, 255, 0), 2)
            
            # Add ID text
            if len(points) > 0:
                center_x = int(np.mean([p[0] for p in points]))
                center_y = int(np.mean([p[1] for p in points]))
                cv2.putText(
                    result, 
                    space_id, 
                    (center_x, center_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.8, 
                    (255, 255, 255), 
                    2, 
                    cv2.LINE_AA
                )
        
        # Draw current polygon
        if self.current_polygon:
            points_array = np.array(self.current_polygon, np.int32)
            cv2.polylines(result, [points_array.reshape((-1, 1, 2))], False, (0, 0, 255), 2)
            
            # Draw points
            for point in self.current_polygon:
                cv2.circle(result, point, 5, (0, 0, 255), -1)
        
        return result
    
    def handle_click(self, event, x, y, flags, param):
        """Mouse event handler for polygon drawing."""
        if event == cv2.EVENT_LBUTTONDOWN:
            # Start or continue drawing
            self.current_polygon.append((x, y))
        
        elif event == cv2.EVENT_RBUTTONDOWN:
            # Complete polygon
            if len(self.current_polygon) >= 3:
                if self.current_id is None:
                    # Generate new ID
                    existing_ids = list(self.polygons.keys())
                    if existing_ids:
                        # Find the highest numeric ID and increment
                        numeric_ids = [
                            int(id.replace("P", "")) 
                            for id in existing_ids 
                            if id.startswith("P") and id[1:].isdigit()
                        ]
                        if numeric_ids:
                            next_id = max(numeric_ids) + 1
                            self.current_id = f"P{next_id}"
                        else:
                            self.current_id = "P1"
                    else:
                        self.current_id = "P1"
                
                # Save polygon
                self.polygons[self.current_id] = self.current_polygon.copy()
                self.space_manager.add_space(self.current_id, self.current_polygon.copy())
                
                # Reset for next polygon
                self.current_polygon = []
                self.current_id = None
            else:
                # Not enough points, reset
                self.current_polygon = []
    
    def run(self):
        """Run the configuration interface."""
        if not self.load_image():
            return False
        
        # Create window and set mouse callback
        window_name = "Parking Space Configuration"
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, self.handle_click)
        
        print("Parking Space Configuration")
        print("---------------------------")
        print("Left click: Add polygon point")
        print("Right click: Complete polygon (minimum 3 points)")
        print("ESC: Exit and save")
        print("DELETE: Remove last space")
        
        while True:
            # Draw all polygons
            display = self.draw_polygons(self.image.copy())
            
            # Show instructions
            cv2.putText(
                display, 
                "Left click: Add point | Right click: Complete | ESC: Exit | DEL: Remove last", 
                (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, 
                (255, 255, 255), 
                2, 
                cv2.LINE_AA
            )
            
            # Display image
            cv2.imshow(window_name, display)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == 27:  # ESC
                break
            elif key == 8 or key == 127:  # BACKSPACE or DELETE
                # Remove the last added space
                if self.polygons:
                    last_id = list(self.polygons.keys())[-1]
                    self.space_manager.remove_space(last_id)
                    del self.polygons[last_id]
        
        cv2.destroyAllWindows()
        
        # Save spaces
        self.space_manager.save_spaces()
        print(f"Saved {len(self.polygons)} parking spaces to {self.output_path}")
        
        return True


def main():
    parser = argparse.ArgumentParser(description="Configure parking spaces")
    
    # Add arguments for image or camera source
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--image", help="Path to image file")
    source_group.add_argument("--camera", help="Camera source (URL or device index)")
    
    # Add output path argument
    parser.add_argument("--output", help="Path to save space coordinates JSON")
    
    args = parser.parse_args()
    
    # Create configurator
    configurator = ParkingSpaceConfigurator(
        image_path=args.image,
        camera_source=args.camera,
        output_path=args.output
    )
    
    # Run configuration
    configurator.run()


if __name__ == "__main__":
    main() 