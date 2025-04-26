#!/usr/bin/env python3
import json
import os
from pathlib import Path
import numpy as np
from shapely.geometry import Polygon

class ParkingSpaceManager:
    def __init__(self, spaces_path=None):
        """
        Initialize the parking space manager.
        
        Args:
            spaces_path (str, optional): Path to the JSON file storing parking space coordinates.
                                        Defaults to 'data/spaces.json'.
        """
        if spaces_path is None:
            spaces_path = Path.cwd() / 'data' / 'spaces.json'
        self.spaces_path = Path(spaces_path)
        
        # Ensure directory exists
        os.makedirs(self.spaces_path.parent, exist_ok=True)
        
        # Load existing spaces or create empty structure
        self.spaces = self._load_spaces()
        
    def _load_spaces(self):
        """Load parking spaces from JSON file."""
        if not self.spaces_path.exists():
            return {"spaces": {}}
        
        try:
            with open(self.spaces_path, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return {"spaces": {}}
    
    def save_spaces(self):
        """Save parking spaces to JSON file."""
        with open(self.spaces_path, 'w') as f:
            json.dump(self.spaces, f, indent=2)
    
    def add_space(self, space_id, coordinates):
        """
        Add a new parking space.
        
        Args:
            space_id (str): Identifier for the parking space
            coordinates (list): List of [x, y] coordinate pairs defining the polygon
        """
        if "spaces" not in self.spaces:
            self.spaces["spaces"] = {}
            
        self.spaces["spaces"][space_id] = {
            "coordinates": coordinates,
            "polygon": coordinates
        }
        self.save_spaces()
    
    def remove_space(self, space_id):
        """Remove a parking space by ID."""
        if space_id in self.spaces["spaces"]:
            del self.spaces["spaces"][space_id]
            self.save_spaces()
    
    def get_all_spaces(self):
        """Return all parking spaces."""
        return self.spaces.get("spaces", {})
    
    def get_space(self, space_id):
        """Get a specific parking space by ID."""
        return self.spaces.get("spaces", {}).get(space_id)
    
    def get_polygons(self):
        """Get all parking spaces as Shapely polygons."""
        polygons = {}
        for space_id, space_data in self.spaces.get("spaces", {}).items():
            coords = space_data.get("polygon", [])
            if coords:
                polygons[space_id] = Polygon(coords)
        return polygons
    
    def check_occupancy(self, detections):
        """
        Check which parking spaces are occupied based on object detections.
        
        Args:
            detections (list): List of detected objects, each containing:
                              - 'bbox': [x1, y1, x2, y2] bounding box
                              - 'class': object class (0=occupied, 1=vacant)
                              - 'confidence': detection confidence
        
        Returns:
            dict: Dictionary mapping space_id to occupancy status (True=occupied, False=vacant)
        """
        occupancy = {}
        polygons = self.get_polygons()
        
        for space_id, polygon in polygons.items():
            space_occupied = False
            
            for detection in detections:
                if detection['class'] == 0:  # 'occupied' class
                    # Convert bbox to polygon
                    x1, y1, x2, y2 = detection['bbox']
                    box_polygon = Polygon([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
                    
                    # Check intersection
                    if polygon.intersects(box_polygon):
                        intersection_area = polygon.intersection(box_polygon).area
                        if intersection_area / polygon.area > 0.3:  # At least 30% overlap
                            space_occupied = True
                            break
            
            occupancy[space_id] = space_occupied
            
        return occupancy 