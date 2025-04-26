#!/usr/bin/env python3
import cv2
import threading
import time
import numpy as np
import base64
from pathlib import Path

class CameraStream:
    def __init__(self, source, auth=None, buffer_size=5):
        """
        Initialize a camera stream.
        
        Args:
            source (str): Camera source (URL or device index)
            auth (tuple, optional): Username and password tuple for RTSP authentication
            buffer_size (int, optional): Size of the frame buffer
        """
        self.source = source
        self.auth = auth
        self.buffer_size = buffer_size
        self.frames = []
        self.running = False
        self.lock = threading.Lock()
        self.thread = None
        
        # Check if source is a number (device index)
        try:
            source_num = int(source)
            self.connection_string = source_num
        except ValueError:
            # Handle RTSP authentication if provided
            if auth and "://" in source:
                protocol, rest = source.split("://", 1)
                username, password = auth
                auth_string = f"{username}:{password}@"
                self.connection_string = f"{protocol}://{auth_string}{rest}"
            else:
                self.connection_string = source
    
    def start(self):
        """Start the camera stream in a separate thread."""
        self.running = True
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()
        
        # Wait for first frame
        timeout = 10
        start_time = time.time()
        while not self.frames and time.time() - start_time < timeout:
            time.sleep(0.1)
            
        return len(self.frames) > 0
    
    def stop(self):
        """Stop the camera stream."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
    
    def _update(self):
        """Update the frame buffer continuously."""
        cap = cv2.VideoCapture(self.connection_string)
        
        # Set buffer size
        cap.set(cv2.CAP_PROP_BUFFERSIZE, self.buffer_size)
        
        # Check if camera opened successfully
        if not cap.isOpened():
            print(f"Error: Could not open camera {self.source}")
            self.running = False
            return
        
        while self.running:
            ret, frame = cap.read()
            
            if not ret:
                print(f"Error: Could not read frame from {self.source}")
                time.sleep(1)  # Wait before retrying
                continue
                
            with self.lock:
                self.frames = [frame]  # Keep only the latest frame
        
        cap.release()
    
    def read(self):
        """Read the latest frame from the buffer."""
        with self.lock:
            if not self.frames:
                return None
            return self.frames[-1].copy()
    
    def get_jpeg(self, quality=90):
        """
        Convert the latest frame to JPEG format.
        
        Args:
            quality (int): JPEG quality (0-100)
            
        Returns:
            bytes: JPEG image data
        """
        frame = self.read()
        if frame is None:
            return None
            
        ret, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not ret:
            return None
            
        return jpeg.tobytes()
    
    def get_base64_jpeg(self, quality=90):
        """
        Convert the latest frame to base64-encoded JPEG.
        
        Returns:
            str: Base64 encoded JPEG image
        """
        jpeg = self.get_jpeg(quality)
        if jpeg is None:
            return None
            
        return base64.b64encode(jpeg).decode('utf-8')
    
    def is_running(self):
        """Check if the stream is active."""
        return self.running and self.thread and self.thread.is_alive()
    
    def save_frame(self, path, filename=None):
        """
        Save the current frame to disk.
        
        Args:
            path (str): Directory path
            filename (str, optional): Filename, defaults to timestamp
            
        Returns:
            str: Path to saved file
        """
        frame = self.read()
        if frame is None:
            return None
            
        if filename is None:
            filename = f"frame_{int(time.time())}.jpg"
            
        save_path = Path(path) / filename
        Path(path).mkdir(parents=True, exist_ok=True)
        
        cv2.imwrite(str(save_path), frame)
        return str(save_path) 