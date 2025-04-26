#!/usr/bin/env python3
import streamlit as st
import cv2
import numpy as np
import time
import pandas as pd
from PIL import Image
import io
from pathlib import Path
import os
import tempfile
from ultralytics import YOLO
from scipy.spatial import distance

# Set page configuration
st.set_page_config(
    page_title="Parking Space Counter",
    page_icon="🅿️",
    layout="wide",
    initial_sidebar_state="expanded",
)

@st.cache_resource
def load_model():
    """Load the YOLOv8 model (cached to avoid reloading)."""
    model = YOLO("models/parking_detector3/weights/best.pt")
    return model

def find_parking_spots(image, occupied_boxes):
    """
    Identify and locate empty parking spots based on parking lot layout and occupied vehicles.
    
    Args:
        image: The input image
        occupied_boxes: List of bounding boxes for occupied spaces
        
    Returns:
        empty_spots: List of bounding boxes for empty parking spaces
    """
    # If no occupied boxes were detected, we can't reliably find empty spaces
    if not occupied_boxes or len(occupied_boxes) < 2:
        return []
    
    # Get image dimensions
    height, width = image.shape[:2]
    
    # Calculate average vehicle size and position
    avg_width = sum(box[2] - box[0] for box in occupied_boxes) / len(occupied_boxes)
    avg_height = sum(box[3] - box[1] for box in occupied_boxes) / len(occupied_boxes)
    
    # Find the average Y coordinate range where vehicles are parked (parking lot ground level)
    y_positions = [(box[1] + box[3]) / 2 for box in occupied_boxes]
    avg_y = sum(y_positions) / len(y_positions)
    min_y = min(box[1] for box in occupied_boxes) - avg_height * 0.5
    max_y = max(box[3] for box in occupied_boxes) + avg_height * 0.5
    
    # Extract parking rows by clustering the Y-coordinates of detected vehicles
    y_values = [box[1] for box in occupied_boxes] + [box[3] for box in occupied_boxes]
    y_values.sort()
    
    # Find potential parking rows (areas with consistent Y values)
    parking_rows = []
    current_row = [y_values[0]]
    
    for i in range(1, len(y_values)):
        if y_values[i] - y_values[i-1] < avg_height * 0.4:  # If Y values are close, same row
            current_row.append(y_values[i])
        else:
            # New row found
            if len(current_row) >= 3:  # Enough points to be considered a row
                row_avg = sum(current_row) / len(current_row)
                parking_rows.append(row_avg)
            current_row = [y_values[i]]
    
    # Add the last row if it has enough points
    if len(current_row) >= 3:
        row_avg = sum(current_row) / len(current_row)
        parking_rows.append(row_avg)
    
    # Group occupied boxes by rows
    row_boxes = {}
    for row_y in parking_rows:
        row_boxes[row_y] = []
        for box in occupied_boxes:
            box_center_y = (box[1] + box[3]) / 2
            if abs(box_center_y - row_y) < avg_height * 0.6:
                row_boxes[row_y].append(box)
    
    # Find potential empty spaces in each row
    empty_spots = []
    
    for row_y, boxes in row_boxes.items():
        if len(boxes) < 2:
            continue
            
        # Sort boxes by x-coordinate
        sorted_boxes = sorted(boxes, key=lambda box: box[0])
        
        # Find gaps between vehicles in this row
        for i in range(len(sorted_boxes) - 1):
            current_box = sorted_boxes[i]
            next_box = sorted_boxes[i+1]
            
            gap_width = next_box[0] - current_box[2]
            
            # Check if gap is wide enough for a vehicle
            if gap_width >= avg_width * 0.8:
                # This is a potential empty space
                num_spaces = max(1, round(gap_width / avg_width))
                
                for j in range(num_spaces):
                    space_width = gap_width / num_spaces
                    
                    # Define the empty space box
                    x1 = int(current_box[2] + j * space_width)
                    x2 = int(min(x1 + space_width, next_box[0]))
                    y1 = int(row_y - avg_height / 2)
                    y2 = int(row_y + avg_height / 2)
                    
                    # Make sure it's a reasonable size
                    if (x2 - x1) > avg_width * 0.7 and (y2 - y1) > avg_height * 0.7:
                        # Check it's in the valid parking area (ground level, not building)
                        if min_y <= y1 <= max_y and min_y <= y2 <= max_y:
                            empty_spots.append([x1, y1, x2, y2])
        
        # Check for potential spaces at the beginning and end of rows
        # Left edge
        first_box = sorted_boxes[0]
        if first_box[0] > avg_width:  # If there's enough space before the first car
            x1 = int(max(0, first_box[0] - avg_width))
            x2 = int(first_box[0])
            y1 = int(row_y - avg_height / 2)
            y2 = int(row_y + avg_height / 2)
            
            if min_y <= y1 <= max_y and min_y <= y2 <= max_y:
                empty_spots.append([x1, y1, x2, y2])
        
        # Right edge
        last_box = sorted_boxes[-1]
        if width - last_box[2] > avg_width:  # If there's enough space after the last car
            x1 = int(last_box[2])
            x2 = int(min(width, last_box[2] + avg_width))
            y1 = int(row_y - avg_height / 2)
            y2 = int(row_y + avg_height / 2)
            
            if min_y <= y1 <= max_y and min_y <= y2 <= max_y:
                empty_spots.append([x1, y1, x2, y2])
    
    # Remove any spots that would overlap with buildings or non-parking areas
    # by checking if they're in the same general Y range as the detected vehicles
    filtered_spots = []
    for spot in empty_spots:
        spot_center_y = (spot[1] + spot[3]) / 2
        
        # Check if spot is in the same general area as detected vehicles
        if min_y <= spot_center_y <= max_y:
            # Make sure the spot doesn't fall outside the image
            spot[0] = max(0, spot[0])
            spot[1] = max(0, spot[1])
            spot[2] = min(width, spot[2])
            spot[3] = min(height, spot[3])
            
            if spot[2] - spot[0] > 30 and spot[3] - spot[1] > 30:  # Minimum size check
                filtered_spots.append(spot)
    
    # Further filter spots to ensure they don't overlap with each other
    final_spots = []
    for spot in filtered_spots:
        overlapped = False
        for existing_spot in final_spots:
            # Calculate intersection
            x1 = max(spot[0], existing_spot[0])
            y1 = max(spot[1], existing_spot[1])
            x2 = min(spot[2], existing_spot[2])
            y2 = min(spot[3], existing_spot[3])
            
            if x1 < x2 and y1 < y2:
                overlap_area = (x2 - x1) * (y2 - y1)
                spot_area = (spot[2] - spot[0]) * (spot[3] - spot[1])
                
                if overlap_area > 0.3 * spot_area:
                    overlapped = True
                    break
        
        if not overlapped:
            final_spots.append(spot)
    
    # Limit the number of empty spots to a reasonable number relative to occupied spots
    # In most parking lots, there's usually not more empty spaces than occupied ones
    max_empty_spots = max(3, len(occupied_boxes) // 2)
    if len(final_spots) > max_empty_spots:
        final_spots = final_spots[:max_empty_spots]
    
    return final_spots

def process_uploaded_image(image_bytes, use_manual_total=False, manual_total=None, show_empty_spaces=True):
    """
    Process an uploaded image and detect parking spaces.
    
    Args:
        image_bytes: Uploaded image bytes
        use_manual_total: Whether to use manually set total spaces
        manual_total: User-provided total number of parking spaces
        show_empty_spaces: Whether to highlight empty spaces
        
    Returns:
        annotated_frame: Image with annotations
        detections: List of detected vehicles
        occupied_count: Number of occupied spaces
        total_spaces: Total number of parking spaces
        empty_spots: List of detected empty spaces
    """
    # Convert the image bytes to OpenCV format
    image = Image.open(io.BytesIO(image_bytes))
    image_array = np.array(image)
    
    # Convert RGB to BGR for OpenCV
    if len(image_array.shape) == 3 and image_array.shape[2] == 3:
        image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
    
    # Get model
    model = load_model()
    
    # Process the image with the model
    results = model(image_array, conf=0.25, verbose=False)
    
    # Create a copy for annotation
    annotated_frame = image_array.copy()
    
    # Count occupied parking spaces (detected vehicles)
    detections = []
    occupied_boxes = []
    
    for result in results:
        boxes = result.boxes
        for i, box in enumerate(boxes):
            cls = int(box.cls[0].item())
            class_name = model.names[cls]
            
            # If the detected object is an occupied parking space or a vehicle
            if class_name in ['occupied', 'car', 'truck', 'bus', 'motorcycle']:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = float(box.conf[0].item())
                
                # Convert to integers
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # Add to detections
                detections.append({
                    'class': class_name,
                    'confidence': confidence,
                    'bbox': [x1, y1, x2, y2]
                })
                
                occupied_boxes.append([x1, y1, x2, y2])
                
                # Draw red rectangle for occupied space
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(annotated_frame, f"{class_name}", 
                           (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    # Count occupied spaces
    occupied_count = len(detections)
    
    # Find empty parking spots if requested
    empty_spots = []
    if show_empty_spaces:
        # First add any vacant spaces detected by the model
        for result in results:
            boxes = result.boxes
            for i, box in enumerate(boxes):
                cls = int(box.cls[0].item())
                class_name = model.names[cls]
                
                if class_name == 'vacant':
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    
                    # Convert to integers
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    
                    # Add to empty spots
                    empty_spots.append([x1, y1, x2, y2])
        
        # If we have occupied spots but not enough empty spots found by the model,
        # try to estimate additional empty spots
        if occupied_count > 0 and len(empty_spots) < 3:
            estimated_empty_spots = find_parking_spots(image_array, occupied_boxes)
            empty_spots.extend(estimated_empty_spots)
            
        # Draw green rectangles for empty spaces
        for spot in empty_spots:
            x1, y1, x2, y2 = spot
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated_frame, "Available", 
                       (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Determine total spaces
    total_detected = occupied_count + len(empty_spots)
    
    if use_manual_total and manual_total is not None and manual_total >= occupied_count:
        total_spaces = manual_total
    else:
        # If no manual setting or invalid setting, use detected total
        # with a small buffer for potentially missed spaces
        total_spaces = max(total_detected, int(occupied_count * 1.2))
    
    # Calculate available spaces
    available_spaces = total_spaces - occupied_count
    
    # Add summary text to image
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(annotated_frame, f"Occupied: {occupied_count}", (10, 30), font, 1, (0, 0, 255), 2)
    cv2.putText(annotated_frame, f"Available: {available_spaces}", (10, 70), font, 1, (0, 255, 0), 2)
    cv2.putText(annotated_frame, f"Total: {total_spaces}", (10, 110), font, 1, (255, 255, 255), 2)
    
    if use_manual_total:
        cv2.putText(annotated_frame, "Manual setting applied", (10, 150), font, 0.7, (0, 255, 255), 2)
    
    # Convert back to RGB for display
    annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
    
    return annotated_frame, detections, occupied_count, total_spaces, empty_spots

def main():
    # Add custom CSS
    st.markdown("""
    <style>
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .big-font {
        font-size: 4rem !important;
        font-weight: bold;
        text-align: center;
    }
    .medium-font {
        font-size: 1.5rem !important;
        text-align: center;
    }
    .metrics-container {
        display: flex;
        justify-content: space-between;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    .metric-box {
        padding: 1rem;
        border-radius: 5px;
        text-align: center;
        flex: 1;
        margin: 0 0.5rem;
    }
    .occupied {
        background-color: rgba(255, 0, 0, 0.1);
        border: 1px solid red;
    }
    .available {
        background-color: rgba(0, 255, 0, 0.1);
        border: 1px solid green;
    }
    .total {
        background-color: rgba(0, 0, 255, 0.1);
        border: 1px solid blue;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # App header
    st.title("Parking Space Counter")
    st.subheader("Upload an image of a parking lot to count occupied and available spaces")
    
    # Initialize session state for storing application state
    if 'manual_total' not in st.session_state:
        st.session_state['manual_total'] = 50
    if 'use_manual_total' not in st.session_state:
        st.session_state['use_manual_total'] = False
    if 'show_empty_spaces' not in st.session_state:
        st.session_state['show_empty_spaces'] = True
    
    # Sidebar
    with st.sidebar:
        st.header("Settings")
        
        # Option to manually set total parking spaces
        use_manual_total = st.checkbox("Manually set total parking spaces", value=st.session_state['use_manual_total'])
        st.session_state['use_manual_total'] = use_manual_total
        
        if use_manual_total:
            manual_total = st.number_input(
                "Total parking spaces", 
                min_value=1, 
                value=st.session_state['manual_total']
            )
            st.session_state['manual_total'] = manual_total
        else:
            manual_total = None
        
        # Option to show empty spaces
        show_empty_spaces = st.checkbox("Show available parking spaces", value=st.session_state['show_empty_spaces'])
        st.session_state['show_empty_spaces'] = show_empty_spaces
        
        if show_empty_spaces:
            st.info("Green boxes show detected available parking spaces. For better accuracy, use images with clearly visible parking spaces.")
        
        st.header("How It Works")
        st.markdown(
            "1. Upload a parking lot image\n"
            "2. Red boxes: Detected occupied spaces\n"
            "3. Green boxes: Estimated available spaces\n"
            "4. For accuracy, set the exact total number of spaces manually\n"
        )
    
    # Main content - centered upload
    col1, col2 = st.columns([3, 2])
    
    with col1:
        # Image upload
        uploaded_file = st.file_uploader("Upload a parking lot image", type=["jpg", "jpeg", "png"])
        
        if uploaded_file:
            # Process the image
            with st.spinner("Analyzing parking lot..."):
                processed_frame, detections, occupied_count, total_spaces, empty_spots = process_uploaded_image(
                    uploaded_file.getvalue(),
                    use_manual_total=st.session_state['use_manual_total'],
                    manual_total=st.session_state['manual_total'] if st.session_state['use_manual_total'] else None,
                    show_empty_spaces=st.session_state['show_empty_spaces']
                )
            
            # Display the results
            st.image(processed_frame, caption="Parking Space Analysis", use_column_width=True)
    
    with col2:
        if uploaded_file:
            # Calculate available spaces
            available_spaces = total_spaces - occupied_count
            
            # Display metrics with custom styling
            st.markdown(
                f"""
                <div class="metrics-container">
                    <div class="metric-box occupied">
                        <h3>Occupied Spaces</h3>
                        <p class="big-font">{occupied_count}</p>
                    </div>
                    <div class="metric-box available">
                        <h3>Available Spaces</h3>
                        <p class="big-font">{available_spaces}</p>
                    </div>
                    <div class="metric-box total">
                        <h3>Total Spaces</h3>
                        <p class="big-font">{total_spaces}</p>
                    </div>
                </div>
                """, 
                unsafe_allow_html=True
            )
            
            # Display occupancy percentage
            occupancy_percent = (occupied_count / total_spaces) * 100 if total_spaces > 0 else 0
            availability_percent = 100 - occupancy_percent
            
            # Progress bar for availability
            st.subheader("Parking Lot Status")
            st.progress(availability_percent / 100)
            st.markdown(f"<p style='text-align: center;'>{availability_percent:.1f}% Available</p>", unsafe_allow_html=True)
            
            # Show detected empty spaces info
            if st.session_state['show_empty_spaces']:
                st.subheader(f"Detected Available Spaces: {len(empty_spots)}")
                
                if len(empty_spots) == 0 and occupied_count > 0:
                    st.info("No available spaces detected in the parking area. The lot appears to be full.")
                else:
                    st.write("Green boxes on the image show detected available parking spaces.")
            
            # Usage note
            if not st.session_state['use_manual_total']:
                st.warning("Note: Total spaces are estimated. For accurate results, check 'Manually set total parking spaces' in the sidebar and enter the correct number.")
            
            # Display breakdown by vehicle type
            if detections:
                vehicle_counts = {}
                for detection in detections:
                    vehicle_class = detection['class']
                    if vehicle_class in vehicle_counts:
                        vehicle_counts[vehicle_class] += 1
                    else:
                        vehicle_counts[vehicle_class] = 1
                
                st.subheader("Detected Vehicle Types:")
                
                # Create columns for vehicle types
                cols = st.columns(len(vehicle_counts))
                for i, (vehicle_type, count) in enumerate(vehicle_counts.items()):
                    with cols[i]:
                        st.metric(f"{vehicle_type.capitalize()}", count)

if __name__ == "__main__":
    main() 