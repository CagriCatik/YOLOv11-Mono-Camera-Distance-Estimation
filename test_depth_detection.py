import os
import numpy as np
from ultralytics import YOLO
import cv2
from PIL import Image
import torch  
import DepthPro
import logging 

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_model(model_path):
    """Loads the YOLO model from the specified path."""
    if not os.path.exists(model_path):
        logging.error(f"Model file not found at {model_path}")
        raise FileNotFoundError(f"Model file not found at {model_path}")
    logging.info(f"Loading model from {model_path}")
    return YOLO(model_path)

def load_image(image_path):
    """Loads an image from the specified path."""
    if not os.path.exists(image_path):
        logging.error(f"Image file not found at {image_path}")
        raise FileNotFoundError(f"Image file not found at {image_path}")
    logging.info(f"Loading image from {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        logging.error(f"Error loading image from {image_path}")
        raise ValueError(f"Error loading image from {image_path}")
    return image

def resize_image(image):
    """Resizes the image by the given scale factor."""
    new_width = int(image.shape[1])
    new_height = int(image.shape[0])
    logging.info(f"Resizing image to {new_width}x{new_height}")
    return cv2.resize(image, (new_width, new_height))

def process_results(results, image, model):
    """Draws bounding boxes for 'person' detections on the image."""
    person_boxes = []
    logging.info("Processing detection results")
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()  # Bounding boxes
        classes = result.boxes.cls.cpu().numpy()  # Class ids
        for box, cls in zip(boxes, classes):
            if model.names[int(cls)] == 'person':  # Use `model.names` for class names
                x1, y1, x2, y2 = map(int, box[:4])
                logging.info(f"Detected person at {(x1, y1, x2, y2)}")
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw rectangle
                person_boxes.append((x1, y1, x2, y2))
    return image, person_boxes

def save_image(output_path, image):
    """Saves the processed image to the specified output path."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, image)
    logging.info(f"Result saved to {output_path}")

def add_depth_information(image, person_boxes, depth):
    """Adds depth information to person bounding boxes."""
    for x1, y1, x2, y2 in person_boxes:
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        depth_value = depth[center_y, center_x]
        
        text = f'Depth: {depth_value:.2f}m'
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.2
        font_thickness = 2
        text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
        cv2.putText(image, text, (x1, y1 - 10), font, font_scale, (0, 255, 0), font_thickness)
        logging.info(f"Added depth information: {text} at {(x1, y1)}")

def visualize_depth(depth):
    """Visualizes the depth map."""
    depth_np_normalized = (depth - depth.min()) / (depth.max() - depth.min())
    inv_depth_np_normalized = 1.0 - depth_np_normalized
    depth_colormap = cv2.applyColorMap((inv_depth_np_normalized * 255).astype(np.uint8), cv2.COLORMAP_TURBO)

    # Display the colormapped inverted depth map
    logging.info("Visualizing inverted depth map")
    #cv2.imshow('Inverted Depth Map', depth_colormap)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Save the colormapped image to file
    cv2.imwrite('data_output/inverted_depth_map.jpg', depth_colormap)
    logging.info("Inverted depth map saved to 'inverted_depth_map.jpg'")

def run_inference_on_image(model_path, image_path, output_path, show_image=False):
    """Runs inference on the image, processes results, and integrates depth information."""
    logging.info("Starting inference process")

    # Load model and image
    model = load_model(model_path)
    
    # Check if GPU is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")
    
    # Move YOLO model to GPU if available
    model.to(device)

    image = load_image(image_path)

    # Run YOLO inference
    logging.info("Running YOLO inference")
    results = model.predict(image)

    # Process results and extract bounding boxes
    processed_image, person_boxes = process_results(results, image, model)

    # Load depth model and move it to GPU
    logging.info("Loading depth model")
    depth_model, transform = DepthPro.create_model_and_transforms()
    depth_model = depth_model.to(device)  # Move depth model to GPU
    depth_model.eval()

    # Load RGB image and transform for depth model
    rgb_image, _, f_px = DepthPro.load_rgb(image_path)
    depth_input = transform(rgb_image).to(device)  # Move depth input to GPU
    logging.info("Running depth inference")
    prediction = depth_model.infer(depth_input, f_px=f_px)
    depth = prediction["depth"].squeeze().cpu().numpy()

    # Add depth information to the detections
    add_depth_information(processed_image, person_boxes, depth)

    # Save the processed image with person detections and depth info
    save_image(output_path, processed_image)

    # Optionally display the result
    if show_image:
        logging.info("Displaying the processed image with detections and depth information")
        #cv2.imshow('Person Detection with Depth', processed_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Visualize depth map
    visualize_depth(depth)

# Configuration parameters
model_path = "checkpoints/yolo11s.pt"
image_path = "data_input/example_single_beach.jpg"
output_path = "data_output/result_yolo11s_with_depth.png"

# Run the YOLO inference with depth estimation
run_inference_on_image(model_path, image_path, output_path, show_image=True)
