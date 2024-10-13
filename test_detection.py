import os
import numpy as np
from ultralytics import YOLO
import cv2
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Set the logging level
    format='%(asctime)s - %(levelname)s - %(message)s',  # Define the log format
    handlers=[
        logging.FileHandler("inference.log"),  # Save logs to a file
        logging.StreamHandler()  # Output logs to the console
    ]
)

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

def resize_image(image, scale_factor):
    """Resizes the image by the given scale factor."""
    new_width = int(image.shape[1] * scale_factor)
    new_height = int(image.shape[0] * scale_factor)
    logging.info(f"Resizing image to {new_width}x{new_height}")
    return cv2.resize(image, (new_width, new_height))

def process_results(results, image, model):
    """Draws bounding boxes for 'person' detections on the image."""
    logging.info("Processing detection results")
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()  # Bounding boxes
        classes = result.boxes.cls.cpu().numpy()  # Class ids
        for box, cls in zip(boxes, classes):
            if model.names[int(cls)] == 'person':  # Use `model.names` for class names
                x1, y1, x2, y2 = map(int, box[:4])
                logging.debug(f"Detected 'person' at [{x1}, {y1}, {x2}, {y2}]")
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw rectangle
    return image

def save_image(output_path, image):
    """Saves the processed image to the specified output path."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, image)
    logging.info(f"Result saved to {output_path}")

def run_inference_on_image(model_path, image_path, output_path, scale_factor=1.0, show_image=False):
    """Runs inference on the image and processes the results."""
    # Load model and image
    logging.info("Starting inference process")
    model = load_model(model_path)
    image = load_image(image_path)

    # Optionally resize the image
    if scale_factor != 1.0:
        image = resize_image(image, scale_factor)

    # Run YOLO inference
    logging.info("Running YOLO inference")
    results = model.predict(image)

    # Process results (e.g., drawing bounding boxes)
    processed_image = process_results(results, image, model)

    # Save the processed image
    save_image(output_path, processed_image)

    # Optionally display the result
    if show_image:
        logging.info("Displaying processed image")
        cv2.imshow('Person Detection', processed_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# Configuration parameters
model_path = "checkpoints/yolo11s.pt"
image_path = "data_input/example_single_beach.jpg"
output_path = "data_output/result_example_single_beach.png"
scale_factor = 0.5  # Change scale as needed

# Run the YOLO inference
run_inference_on_image(model_path, image_path, output_path, scale_factor, show_image=True)
