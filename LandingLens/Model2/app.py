from PIL import Image
from landingai.predict import Predictor
from landingai.visualize import overlay_predictions
from landingai.postprocess import segmentation_class_pixel_coverage
import matplotlib.pyplot as plt
import os
from pathlib import Path

ENDPOINT_ID = os.getenv("ENDPOINT_ID")
API_KEY = os.getenv("API_KEY")

# Load all images from penguin pics directory
image_files = list(Path("../penguin-pics").glob("*.jpg"))

open_images = [
    Image.open(image_files[0]),
    Image.open(image_files[1]),
    Image.open(image_files[2]),
]  # open 3 for now

predictions = []

# Run inference on all images
predictor = Predictor(ENDPOINT_ID, api_key=API_KEY)

color_map = {"Penguin": "blue"}

options = {"color_map": color_map}

for img in open_images:
    predictions.extend(predictor.predict(img))

    # display(overlay_predictions(predictions, img, options=options))


coverage = segmentation_class_pixel_coverage(predictions)
coverage
