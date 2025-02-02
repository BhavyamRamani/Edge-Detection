from fastapi import FastAPI, File, UploadFile, Response
from fastapi.responses import FileResponse
import cv2
import numpy as np
from io import BytesIO
from PIL import Image
import os
 
app = FastAPI()
 
@app.post("/process_image/")
async def process_image(file: UploadFile = File(...)):
    """Process an image and return the edges as a file."""
    # Read the image
    contents = await file.read()
    image = Image.open(BytesIO(contents)).convert('L')  # Convert to grayscale
    image = np.array(image)
 
    # Apply Gaussian blur
    blur_image = cv2.GaussianBlur(image, (3, 3), 0)
 
    # Detect edges using Canny
    edges = cv2.Canny(blur_image, 100, 200)
 
    # Save the processed image to a temporary file
    output_path = "processed_image.png"  # Choose a temporary file path
    cv2.imwrite(output_path, edges)
 
    # Return a FileResponse
    return FileResponse(output_path, media_type="image/png", filename="processed_image.png")
