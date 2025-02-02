from fastapi import FastAPI, File, UploadFile, Response
from fastapi.responses import StreamingResponse
import cv2
import numpy as np
import io

app = FastAPI()

@app.post("/process_image/")
async def process_image(file: UploadFile = File(...)):
  # Read the image file
  contents = await file.read()
  nparr = np.fromstring(contents, np.uint8)
  image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

  # Convert the image to grayscale
  gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

  # Apply Gaussian blur
  blur_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

  # Detect edges using Canny edge detection
  edges = cv2.Canny(blur_image, 10, 70)

  # Save the processed image to an in-memory buffer
  _, buffer = cv2.imencode(".jpg", edges)

  # Create a generator function to stream the image data
  def iterfile():
    yield buffer.tobytes()

  # Return the image as a streaming response
  return StreamingResponse(iterfile(), media_type="image/jpeg")
