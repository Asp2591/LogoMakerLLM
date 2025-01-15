from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from huggingface_hub import InferenceClient
import os

# Initialize FastAPI
app = FastAPI()

# Allow CORS for all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this to specific domains for better security in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the InferenceClient
token = os.getenv("HF_API_TOKEN")

client = InferenceClient("prithivMLmods/Flux-Dalle-Mix-LoRA", token=token)

# Define a request model
class ImageRequest(BaseModel):
    prompt: str

# Serve the HTML file
@app.get("/", response_class=HTMLResponse)
async def read_index():
    try:
        # Ensure the file exists
        with open("main.html", "r") as file:
            content = file.read()
        return HTMLResponse(content=content)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="HTML file not found.")

# Endpoint to generate an image
@app.post("/generate-image/")
async def generate_image(request: ImageRequest):
    try:
        # Generate the image from the prompt
        image = client.text_to_image(request.prompt)

        # Save the generated image
        output_path = "output.png"
        image.save(output_path)

        # Return the generated image as a response
        return FileResponse(output_path, media_type="image/png", filename="generated_logo.png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate image: {str(e)}")

# Run the application with: uvicorn app:app --reload
