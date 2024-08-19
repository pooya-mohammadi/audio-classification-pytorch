from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Request, Response
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
import torch
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor
from settings import Config
from pathlib import Path
import librosa
import numpy as np
from deep_utils import PickleUtils
import tempfile
import shutil
import os
import secrets

# Initialize FastAPI app without the security scheme in the OpenAPI docs
app = FastAPI(
    title="Audio Classification API",
    description="API for audio sentiment classification",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# Define the origins that should be allowed to make requests to your API
origins = [
    "http://localhost:5173",  # Your frontend URL
    # Add more origins as needed
]

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Allows specific origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers (Content-Type, Authorization, etc.)
)

# HTTP Basic Authentication
security = HTTPBasic()

# Define a username and password for authentication
USERNAME = "user"
PASSWORD = "password"

# Load configurations and model
config = Config()
inference_dir = Path("./results/exp_0/best")
label2id = PickleUtils.load_pickle(inference_dir / "label2id.pkl")
id2label = {int(v): k for k, v in label2id.items()}
feature_extractor = AutoFeatureExtractor.from_pretrained(config.feature_extractor)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModelForAudioClassification.from_pretrained(
    inference_dir,
    num_labels=len(label2id),
    label2id=label2id,
    id2label=id2label
)
model = model.to(device)

sentiment_classes_mapping = {
    'S': 'sadness', 'A': 'anger',
    'H': 'happiness', 'W': 'surprise',
    'F': 'fear', 'N': 'neutral'
}

def authenticate(credentials: HTTPBasicCredentials = Depends(security)):
    correct_username = secrets.compare_digest(credentials.username, USERNAME)
    correct_password = secrets.compare_digest(credentials.password, PASSWORD)
    if not (correct_username and correct_password):
        raise HTTPException(
            status_code=401,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username

# Apply global authentication middleware
@app.middleware("http")
async def basic_auth_middleware(request: Request, call_next):
    # List of paths that need authentication
    paths_to_authenticate = ["/docs", "/redoc", "/openapi.json", "/predict-audio/"]
    
    # Check if the request path is one that requires authentication
    if any(request.url.path.startswith(path) for path in paths_to_authenticate):
        # Extract the credentials
        credentials = security(request)
        try:
            # Attempt to authenticate
            authenticate(await credentials)
        except HTTPException as e:
            # Return a response that prompts the client to authenticate
            return Response(
                content="Unauthorized",
                status_code=401,
                headers={"WWW-Authenticate": "Basic"},
            )
    
    response = await call_next(request)
    return response

@app.post("/predict-audio/")
async def predict_audio(
    file: UploadFile = File(...), 
    username: str = Depends(authenticate)
):
    try:
        # Create a temporary file to store the uploaded audio
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            shutil.copyfileobj(file.file, temp_file)

        # Process the audio file
        audio_array = get_audio(temp_file.name)
        os.remove(temp_file.name)
        
        # Prepare the audio tensor for model inference
        with torch.no_grad():
            audio_tensor = torch.tensor(audio_array).to(device=device)
            audio_tensor = audio_tensor[None, ...]
            output = model(audio_tensor)
            logits = output.logits[0]
            cls_index = torch.argmax(logits).item()
            cls_name = id2label[cls_index]

        # Return the prediction result
        return {"class_name": sentiment_classes_mapping[cls_name]}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def get_audio(file_path: str):
    audio, _ = librosa.load(file_path, sr=feature_extractor.sampling_rate)
    inputs = feature_extractor(
        audio,
        sampling_rate=feature_extractor.sampling_rate,
        max_length=16000,
        truncation=True,
        return_tensors="pt"
    )
    return inputs['input_values'][0]

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
