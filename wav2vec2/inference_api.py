import os
import shutil
import tempfile
from pathlib import Path

import librosa
import torch
from deep_utils import PickleUtils
from fastapi import FastAPI, UploadFile, File, HTTPException
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor

from settings import Config

# Initialize FastAPI app
app = FastAPI()

# Load configurations and model
config = Config()
inference_dir = Path("./results/best")
label2id = PickleUtils.load_pickle(inference_dir / "label2id.pkl")
id2label = {int(v): k for k, v in label2id.items()}
feature_extractor = AutoFeatureExtractor.from_pretrained(config.feature_extractor)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModelForAudioClassification.from_pretrained(
    inference_dir,
    num_labels=len(label2id),
    label2id=label2id,
    id2label=id2label
).to(device).eval()


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


@app.post("/predict-audio/")
async def predict_audio(file: UploadFile = File(...)):
    try:
        # Create a temporary directory to store the uploaded file
        # with tempfile.TemporaryDirectory() as tmpdir:
        # Define the path for the temporary file
        # temp_file_path = Path(tmpdir) / file.filename

        # Save the uploaded file to the temporary directory
        with tempfile.NamedTemporaryFile(delete=True, suffix=".wav") as temp_file:
            shutil.copyfileobj(file.file, temp_file)
            # Process the audio file
            audio_array = get_audio(temp_file.name)

        # Prepare the audio tensor for model inference
        with torch.no_grad():
            audio_tensor = torch.tensor(audio_array).to(device=device)
            audio_tensor = audio_tensor[None, ...]
            output = model(audio_tensor)
            logits = output.logits[0]
            cls_index = torch.argmax(logits).item()
            cls_name = id2label[cls_index]

        # Return the prediction result
        return {"class_index": cls_index, "class_name": cls_name}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == '__main__':
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
