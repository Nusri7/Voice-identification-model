from fastapi import FastAPI, File, UploadFile
from speechbrain.inference import SpeakerRecognition
import io
import numpy as np
import torch
from fastapi.responses import JSONResponse

# Initialize FastAPI app
app = FastAPI()

# Load the pretrained speaker recognition model
verification = SpeakerRecognition.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb", 
    savedir="pretrained_models/spkrec-ecapa-voxceleb"
)

# Function to compare two audio files for speaker verification
def verify_speaker(file1: UploadFile, file2: UploadFile):
    # Read audio files into memory
    audio1 = file1.file.read()
    audio2 = file2.file.read()

    # Load the audio files as waveforms
    wav1, fs1 = verification.load_audio(io.BytesIO(audio1))
    wav2, fs2 = verification.load_audio(io.BytesIO(audio2))

    # Compute speaker embeddings
    embedding1 = verification.encode_file(wav1)
    embedding2 = verification.encode_file(wav2)

    # Compare the embeddings to calculate the similarity score
    score, prediction = verification.verify_batch(embedding1, embedding2)

    return score.item(), prediction.item()

# Create an endpoint for speaker verification
@app.post("/verify-speech/")
async def verify_speech(file1: UploadFile = File(...), file2: UploadFile = File(...)):
    try:
        score, prediction = verify_speaker(file1, file2)

        result = {
            "similarity_score": score,
            "prediction": "Same speaker" if prediction == 1 else "Different speakers"
        }
        return JSONResponse(content=result)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

# Optionally, create a home route
@app.get("/")
def read_root():
    return {"message": "Welcome to the Speaker Verification API!"}
