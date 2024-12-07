import os
import shutil
from fastapi.responses import JSONResponse
from fastapi import FastAPI, HTTPException
from fastapi import FastAPI, UploadFile, File, HTTPException
import torchaudio
from speechbrain.inference import SpeakerRecognition

# Initialize FastAPI
app = FastAPI()

# Load the pre-trained speaker recognition model
speaker_verification = SpeakerRecognition.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="tmp_model"
    
)

# Temporary folder to save uploaded files
UPLOAD_FOLDER = "uploaded_audio"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Function to calculate similarity score
def get_similarity(audio_path1: str, audio_path2: str):
    try:
        # Load audio files
        signal1, _ = torchaudio.load(audio_path1)
        signal2, _ = torchaudio.load(audio_path2)

        # Get similarity score and prediction
        score, prediction = speaker_verification.verify_batch(signal1, signal2)
        return float(score), bool(prediction)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up temporary files
        if os.path.exists(audio_path1):
            os.remove(audio_path1)
        if os.path.exists(audio_path2):
            os.remove(audio_path2)

# Define FastAPI endpoint
@app.post("/compare-voices")
async def compare_voices(
    file1: UploadFile = File(..., description="First audio file"),
    file2: UploadFile = File(..., description="Second audio file")
):
    """
    Compare voices from two uploaded audio files and return similarity score and prediction.
    """
    try:
        # Save uploaded files temporarily
        file1_path = os.path.join(UPLOAD_FOLDER, file1.filename)
        file2_path = os.path.join(UPLOAD_FOLDER, file2.filename)

        with open(file1_path, "wb") as f1:
            shutil.copyfileobj(file1.file, f1)
        with open(file2_path, "wb") as f2:
            shutil.copyfileobj(file2.file, f2)

        # Get similarity score
        score, is_same_user = get_similarity(file1_path, file2_path)

        return JSONResponse(
            {
                "Similarity Score": f"{score:.4f}",
                "Same User Prediction": "Yes" if is_same_user else "No"
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
