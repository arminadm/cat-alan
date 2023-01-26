# ai
import torch
import mlflow
import torchaudio
import streamlit as st

@st.cache(suppress_st_warning=True)
def load_model():
    model_path = "examples/model"
    model = mlflow.pytorch.load_model(model_path, map_location="cpu")
    # Switch off dropout
    model.eval()
    return model

classes = {
    0: "Happy",
    1: "Resting",
    2: "Angry",
    3: "Paining",
    4: "Mother_Call",
    5: "Warning",
    6: "Hunting",
    7: "Fighting",
    8: "Defence",
    9: "Mating",
}

emojis = {
    0: "🐱 : Cat is happy.",
    1: "💤 : Cat is tired.",
    2: "😾 : Cat is angry.",
    3: "😿 : Cat sounds like it's in pain.",
    4: "🙀 : Cat is calling for mum.",
    5: "⚠️ : Cat is giving you a warning." ,
    6: "😼 : Cat wants to hunt.",
    7: "⚔️ : Cat is about to throw hands.",
    8: "🛡️  : Cat is on the defence.",
    9: "😻 : Cat wants to mate.",
}


def preprocess_audio(audio):
    device = "cpu"
    mono = torch.mean(audio, axis=0, keepdim=True)
    reshaped = torch.unsqueeze(mono, 0)
    return reshaped.to(device)


def predict(model, audio):
    log_softmax = model(audio)
    probabilities = torch.squeeze(torch.exp(log_softmax))
    return probabilities.cpu().detach().numpy()


model = load_model()

# backend 
from flask import Flask, request
from datetime import datetime
from hashlib import sha256
import os

app = Flask(__name__)

@app.route('/upload_cat_voice', methods=['POST', 'GET'])
def upload_cat_voice():
    if request.method == 'POST':
        uploaded_file = request.files['cat_voice']
        extension = uploaded_file.filename.split('.')[-1]
        
        if extension in ["mp3", "wav", "m4a"]:
            # generate unique audio name for ignoring conflict
            temp_file_name = sha256(str(datetime.now()).encode('utf8')).hexdigest()
            temp_audio_path = f"save_temp_files/{temp_file_name}.{extension}"    
            # Save the audio file so we can access the array data
            uploaded_file.save(temp_audio_path)
        else:
            return {"error": "extension is not supported"}
        
        audio, _ = torchaudio.load(temp_audio_path)

        # remove the audio after we're done
        os.remove(temp_audio_path)

        # ai progress
        model_input = preprocess_audio(audio)
        probabilities = predict(model, model_input)

        sorted_pairs = sorted(zip(classes.values(), probabilities), key=lambda x: x[1])
        
        # converting sorted pairs to json serializable pairs
        sorted_pairs_standard = {}
        for item in sorted_pairs:
            # we have to convert float32 to float
            if item[1] <= 0.0001:
                # if number is too small we consider it as 0
                sorted_pairs_standard[item[0]] = 0
            else:
                sorted_pairs_standard[item[0]] = round(float(item[1] * 100.0), 1)
        
        # find the top probabilities
        top_probability = sorted_pairs[9]
        key, value = top_probability
        sorted_pairs_standard['top'] = key
        
        return sorted_pairs_standard

if __name__ == '__main__':
    app.run(port=8080, host="0.0.0.0", debug=False)