from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import requests
import torch
from transformers import AutoTokenizer, AutoModel, pipeline
from datasets import load_dataset
import pandas as pd
import numpy as np
from numpy import dot
from numpy.linalg import norm
from fastapi.staticfiles import StaticFiles
from collections import defaultdict
import time
import urllib.parse
import os

# Import your recorder function
from static.recorder import record_and_transcribe

# ----------------------------
# Create the FastAPI app & Configure CORS
# ----------------------------
app = FastAPI(title="Medical Adverse Event Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development; restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create the static directory if it doesn't exist
static_dir = os.path.join(os.path.dirname(__file__), "static")
if not os.path.exists(static_dir):
    os.makedirs(static_dir)

app.mount("/static", StaticFiles(directory=static_dir), name="static")

# ----------------------------
# Utility Functions for Prediction
# ----------------------------
def cosine_similarity(vec1, vec2):
    """Compute cosine similarity between two vectors."""
    # Debug: check the types
    print("[DEBUG] In cosine_similarity:")
    print("        type(vec1) =", type(vec1), "shape:", getattr(vec1, "shape", None))
    print("        type(vec2) =", type(vec2), "shape:", getattr(vec2, "shape", None))
    return dot(vec1, vec2) / (norm(vec1) * norm(vec2))

def embed_text(text: str):
    """
    Returns a 512-dimensional vector for the given text using Bioformer‑8L.
    Uses the [CLS] token representation.
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].squeeze().numpy()

def find_similar_symptoms(user_symptom, df, top_k=3, threshold=0.75):
    user_emb = embed_text(user_symptom.lower())
    similarities = []
    for _, row in df.iterrows():
        sim_score = cosine_similarity(user_emb, row["symptom_embedding"])
        similarities.append((row["text"], row["label"], sim_score))
    similarities.sort(key=lambda x: x[2], reverse=True)
    return [match for match in similarities if match[2] >= threshold][:top_k]

def query_adverse_events(disease, limit=2, retries=3, api_key=None):
    """
    Query the openFDA FAERS API for adverse events related to a given disease.
    Handles missing results, encodes search terms correctly, and includes a retry mechanism.
    """
    base_url = "https://api.fda.gov/drug/event.json"
    encoded_disease = urllib.parse.quote(disease)
    
    params = {
        "search": f'patient.reaction.reactionmeddrapt:"{encoded_disease}"',
        "limit": limit
    }
    
    if api_key:
        params["api_key"] = api_key  # Use API key if available

    attempt = 0
    while attempt < retries:
        response = requests.get(base_url, params=params)
        
        print("[DEBUG] Received status code:", response.status_code)
        if response.status_code == 200:
            data = response.json()
            print("[DEBUG] query_adverse_events: JSON keys:", list(data.keys()))
            if "results" in data and data["results"]:
                return data  # Return only if data is present
            print(f"[DEBUG] No results found for '{disease}'. Returning error.")
            return {"error": f"No adverse events found for {disease}"}

        elif response.status_code == 404:
            print(f"[DEBUG] 404 Error: Disease '{disease}' not found in openFDA database.")
            return {"error": f"Disease '{disease}' not found in openFDA database."}
        elif response.status_code in [429, 500, 502, 503, 504]:
            time.sleep(2 ** attempt)
            attempt += 1
        else:
            return {"error": f"Unexpected API error: {response.status_code}"}
    return {"error": "API request failed after multiple attempts"}

# ----------------------------
# Multi-Symptom Extraction using NER
# ----------------------------
ner_pipeline = pipeline("ner", model="d4data/biomedical-ner-all", aggregation_strategy="simple")

def extract_symptoms_ner(text: str):
    """
    Extract multiple symptoms from text using a biomedical NER model.
    Returns a list of unique symptom strings.
    """
    print(f"[DEBUG] extract_symptoms_ner called with text[:50]='{text[:50]}'...")
    ner_results = ner_pipeline(text)
    symptoms = []
    for entity in ner_results:
        if entity.get("entity_group", "").upper() in ["SYMPTOM", "DISEASE", "PROBLEM", "FINDING"]:
            symptoms.append(entity["word"])
    
    text_lower = text.lower()
    
    # Additional medical keywords
    keywords = [
        "pain", "ache", "sore", "fever", "cough", "breathing", "breath", 
        "tired", "fatigue", "dizzy", "nausea", "vomit", "headache", "migraine",
        "asthma", "allergy", "swelling", "rash", "itching", "blood", 
        "pressure", "diabetes", "heart", "chest", "stomach", "back", 
        "joint", "muscle", "throat", "nose", "ear", "eye", "skin",
        "shaking", "shivering", "trembling", "sleep", "insomnia", "cold",
        "hot", "sweating", "chills", "numbness", "tingling", "burning"
    ]
    
    # Check for medical keywords
    words = text_lower.split()
    for i, word in enumerate(words):
        for keyword in keywords:
            if keyword in word and word not in [s.lower() for s in symptoms]:
                # Try to capture multi-word symptoms by looking at surrounding words
                if i > 0 and i < len(words) - 1:
                    if words[i-1] in ["severe", "mild", "chronic", "acute", "persistent", "recurring", "intense"]:
                        symptoms.append(f"{words[i-1]} {word}")
                    else:
                        symptoms.append(word)
                else:
                    symptoms.append(word)
    
    # Look for common symptom phrases
    symptom_phrases = [
        "trouble breathing", "difficulty breathing", "short of breath", "shortness of breath",
        "chest pain", "back pain", "sore throat", "runny nose", "stuffy nose",
        "family history", "medical history", "hard to breathe", "can't breathe",
        "wheezing", "blocked nose", "blocked airways", "coughing", "sneezing",
        "high fever", "high temperature", "feeling weak", "weakness", "dizziness",
        "feeling tired", "exhaustion", "lack of energy", "loss of appetite"
    ]
    for phrase in symptom_phrases:
        if phrase in text_lower and phrase not in [s.lower() for s in symptoms]:
            symptoms.append(phrase)
    
    # Extract duration and severity
    duration_terms = ["days", "weeks", "months", "years", "chronic", "acute", "persistent", "recurring"]
    severity_terms = ["mild", "moderate", "severe", "intense", "unbearable", "slight", "extreme"]
    
    for term in duration_terms:
        if term in text_lower:
            for i in range(len(words)-1):
                if words[i].isdigit() and words[i+1] == term:
                    symptoms.append(f"{words[i]} {term}")
    
    for term in severity_terms:
        if term in text_lower:
            next_word_index = text_lower.find(term) + len(term)
            if next_word_index < len(text_lower):
                next_words = text_lower[next_word_index:].strip().split()
                if next_words and next_words[0] in keywords:
                    symptoms.append(f"{term} {next_words[0]}")
    
    if symptoms:
        print(f"[DEBUG] Extracted symptoms: {symptoms}")
    else:
        print("[DEBUG] No symptoms extracted.")
    return list(set(symptoms))

# ----------------------------
# Aggregation and Summary Functions
# ----------------------------
def aggregate_predictions(predictions):
    """
    Aggregate predictions by disease. For each disease, average similarity scores,
    merge matched symptoms, and combine adverse events.
    """
    
    aggregated = defaultdict(lambda: {
        "disease": None,
        "matched_symptoms": [],
        "similarity_sum": 0.0,
        "count": 0,
        "adverse_events": []
    })
    
    for pred in predictions:
        disease = pred["disease"]
        aggregated[disease]["disease"] = disease
        aggregated[disease]["matched_symptoms"].append(pred["matched_symptom"])
        aggregated[disease]["similarity_sum"] += pred["similarity"]
        aggregated[disease]["count"] += 1
        if isinstance(pred["adverse_events"], list):
            aggregated[disease]["adverse_events"].extend(pred["adverse_events"])
        elif pred["adverse_events"] is not None:
            aggregated[disease]["adverse_events"].append(pred["adverse_events"])

    result = []
    for disease, data in aggregated.items():
        data["average_similarity"] = round(data["similarity_sum"] / data["count"], 4)
        data["matched_symptoms"] = list(set(data["matched_symptoms"]))
        result.append(data)
    return result

def generate_summary(aggregated_predictions, alerts):
    """
    Generate a descriptive summary paragraph based on aggregated predictions and alerts.
    """
    
    if aggregated_predictions:
        summary = "Based on the analysis of your symptoms, the following conditions may be present:"
        for agg in aggregated_predictions:
            summary += (
                f"\n- There is a possibility of {agg['disease']} "
                f"(matched symptoms: {', '.join(agg['matched_symptoms'])}, "
                f"average similarity: {agg['average_similarity']})."
            )
        if alerts:
            summary += "\n\nHigh risk alerts: " + " ".join(alerts)
        else:
            summary += "\n\nNo immediate high risk alerts were detected."
        summary += "\nIt is highly recommended to consult with a healthcare professional for further evaluation."
    else:
        summary = "No conditions could be predicted from the provided symptoms."
    return summary

# ----------------------------
# Load Dataset & Bioformer‑8L Model
# ----------------------------
print("Loading dataset, Bioformer‑8L model, and tokenizer...")

data_files = {
    "train": "symptom-disease-train-dataset.csv",
    "test": "symptom-disease-test-dataset.csv"
}
dataset = load_dataset("duxprajapati/symptom-disease-dataset", data_files=data_files)
train_df = dataset["train"].to_pandas().head(100)

tokenizer = AutoTokenizer.from_pretrained("bioformers/bioformer-8L")
model = AutoModel.from_pretrained("bioformers/bioformer-8L")

symptom_embeddings = []
for symptom in train_df["text"]:
    symptom_embeddings.append(embed_text(symptom.lower()))
train_df["symptom_embedding"] = symptom_embeddings

print("Initialization complete.")

# ----------------------------
# Pydantic Model for API
# ----------------------------
class TranscriptInput(BaseModel):
    transcript: str

@app.get("/")
def read_root():
    return {"message": "Welcome to the Medical Adverse Event Prediction API!"}

@app.post("/predict")
def predict(input_data: TranscriptInput):
    transcript = input_data.transcript.strip()
    if not transcript:
        raise HTTPException(status_code=400, detail="Transcript cannot be empty.")

    extracted_symptoms = extract_symptoms_ner(transcript) or [transcript]
    
    if len(extracted_symptoms) >= 2:
        extracted_symptoms.append(" and ".join(extracted_symptoms[:3]))

    all_matches = []
    for symptom in extracted_symptoms:
        threshold = min(0.85, 0.65 + (len(symptom.split()) / 100))
        matches = find_similar_symptoms(symptom, train_df, top_k=2, threshold=threshold)
        if matches:
            all_matches.extend(matches)

    extracted_keywords = set()
    for symptom in extracted_symptoms:
        extracted_keywords.update(symptom.lower().split())
    
    filtered_matches = [
        match for match in all_matches
        if any(keyword in set(match[0].lower().split()) for keyword in extracted_keywords)
    ]
    
    if not filtered_matches:
        return {
            "summary": "No matching symptoms found in our database. Please provide more specific medical symptoms.",
            "aggregated_predictions": [],
            "alerts": []
        }

    predictions = []
    for match in filtered_matches:
        matched_symptom, disease, similarity = match
        clean_disease = str(disease)
        
        if any(char.isdigit() for char in clean_disease):
            text_parts = ''.join([c if not c.isdigit() else ' ' for c in clean_disease]).split()
            clean_disease = ' '.join(text_parts).strip() if text_parts else clean_disease
        
        clean_disease = ' '.join(word.capitalize() for word in clean_disease.split())
        adverse_data = query_adverse_events(clean_disease, limit=2)
        
        if isinstance(adverse_data, dict) and "error" in adverse_data:
            adverse_events = []
        elif isinstance(adverse_data, dict) and "results" in adverse_data:
            adverse_events = adverse_data["results"][:2]
        else:
            adverse_events = []

        predictions.append({
            "matched_symptom": matched_symptom,
            "disease": clean_disease,
            "similarity": float(round(similarity, 4)),
            "adverse_events": adverse_events
        })

    aggregated_predictions = aggregate_predictions(predictions)

    # Create alerts
    alerts = []
    for agg in aggregated_predictions:
        if agg["average_similarity"] > 0.85:
            alerts.append(
                f"High risk alert for disease '{agg['disease']}' based on symptoms {agg['matched_symptoms']}."
            )

    return {
        "summary": generate_summary(aggregated_predictions, alerts),
        "aggregated_predictions": aggregated_predictions,
        "alerts": alerts
    }

# ----------------------------
# Command-line Mode: Record & Predict
# ----------------------------
def format_prediction_results(result):
    """Format prediction results in a clear, readable way"""
    formatted_output = f"""
===========================================================
           MEDICAL SYMPTOM ANALYSIS REPORT
===========================================================

SUMMARY:
-----------------------------------------------------------
{result['summary'].split("\n\nHigh risk alerts:")[0]}

HIGH RISK ALERTS:
-----------------------------------------------------------"""
    formatted_output += "\n".join(f"* {alert}" for alert in result['alerts']) if result['alerts'] else "No high risk alerts detected."
    
    formatted_output += """
DETAILED ANALYSIS:
-----------------------------------------------------------"""
    for pred in result['aggregated_predictions']:
        formatted_output += f"""
Condition: {pred['disease']}
Confidence: {pred['average_similarity'] * 100:.2f}%
Matched Symptoms: 
  - {"  - ".join(pred['matched_symptoms'])}"""

    formatted_output += """
===========================================================
NOTE: The 404 errors indicate that the OpenFDA API couldn't 
find adverse event data for these specific conditions.
This is normal for numeric condition codes or rare conditions.
==========================================================="""
    return formatted_output

def record_and_predict():
    record_and_transcribe()
    try:
        with open("output.txt", "r", encoding="utf-8") as f:
            transcript = " ".join(
                line.replace("Transcription:", "").strip()
                for line in f.read().split("\n")
                if line.startswith("Transcription:")
            )
        
        if not transcript:
            return "No speech detected. Please try again."
            
        result = predict(TranscriptInput(transcript=transcript))
        with open("analysis_report.txt", "w", encoding="utf-8") as f:
            f.write(format_prediction_results(result))
        return "Analysis report saved to analysis_report.txt"
        
    except Exception as e:
        return f"Error processing transcript: {e}"

if __name__ == "__main__":
    print("Starting medical symptom analysis...")
    print("Recording audio... Please speak your symptoms clearly.")
    result = record_and_predict()
    print(result)