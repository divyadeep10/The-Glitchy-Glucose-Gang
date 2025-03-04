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
    print(f"[DEBUG] Created static directory at {static_dir}")

# Now mount the static directory
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
    cls_embedding = outputs.last_hidden_state[:, 0, :]
    embedding_np = cls_embedding.squeeze().numpy()
    # Debug
    print("[DEBUG] embed_text:", text[:30], "... => embedding shape:", embedding_np.shape)
    return embedding_np

def find_similar_symptoms(user_symptom, df, top_k=3, threshold=0.75):
    """
    Given a user symptom, compute cosine similarity with each symptom in the dataset.
    Returns top matches meeting the threshold.
    """
    print(f"[DEBUG] find_similar_symptoms called with user_symptom='{user_symptom}' threshold={threshold}")
    user_emb = embed_text(user_symptom.lower())
    similarities = []
    for idx, row in df.iterrows():
        # Debug: check the type of row["symptom_embedding"]
        if not isinstance(row["symptom_embedding"], np.ndarray):
            print(f"[DEBUG] WARNING: row['symptom_embedding'] is not an ndarray at index={idx}. It is:", type(row["symptom_embedding"]))
        sim_score = cosine_similarity(user_emb, row["symptom_embedding"])
        similarities.append((row["text"], row["label"], sim_score))
    similarities.sort(key=lambda x: x[2], reverse=True)
    filtered = [match for match in similarities if match[2] >= threshold]
    return filtered[:top_k]

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
        print(f"[DEBUG] query_adverse_events: Attempt {attempt+1}/{retries} for disease='{disease}'")
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
            print(f"[DEBUG] Server issue {response.status_code}. Retrying ({attempt+1}/{retries})...")
            time.sleep(2 ** attempt)  # Exponential backoff
            attempt += 1
        else:
            print(f"[DEBUG] Unexpected error: {response.status_code}, {response.text}")
            return {"error": f"Unexpected API error: {response.status_code}"}

    print("[DEBUG] query_adverse_events: Gave up after multiple retries.")
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
    print("[DEBUG] aggregate_predictions called with", len(predictions), "prediction items.")
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

        # Debug around adverse_events
        print(f"[DEBUG] Merging adverse_events for disease='{disease}' => Type of pred['adverse_events']: {type(pred['adverse_events'])}")
        try:
            aggregated[disease]["adverse_events"].extend(pred["adverse_events"])
        except TypeError as e:
            print("[DEBUG] ERROR in .extend() - pred['adverse_events'] is not iterable:", pred["adverse_events"])
            raise

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
    print("[DEBUG] generate_summary called.")
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
print("[DEBUG] Loading dataset, Bioformer‑8L model, and tokenizer...")

data_files = {
    "train": "symptom-disease-train-dataset.csv",
    "test": "symptom-disease-test-dataset.csv"
}
dataset = load_dataset("duxprajapati/symptom-disease-dataset", data_files=data_files)
train_df = dataset["train"].to_pandas()

# For demo, embed only a subset
NUM_EXAMPLES = 100
train_df = train_df.head(NUM_EXAMPLES)

tokenizer = AutoTokenizer.from_pretrained("bioformers/bioformer-8L")
model = AutoModel.from_pretrained("bioformers/bioformer-8L")

symptom_embeddings = []
for i, symptom in enumerate(train_df["text"]):
    emb_vec = embed_text(symptom.lower())
    # Debug
    print(f"[DEBUG] Row {i} => symptom='{symptom}' => emb_vec shape={emb_vec.shape}")
    symptom_embeddings.append(emb_vec)
train_df["symptom_embedding"] = symptom_embeddings

print("[DEBUG] Initialization complete.")

# ----------------------------
# Pydantic Model for API
# ----------------------------
class TranscriptInput(BaseModel):
    transcript: str  # The transcribed text from the audio

# ----------------------------
# FastAPI Routes
# ----------------------------
@app.get("/")
def read_root():
    return {"message": "Welcome to the Medical Adverse Event Prediction API!"}

@app.post("/predict")
def predict(input_data: TranscriptInput):
    transcript = input_data.transcript.strip()
    print("[DEBUG] /predict called with transcript:", transcript)
    if not transcript:
        raise HTTPException(status_code=400, detail="Transcript cannot be empty.")

    # Extract multiple symptoms using enhanced NER
    extracted_symptoms = extract_symptoms_ner(transcript)
    
    if not extracted_symptoms:
        print("[DEBUG] No symptoms extracted. Using full transcript for matching.")
        extracted_symptoms = [transcript]
    else:
        if len(extracted_symptoms) >= 2:
            combined_symptoms = " and ".join(extracted_symptoms[:3])
            extracted_symptoms.append(combined_symptoms)
        print("[DEBUG] Using extracted symptoms for matching:", extracted_symptoms)
    
    all_matches = []
    for symptom in extracted_symptoms:
        dynamic_threshold = min(0.85, 0.65 + (len(symptom.split()) / 100))
        print(f"[DEBUG] Checking symptom='{symptom}' with threshold={dynamic_threshold}")
        matches = find_similar_symptoms(symptom, train_df, top_k=2, threshold=dynamic_threshold)
        if matches:
            all_matches.extend(matches)
    
    # Filter matches by keyword overlap
    filtered_matches = []
    extracted_keywords = set()
    for symptom in extracted_symptoms:
        extracted_keywords.update(symptom.lower().split())
    
    for match in all_matches:
        matched_symptom, disease, similarity = match
        matched_keywords = set(matched_symptom.lower().split())
        if any(keyword in matched_keywords for keyword in extracted_keywords):
            filtered_matches.append(match)
        else:
            print(f"[DEBUG] Filtered out unrelated match: {match}")
    
    all_matches = filtered_matches if filtered_matches else all_matches
    
    if not all_matches:
        return {
            "summary": "No matching symptoms found in our database. Please provide more specific medical symptoms.",
            "aggregated_predictions": [],
            "alerts": []
        }

    predictions = []
    for match in all_matches:
        matched_symptom, disease, similarity = match
        clean_disease = disease

        # Remove digits from disease name
        if any(char.isdigit() for char in clean_disease):
            text_parts = ''.join([c if not c.isdigit() else ' ' for c in clean_disease]).split()
            if text_parts:
                clean_disease = ' '.join(text_parts).strip()

        clean_disease = ' '.join(word.capitalize() for word in clean_disease.split())
        similarity_value = float(round(similarity, 4))

        # Query openFDA
        adverse_data = query_adverse_events(clean_disease, limit=2)
        print("[DEBUG] Type of adverse_data:", type(adverse_data))
        print("[DEBUG] Content of adverse_data:", adverse_data)

        # Decide how to store adverse_events
        if isinstance(adverse_data, dict):
            if "results" in adverse_data and isinstance(adverse_data["results"], list):
                adverse_events = adverse_data["results"]
            elif "error" in adverse_data:
                adverse_events = []
                print("[DEBUG] openFDA error message:", adverse_data["error"])
            else:
                adverse_events = []
                print("[DEBUG] Warning: Unexpected format in adverse_data dict.")
        else:
            adverse_events = []
            print("[DEBUG] Warning: adverse_data is not a dict.")

        print("[DEBUG] Final adverse_events type:", type(adverse_events))
        predictions.append({
            "matched_symptom": matched_symptom,
            "disease": clean_disease,
            "similarity": similarity_value,
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

    summary = generate_summary(aggregated_predictions, alerts)
    return {
        "summary": summary,
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
{result['summary'].split("\\n\\nHigh risk alerts:")[0]}

HIGH RISK ALERTS:
-----------------------------------------------------------
"""
    if result['alerts']:
        for alert in result['alerts']:
            formatted_output += f"* {alert}\n"
    else:
        formatted_output += "No high risk alerts detected.\n"
    
    formatted_output += """
DETAILED ANALYSIS:
-----------------------------------------------------------
"""
    
    for pred in result['aggregated_predictions']:
        formatted_output += f"""
Condition: {pred['disease']}
Confidence: {pred['average_similarity'] * 100:.2f}%
Matched Symptoms: 
  - {chr(10).join(pred['matched_symptoms'])}

"""
    
    formatted_output += """
===========================================================
NOTE: The 404 errors indicate that the OpenFDA API couldn't 
find adverse event data for these specific conditions.
This is normal for numeric condition codes or rare conditions.
===========================================================
"""
    return formatted_output

def record_and_predict():
    print("[DEBUG] Starting audio recording and transcription...")
    record_and_transcribe()
    try:
        with open("output.txt", "r", encoding="utf-8") as f:
            content = f.read()
        # Debug: check the file content
        print("[DEBUG] Read from output.txt => Type:", type(content))
        print("[DEBUG] Content:", content)

        transcript = ""
        for line in content.split("\n"):
            if line.startswith("Transcription:"):
                transcript_line = line.replace("Transcription:", "").strip()
                transcript += transcript_line + " "
        
        transcript = transcript.strip()
        print("[DEBUG] Final Transcript after parsing:", transcript)
        
        if not transcript:
            print("[DEBUG] No speech detected. Please try again.")
            return
        
        input_data = TranscriptInput(transcript=transcript)
        result = predict(input_data)
        
        formatted_result = format_prediction_results(result)
        print("\n[DEBUG] Prediction Result:")
        print(formatted_result)
        # Save formatted result to a file
        with open("analysis_report.txt", "w", encoding="utf-8") as f:
            f.write(formatted_result)
        print("[DEBUG] Analysis report saved to analysis_report.txt")
        
    except FileNotFoundError:
        print("[DEBUG] No output file found. Recording may have failed.")
    except Exception as e:
        print(f"[DEBUG] Error processing transcript: {e}")