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

# Mount the "static" folder for static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# ----------------------------
# Utility Functions for Prediction
# ----------------------------
def cosine_similarity(vec1, vec2):
    """Compute cosine similarity between two vectors."""
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
    return cls_embedding.squeeze().numpy()

def find_similar_symptoms(user_symptom, df, top_k=3, threshold=0.75):
    """
    Given a user symptom, compute cosine similarity with each symptom in the dataset.
    Returns top matches meeting the threshold.
    """
    user_emb = embed_text(user_symptom.lower())
    similarities = []
    for _, row in df.iterrows():
        sim_score = cosine_similarity(user_emb, row["symptom_embedding"])
        similarities.append((row["text"], row["label"], sim_score))
    similarities.sort(key=lambda x: x[2], reverse=True)
    filtered = [match for match in similarities if match[2] >= threshold]
    return filtered[:top_k]

def query_adverse_events(disease, limit=2):
    """
    Query the openFDA FAERS API for adverse events related to a given disease.
    """
    url = "https://api.fda.gov/drug/event.json"
    params = {
        "search": f'patient.reaction.reactionmeddrapt:"{disease}"',
        "limit": limit
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error querying adverse events: {response.status_code}")
        return None

# ----------------------------
# Multi-Symptom Extraction using NER
# ----------------------------
# Load a biomedical NER pipeline.
ner_pipeline = pipeline("ner", model="d4data/biomedical-ner-all", aggregation_strategy="simple")

def extract_symptoms_ner(text: str):
    """
    Extract multiple symptoms from text using a biomedical NER model.
    Returns a list of unique symptom strings.
    """
    ner_results = ner_pipeline(text)
    symptoms = []
    for entity in ner_results:
        if entity.get("entity_group", "").upper() in ["SYMPTOM", "DISEASE", "PROBLEM", "FINDING"]:
            symptoms.append(entity["word"])
    
    text_lower = text.lower()
    
    # Add more medical keywords for better extraction
    keywords = ["pain", "ache", "sore", "fever", "cough", "breathing", "breath", 
                "tired", "fatigue", "dizzy", "nausea", "vomit", "headache", "migraine",
                "asthma", "allergy", "swelling", "rash", "itching", "blood", 
                "pressure", "diabetes", "heart", "chest", "stomach", "back", 
                "joint", "muscle", "throat", "nose", "ear", "eye", "skin",
                "shaking", "shivering", "trembling", "sleep", "insomnia", "cold",
                "hot", "sweating", "chills", "numbness", "tingling", "burning"]
    
    # Check for medical keywords in the text
    words = text_lower.split()
    for i, word in enumerate(words):
        for keyword in keywords:
            if keyword in word and word not in [s.lower() for s in symptoms]:
                # Try to capture multi-word symptoms by looking at surrounding words
                if i > 0 and i < len(words) - 1:
                    # Check if previous word is an adjective
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
        if phrase in text.lower() and phrase not in [s.lower() for s in symptoms]:
            symptoms.append(phrase)
    # Extract duration and severity information
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
    
    # Print extracted symptoms for debugging
    if symptoms:
        print(f"Extracted symptoms: {symptoms}")
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
        aggregated[disease]["adverse_events"].extend(pred["adverse_events"])
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
            summary += f"\n- There is a possibility of {agg['disease']} (matched symptoms: {', '.join(agg['matched_symptoms'])}, average similarity: {agg['average_similarity']})."
        if alerts:
            summary += "\n\nHigh risk alerts: " + " ".join(alerts)
        else:
            summary += "\n\nNo immediate high risk alerts were detected."
        summary += "\nIt is highly recommended to consult with a healthcare professional for further evaluation and necessary precautions."
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
train_df = dataset["train"].to_pandas()

# For demo purposes, embed a subset (first 10 rows)
NUM_EXAMPLES = 100
train_df = train_df.head(NUM_EXAMPLES)

tokenizer = AutoTokenizer.from_pretrained("bioformers/bioformer-8L")
model = AutoModel.from_pretrained("bioformers/bioformer-8L")

symptom_embeddings = []
for symptom in train_df["text"]:
    embedding_vector = embed_text(symptom.lower())
    symptom_embeddings.append(embedding_vector)
train_df["symptom_embedding"] = symptom_embeddings

print("Initialization complete.")

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
    if not transcript:
        raise HTTPException(status_code=400, detail="Transcript cannot be empty.")

    # Extract multiple symptoms using enhanced NER
    extracted_symptoms = extract_symptoms_ner(transcript)
    
    # If no symptoms extracted, use the entire transcript as a single symptom
    if not extracted_symptoms:
        print(f"No symptoms extracted from: '{transcript}'. Using full transcript for matching.")
        extracted_symptoms = [transcript]
    else:
        # Also try matching the combination of symptoms for better context
        if len(extracted_symptoms) >= 2:
            combined_symptoms = " and ".join(extracted_symptoms[:3])  # Combine top 3 symptoms
            extracted_symptoms.append(combined_symptoms)
        print(f"Using extracted symptoms for matching: {extracted_symptoms}")
    
    # For each extracted symptom, perform similarity matching
    all_matches = []
    for symptom in extracted_symptoms:
        # Adjust threshold based on symptom length - require higher threshold for longer text
        dynamic_threshold = min(0.85, 0.65 + (len(symptom.split()) / 100))
        matches = find_similar_symptoms(symptom, train_df, top_k=2, threshold=dynamic_threshold)
        if matches:
            all_matches.extend(matches)
    
    # After getting all_matches, add filtering step
    filtered_matches = []
    extracted_keywords = set()
    for symptom in extracted_symptoms:
        extracted_keywords.update(symptom.lower().split())
    
    for match in all_matches:
        matched_symptom, disease, similarity = match
        # Check if there's at least some keyword overlap
        matched_keywords = set(matched_symptom.lower().split())
        if any(keyword in matched_keywords for keyword in extracted_keywords):
            filtered_matches.append(match)
        else:
            print(f"Filtered out unrelated match: {matched_symptom} (similarity: {similarity})")
    
    all_matches = filtered_matches if filtered_matches else all_matches  # Use filtered if not empty
    
    if not all_matches:
        return {
            "summary": "No matching symptoms found in our database. Please provide more specific medical symptoms.",
            "aggregated_predictions": [],
            "alerts": []
        }
    
    predictions = []
    for match in all_matches:
        matched_symptom, disease, similarity = match
        # Clean up disease name - remove numbers and convert to proper title case
        clean_disease = disease
        # Remove numeric codes that might appear in disease names
        if any(char.isdigit() for char in clean_disease):
            # Try to extract just the text part of the disease
            text_parts = ''.join([c if not c.isdigit() else ' ' for c in clean_disease]).split()
            if text_parts:
                clean_disease = ' '.join(text_parts).strip()
        
        # Convert to title case for better readability
        clean_disease = ' '.join(word.capitalize() for word in clean_disease.split())
        
        similarity_value = float(round(similarity, 4))
        adverse_data = query_adverse_events(clean_disease, limit=2)
        adverse_events = adverse_data.get("results", []) if adverse_data else []
        predictions.append({
            "matched_symptom": matched_symptom,
            "disease": clean_disease,  # Use the cleaned disease name
            "similarity": similarity_value,
            "adverse_events": adverse_events
        })
    
    # Aggregate predictions by disease.
    aggregated_predictions = aggregate_predictions(predictions)
    
    # Create alerts based on aggregated average similarity.
    alerts = []
    for agg in aggregated_predictions:
        if agg["average_similarity"] > 0.85:
            alerts.append(
                f"High risk alert for disease '{agg['disease']}' based on symptoms {agg['matched_symptoms']}."
            )
    
    # Generate a descriptive summary paragraph.
    summary = generate_summary(aggregated_predictions, alerts)
    
    return {"summary": summary, "aggregated_predictions": aggregated_predictions, "alerts": alerts}

# ----------------------------
# Command-line Mode: Record & Predict
# ----------------------------
def format_prediction_results(result):
    """Format prediction results in a clear, readable way"""
    
    formatted_output = """
===========================================================
           MEDICAL SYMPTOM ANALYSIS REPORT
===========================================================

SUMMARY:
-----------------------------------------------------------
{}

HIGH RISK ALERTS:
-----------------------------------------------------------
""".format(result['summary'].split("\n\nHigh risk alerts:")[0])

    # Add alerts section
    if result['alerts']:
        for alert in result['alerts']:
            formatted_output += "* {}\n".format(alert)
    else:
        formatted_output += "No high risk alerts detected.\n"
    
    # Add detailed predictions section
    formatted_output += """
DETAILED ANALYSIS:
-----------------------------------------------------------
"""
    
    for pred in result['aggregated_predictions']:
        formatted_output += """
Condition: {}
Confidence: {:.2f}%
Matched Symptoms: 
  - {}

""".format(
            pred['disease'],
            pred['average_similarity'] * 100,
            "\n  - ".join(pred['matched_symptoms'])
        )
    
    formatted_output += """
===========================================================
NOTE: The 404 errors indicate that the OpenFDA API couldn't 
find adverse event data for these specific conditions.
This is normal for numeric condition codes or rare conditions.
===========================================================
"""
    
    return formatted_output
# After the format_prediction_results function, add the record_and_predict function:

def record_and_predict():
    print("Starting audio recording and transcription...")
    # The record_and_transcribe function doesn't return the transcript
    # It writes to a file instead, so we need to read that file
    record_and_transcribe()
    try:
        # Read the transcript from the output file
        with open("output.txt", "r", encoding="utf-8") as f:
            content = f.read()
        
        # Parse the content to extract the transcript
        transcript = ""
        for line in content.split("\n"):
            if line.startswith("Transcription:"):
                transcript += line.replace("Transcription:", "").strip() + " "
        
        transcript = transcript.strip()
        print("\nFinal Transcript:")
        print(transcript)
        
        # Check if transcript is empty
        if not transcript:
            print("No speech detected. Please try again and speak clearly into the microphone.")
            return
        
        # Create a TranscriptInput and call the predict function directly
        input_data = TranscriptInput(transcript=transcript)
        result = predict(input_data)
        
        # Format the results in a more readable way
        formatted_result = format_prediction_results(result)
        print("\nPrediction Result:")
        print(formatted_result)
        
        # Save formatted result to a file
        with open("analysis_report.txt", "w", encoding="utf-8") as f:
            f.write(formatted_result)
        print("Analysis report saved to analysis_report.txt")
        
    except FileNotFoundError:
        print("No output file found. Recording may have failed.")
    except Exception as e:
        print(f"Error processing transcript: {e}")

# ----------------------------
# Main Block
# ----------------------------
if __name__ == "__main__":
    # Uncomment the desired mode:
    
    # Option 1: Run as FastAPI web server
    # uvicorn.run(app, host="0.0.0.0", port=8000)
    
    # Option 2: Run in command-line mode (record audio, transcribe, and predict)
    record_and_predict()
