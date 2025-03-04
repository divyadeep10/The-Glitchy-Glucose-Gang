# record.py
import sys
import os
import re
import sounddevice as sd
import numpy as np
import whisper
from scipy.io.wavfile import write
import re

# Configuration settings
fs = 16000          # Sample rate (Hz)
chunk_duration = 5  # Record audio in chunks (seconds)
output_text_file = "output.txt"

# Load Whisper model (using a small model; adjust as needed)
whisper_model = whisper.load_model("tiny")

def clean_transcript(text):
    """Remove filler words and clean up the transcript."""
    filler_words = [
        r'\bum\b', r'\buh\b', r'\blike\b', r'\byou know\b', r'\bjust\b', 
        r'\bactually\b', r'\bbasically\b', r'\bliterally\b', r'\bkind of\b', r'\bsort of\b',
        r'\bi mean\b', r'\bi guess\b', r'\bi think\b', r'\bwell\b', r'\byeah\b', r'\bokay\b',
        r'\bright\b', r'\banyway\b', r'\banyways\b', r'\bso yeah\b', r'\buhm\b', r'\ber\b'
    ]
    cleaned = text.lower()
    for filler in filler_words:
        cleaned = re.sub(filler, ' ', cleaned)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned

# Modify the record_and_transcribe function to accept pre-loaded models
def record_and_transcribe(preloaded_model=None):
    # Use the preloaded model if provided, otherwise use the global one
    model = preloaded_model if preloaded_model is not None else whisper_model
    
    # Initialize an empty transcript
    final_transcript = ""
    
    try:
        print("Press Ctrl + C to stop manually. Auto-stop enabled for silence.")
        while True:
            print(f"\nRecording for {chunk_duration} seconds...")
            audio = sd.rec(int(chunk_duration * fs), samplerate=fs, channels=1, dtype='int16')
            sd.wait()  # Wait for the recording to finish

            # If volume is too low, assume silence and stop
            if np.max(audio) < 500:
                print("Silence detected. Stopping transcription.")
                break

            # Save audio chunk to temporary file
            temp_audio_file = "temp_chunk.wav"
            write(temp_audio_file, fs, audio)

            print("Transcribing audio...")
            result = model.transcribe(temp_audio_file)
            transcription = result["text"]
            print(f"Transcription: {transcription}")

            final_transcript += " " + transcription
            os.remove(temp_audio_file)
    except KeyboardInterrupt:
        print("\nRecording stopped manually.")
    
    # Clean and return the transcript
    cleaned_transcript = clean_transcript(final_transcript)
    
    # Write to output file
    with open(output_text_file, "w", encoding="utf-8") as f:
        f.write(f"Transcription: {cleaned_transcript}\n")
    
    return cleaned_transcript

# For testing record.py independently, you can include this:
if __name__ == "__main__":
    transcript = record_and_transcribe()
    print("\nFinal Transcript:")
    print(transcript)