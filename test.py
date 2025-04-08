import os
import pickle
import time
import ast
import json
import joblib  # Importa joblib
import logging
import numpy as np
import ast # Aggiunto per la gestione delle eccezioni
from datetime import datetime
import pandas as pd
import requests  # Aggiunto per chiamare il server del modello locale ( ollama serve llama3.1)
import requests

def get_chat_completion(prompt, model="llama3:latest", max_tokens=1200):
    base_url = "http://localhost:11434"
    url = f"{base_url}/api/chat"

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {
            "raw": True  # << IL FIX IMPORTANTE!
        }
    }

    try:
        # Check se il server Ollama è attivo
        test_response = requests.get(f"{base_url}/api/tags")
        if test_response.status_code != 200:
            print("⚠️ Il server Ollama non è attivo o non risponde. Avvialo con 'ollama serve'")
            return None

        response = requests.post(url, json=payload)
        response.raise_for_status()

        data = response.json()

        # Estrai il contenuto
        if "message" in data and "content" in data["message"]:
            return data["message"]["content"]
        else:
            print("⚠️ Struttura della risposta inattesa:", data)
            return None

    except requests.exceptions.HTTPError as http_err:
        print(f"❌ HTTP error: {http_err}")
        try:
            print("💬 Dettaglio errore:", response.json())
        except Exception:
            pass
        return None
    except requests.exceptions.RequestException as e:
        print(f"❌ Errore chiamata al modello Ollama: {e}")
        return None
 
def main():
    test_prompt = "What is the capital of France?"
    response = get_chat_completion(test_prompt, model="llama3:latest")
    print("✅ Risposta:", response)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n\nInterruzione manuale rilevata. Uscita in modo sicuro...")