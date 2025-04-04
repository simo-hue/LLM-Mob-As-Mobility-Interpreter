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

# Helper function
def get_chat_completion(prompt, model="llama3.1", json_mode=False, max_tokens=1200):
    base_url = "http://localhost:11434"
    url = f"{base_url}/api/chat"  # Endpoint corretto per chat con Ollama

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],  # Struttura corretta per API chat
        "stream": False,
        "max_tokens": max_tokens
    }

    try:
        # Verifica se il server Ollama è attivo controllando i modelli installati
        test_response = requests.get(f"{base_url}/api/tags")
        if test_response.status_code != 200:
            print("⚠️ Errore: Il server Ollama non è attivo o non risponde correttamente. Avvialo con 'ollama serve'")
            return None

        # Invio della richiesta
        response = requests.post(url, json=payload)
        response.raise_for_status()

        completion = response.json()

        # Controlliamo se la risposta ha la struttura prevista
        if "message" in completion and "content" in completion["message"]:
            return completion if json_mode else completion["message"]["content"]
        else:
            print("⚠️ Errore: Risposta malformata da Ollama", completion)
            return None

    except requests.exceptions.RequestException as e:
        print(f"❌ Errore chiamata al modello Ollama: {e}")
        return None

def main():
    prompt = "Cosa sai dell'intelligenza artificiale?"
    response = get_chat_completion(prompt)
    print(response)  # Stampa il testo generato dal modello


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n\nInterruzione manuale rilevata. Uscita in modo sicuro...")