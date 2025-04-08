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
import re

def convert_to_serializable(obj):
    if isinstance(obj, (np.int64, np.int32, np.int16, np.integer)):
        return int(obj)
    elif isinstance(obj, (np.float64, np.float32, np.floating)):
        return float(obj)
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(i) for i in obj]
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    return obj

def extract_json_from_response(text):
    """
    Estrae il primo oggetto JSON valido da una stringa.
    """
    json_pattern = r'\{.*?\}'
    matches = re.findall(json_pattern, text, re.DOTALL)
    for match in matches:
        try:
            return json.loads(match)
        except json.JSONDecodeError:
            continue
    return None

def single_query_top1(historical_data, X):
    sanitized_history = convert_to_serializable(historical_data)
    sanitized_context = convert_to_serializable(X['context_stay'])
    sanitized_target = convert_to_serializable(X['target_stay'])

    prompt = f"""
    Your task is to predict a user's next location based on their activity pattern.

    You will be provided with <history> (user's past stays), <context> (recent stays), and <target_stay> (the prediction target). Each stay is in the form:
    (start_time, day_of_week, duration, place_id). Note: duration and place_id in target_stay are None.

    Consider:
    1. Recurring patterns in <history>
    2. Recent activities in <context>
    3. Temporal info (start_time, day_of_week) in <target_stay>

    Output ONLY a one-line JSON object with keys: "prediction" (integer place ID) and "reason" (short explanation).

    <history>: {json.dumps(sanitized_history)}
    <context>: {json.dumps(sanitized_context)}
    <target_stay>: {json.dumps(sanitized_target)}
    """

    response = get_chat_completion(prompt)

    if not response:
        return {"prediction": None, "reason": "No response from model"}

    parsed = extract_json_from_response(response)

    if parsed:
        return parsed
    else:
        print("⚠️ Errore nel parsing della risposta:")
        print("📦 Risposta grezza:", response)
        return {"prediction": None, "reason": "Invalid JSON format from model"}
       
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
    historical_data = [
        ('03:01 PM', 'Wednesday', 385, 4), ('10:04 PM', 'Wednesday', 33, 30), ('10:42 PM', 'Wednesday', 155, 46),
        ('02:57 AM', 'Thursday', 99, 40), ('04:43 AM', 'Thursday', 287, 17), ('10:00 AM', 'Thursday', 32, 184),
        ('11:40 AM', 'Thursday', 957, 40), ('04:07 AM', 'Friday', 32, 184), ('04:45 AM', 'Friday', 410, 195),
        ('11:43 AM', 'Friday', 46, 184), ('01:01 PM', 'Friday', 878, 40), ('03:59 AM', 'Saturday', 106, 184),
        ('05:56 AM', 'Saturday', 351, 195), ('11:55 AM', 'Saturday', 32, 184), ('01:05 PM', 'Saturday', 231, 17),
        ('05:09 PM', 'Saturday', 841, 4), ('07:39 AM', 'Sunday', 555, 17), ('04:58 PM', 'Sunday', 816, 4),
        ('06:41 AM', 'Monday', 48, 17), ('08:10 AM', 'Monday', 98, 17), ('09:57 AM', 'Monday', 34, 4),
        ('10:52 AM', 'Monday', 87, 89), ('12:20 PM', 'Monday', 130, 17), ('02:38 PM', 'Monday', 894, 4),
        ('05:32 AM', 'Tuesday', 88, 1), ('07:01 AM', 'Tuesday', 32, 4), ('07:42 AM', 'Tuesday', 128, 17),
        ('10:22 AM', 'Tuesday', 509, 17), ('07:00 PM', 'Tuesday', 40, 193), ('07:48 PM', 'Tuesday', 549, 4)
    ]

    context_data = [
        ('11:10 AM', 'Monday', 370, 17),
        ('05:30 PM', 'Monday', 851, 4),
        ('07:49 AM', 'Tuesday', 141, 17),
        ('10:39 AM', 'Tuesday', 207, 17),
        ('02:40 PM', 'Tuesday', 872, 4)
    ]

    target_stay = ('06:30 PM', 'Tuesday', None, None)

    X = {
        'context_stay': context_data,
        'target_stay': target_stay
    }

    # 🔧 Usa i dati reali definiti sopra
    response = single_query_top1(historical_data, X)
    print("✅ Risposta:", response)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n\nInterruzione manuale rilevata. Uscita in modo sicuro...")