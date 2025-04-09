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
import re
import requests  # Aggiunto per chiamare il server del modello locale ( ollama serve )

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

def extract_json_from_text(text):
    import re
    import json

    try:
        # Rimuove blocchi markdown se presenti
        clean_text = re.sub(r"```(json)?", "", text).replace("```", "").strip()

        # Prova a trovare un blocco JSON
        match = re.search(r'\{.*?\}', clean_text, re.DOTALL)
        if match:
            return json.loads(match.group())

        # Prova a trovare un JSON compatibile semplice (solo prediction e reason)
        match_simple = re.search(r'"prediction"\s*:\s*\[.*?\].*?"reason"\s*:\s*".*?"', clean_text, re.DOTALL)
        if match_simple:
            json_like = '{' + match_simple.group(0).rstrip(',') + '}'
            return json.loads(json_like)

        print("❌ Nessun blocco JSON trovato nel testo.")
        return None

    except Exception as e:
        print("⚠️ Errore di parsing JSON:", e)
        return None

def get_chat_completion(prompt, model="llama3:latest", json_mode=True, max_tokens=1200):
    if not isinstance(prompt, str):
        prompt = str(prompt)

    base_url = "http://localhost:11434"
    url = f"{base_url}/api/chat"

    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "stream": False,
        "max_tokens": max_tokens
    }

    try:
        test_response = requests.get(f"{base_url}/api/tags")
        if test_response.status_code != 200:
            print("⚠️ Il server Ollama non è attivo. Avvialo con 'ollama serve'")
            return None

        response = requests.post(url, json=payload)
        response.raise_for_status()
        data = response.json()

        content = data.get("message", {}).get("content", "")
        
        # DEBUG:
        #print("📦 Risposta JSON (grezza o meno):\n", content)

        if json_mode:
            try:
                parsed = json.loads(content)
                # DEBUG:
                #print("✅ JSON interpretato correttamente:\n", json.dumps(parsed, indent=2))
                return parsed
            except json.JSONDecodeError:
                print("⚠️ Risposta non in formato JSON puro. Provo a estrarre...")
                extracted = extract_json_from_text(content)
                if extracted:
                    # DEBUG:
                    #print("✅ JSON estratto con successo:\n", json.dumps(extracted, indent=2))
                    return json.loads(json.dumps(extracted))
                else:
                    print("❌ Estrazione JSON fallita.")
                return None
        return content

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

def get_dataset(dataname):
    
    # Get training and validation set and merge them
    train_data = pd.read_csv(f"data/{dataname}/{dataname}_train.csv")
    valid_data = pd.read_csv(f"data/{dataname}/{dataname}_valid.csv")

    # Get test data using joblib
    test_file = joblib.load(f"data/{dataname}/{dataname}_testset.pk")  # Usa joblib per caricare il file

    # merge train and valid data
    tv_data = pd.concat([train_data, valid_data], ignore_index=True)
    tv_data.sort_values(['user_id', 'start_day', 'start_min'], inplace=True)
    
    if dataname == 'geolife':
        # Forza la colonna 'duration' a essere di tipo int in modo sicuro
        tv_data['duration'] = pd.to_numeric(tv_data['duration'], errors='coerce', downcast='integer')

    #DEBUG:
    #print("Number of total test sample: ", len(test_file))
    return tv_data, test_file

def convert_to_12_hour_clock(minutes):
    if minutes < 0 or minutes >= 1440:
        return "Invalid input. Minutes should be between 0 and 1439."

    hours = minutes // 60
    minutes %= 60

    period = "AM"
    if hours >= 12:
        period = "PM"

    if hours == 0:
        hours = 12
    elif hours > 12:
        hours -= 12

    return f"{hours:02d}:{minutes:02d} {period}"

def int2dow(int_day):
    tmp = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday',
           3: 'Thursday', 4: 'Friday', 5: 'Saturday', 6: 'Sunday'}
    return tmp[int_day]

def get_logger(logger_name, log_dir='logs/'):
    # Create log dir
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Create a logger instance
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)

    # Create a console handler and set its log level
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)

    # Create a file handler and set its log level
    current_datetime = datetime.now()
    formatted_datetime = current_datetime.strftime("%Y%m%d_%H%M%S")
    log_file = 'log_file' + formatted_datetime + '.log'
    log_file_path = os.path.join(log_dir, log_file)
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging.DEBUG)

    # Create a formatter and add it to the handlers
    formatter = logging.Formatter('%(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    # Add the handlers to the logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger

def get_user_data(train_data, uid, num_historical_stay, logger):
    user_train = train_data[train_data['user_id']==uid]
    logger.info(f"Length of user {uid} train data: {len(user_train)}")
    user_train = user_train.tail(num_historical_stay)
    logger.info(f"Number of user historical stays: {len(user_train)}")
    return user_train

# Organising data
def organise_data(dataname, user_train, test_file, uid, logger, num_context_stay=5):
    # Use another way of organising data
    historical_data = []

    if dataname == 'geolife':
        for _, row in user_train.iterrows():
            historical_data.append(
                (convert_to_12_hour_clock(int(row['start_min'])),
                int2dow(row['weekday']),
                int(row['duration']),
                row['location_id'])
                )
    elif dataname == 'fsq':
        for _, row in user_train.iterrows():
            historical_data.append(
                (convert_to_12_hour_clock(int(row['start_min'])),
                int2dow(row['weekday']),
                row['location_id'])
                )

    logger.info(f"historical_data: {historical_data}")
    logger.info(f"Number of historical_data: {len(historical_data)}")

    # Get user ith test data
    list_user_dict = []
    for i_dict in test_file:
        if dataname == 'geolife':
            i_uid = i_dict['user_X'][0]
        elif dataname == 'fsq':
            i_uid = i_dict['user_X']
        if i_uid == uid:
            list_user_dict.append(i_dict)

    predict_X = []
    predict_y = []
    for i_dict in list_user_dict:
        construct_dict = {}
        if dataname == 'geolife':
            context = list(zip([convert_to_12_hour_clock(int(item)) for item in i_dict['start_min_X'][-num_context_stay:]], 
                            [int2dow(i) for i in i_dict['weekday_X'][-num_context_stay:]], 
                            [int(i) for i in i_dict['dur_X'][-num_context_stay:]], 
                            i_dict['X'][-num_context_stay:]))
        elif dataname == 'fsq':
            context = list(zip([convert_to_12_hour_clock(int(item)) for item in i_dict['start_min_X'][-num_context_stay:]], 
                            [int2dow(i) for i in i_dict['weekday_X'][-num_context_stay:]], 
                            i_dict['X'][-num_context_stay:]))
        target = (convert_to_12_hour_clock(int(i_dict['start_min_Y'])), int2dow(i_dict['weekday_Y']), None, "<next_place_id>")
        construct_dict['context_stay'] = context
        construct_dict['target_stay'] = target
        predict_y.append(i_dict['Y'])
        predict_X.append(construct_dict)

    #logger.info(f"predict_data: {predict_X}")
    logger.info(f"Number of predict_data: {len(predict_X)}")
    logger.info(f"predict_y: {predict_y}")
    logger.info(f"Number of predict_y: {len(predict_y)}")
    return historical_data, predict_X, predict_y

# Make a single query 
def convert_stays(stays):
    def convert_value(x):
        if isinstance(x, (np.integer,)):
            return int(x)
        return x
    return [(t, d, dur, convert_value(pid)) for (t, d, dur, pid) in stays]

def single_query_top1(historical_data, X):
    # 🔁 Pulisce tutti i valori numpy o non serializzabili
    sanitized_history = convert_to_serializable(historical_data)
    sanitized_context = convert_to_serializable(X['context_stay'])
    sanitized_target = convert_to_serializable(X['target_stay'])

    # ✍️ Prompt strutturato per la predizione del prossimo luogo
    prompt = f"""
    Your task is to predict a user's next location based on their activity pattern.

    You will be provided with <history> (user's past stays), <context> (recent stays), and <target_stay> (the prediction target). Each stay is in the form:
    (start_time, day_of_week, duration, place_id). Note: duration and place_id in target_stay are None.

    Consider:
    1. Recurring patterns in <history>
    2. Recent activities in <context>
    3. Temporal info (start_time, day_of_week) in <target_stay>

    Respond ONLY with a single-line JSON object with the following keys:
    - "prediction": a list of integers (place IDs)
    - "reason": a short string explaining your prediction

    Do NOT include any code, explanation, or markdown formatting. Only output valid JSON.

    <history>: {json.dumps(sanitized_history)}
    <context>: {json.dumps(sanitized_context)}
    <target_stay>: {json.dumps(sanitized_target)}
    """

    # 🧠 Chiamata al modello LLM
    response = get_chat_completion(prompt)

    # DEBUG:
    #print("📦 Risposta JSON (grezza o meno):", response)

    # 🧩 Parsing robusto
    if response is None or not isinstance(response, dict):
        print("⚠️ Risposta non valida o assente.")
        return {"prediction": None, "reason": "Invalid JSON format from model"}

    print("✅ JSON interpretato correttamente:\n", json.dumps(response, indent=2))
    return response

# Make a single query of 10 most likely places
def single_query_top10(historical_data, X):
    """
    Make a single query to get top-10 predicted next places.
    param: 
    historical_data: list of past stays
    X: dict with keys 'context_stay' and 'target_stay'
    """
    sanitized_history = convert_to_serializable(historical_data)
    sanitized_context = convert_to_serializable(X['context_stay'])
    sanitized_target = convert_to_serializable(X['target_stay'])

    prompt = f"""
    Your task is to predict a user's next location based on his/her activity pattern.
    You will be provided with <history> which is a list containing this user's historical stays, then <context> which provide contextual information 
    about where and when this user has been to recently. Stays in both <history> and <context> are in chronological order.
    Each stay takes on such form as (start_time, day_of_week, duration, place_id). The detailed explanation of each element is as follows:
    start_time: the start time of the stay in 12h clock format.
    day_of_week: indicating the day of the week.
    duration: an integer indicating the duration (in minute) of each stay. Note that this will be None in the <target_stay> introduced later.
    place_id: an integer representing the unique place ID, which indicates where the stay is.

    Then you need to do next location prediction on <target_stay> which is the prediction target with unknown place ID denoted as <next_place_id> and 
    unknown duration denoted as None, while temporal information is provided.      
    
    Please infer what the <next_place_id> might be (please output the 10 most likely places which are ranked in descending order in terms of probability), considering the following aspects:
    1. the activity pattern of this user that you learned from <history>, e.g., repeated visits to certain places during certain times;
    2. the context stays in <context>, which provide more recent activities of this user; 
    3. the temporal information (i.e., start_time and day_of_week) of target stay, which is important because people's activity varies during different time (e.g., nighttime versus daytime)
    and on different days (e.g., weekday versus weekend).

    Respond ONLY with a single-line JSON object with the following keys:
    - "prediction": a list of integers (place IDs)
    - "reason": a short string explaining your prediction

    Do NOT include any code, explanation, or markdown formatting. Only output valid JSON.


    <history>: {json.dumps(sanitized_history)}
    <context>: {json.dumps(sanitized_context)}
    <target_stay>: {json.dumps(sanitized_target)}
    """

    response = get_chat_completion(prompt)

    # DEBUG:
    #print("📦 Risposta JSON (grezza o meno):", response)

    # La funzione get_chat_completion restituisce già un dizionario o None
    if response is None or not isinstance(response, dict):
        print("⚠️ Risposta non valida o assente.")
        return {"prediction": None, "reason": "Invalid JSON format from model"}

    print("✅ JSON interpretato correttamente:\n", json.dumps(response, indent=2))
    return response

# Make a single query of 10 most likely places without time information
def single_query_top1_wot(historical_data, X):
    """
    Make a single query without using target temporal info (WOT = Without Target time).
    param: 
    historical_data: list of past stays
    X: dict with key 'context_stay' only
    """
    sanitized_history = convert_to_serializable(historical_data)
    sanitized_context = convert_to_serializable(X['context_stay'])

    prompt = f"""
    Your task is to predict a user's next location based on his/her activity pattern.
    You will be provided with <history> which is a list containing this user's historical stays, then <context> which provides contextual information 
    about where and when this user has been to recently. Stays in both <history> and <context> are in chronological order.
    Each stay takes the following form: (start_time, day_of_week, duration, place_id). The meaning of each element is:
    - start_time: the start time of the stay in 12h clock format.
    - day_of_week: the day of the week.
    - duration: duration (in minutes) of the stay.
    - place_id: an integer representing a unique place ID.

    Please infer what the <next_place_id> is (i.e., the most likely place ID), considering:
    1. the user's activity patterns from <history> (e.g., repeated visits to certain places at certain times),
    2. and the recent activities from <context>.

    Respond ONLY with a single-line JSON object with the following keys:
    - "prediction": a list of integers (place IDs)
    - "reason": a short string explaining your prediction

    Do NOT include any code, explanation, or markdown formatting. Only output valid JSON.

    <history>: {json.dumps(sanitized_history)}
    <context>: {json.dumps(sanitized_context)}
    """

    response = get_chat_completion(prompt)

    # DEBUG:
    #print("📦 Risposta JSON (grezza o meno):", response)

    # La funzione get_chat_completion restituisce già un dizionario o None
    if response is None or not isinstance(response, dict):
        print("⚠️ Risposta non valida o assente.")
        return {"prediction": None, "reason": "Invalid JSON format from model"}

    print("✅ JSON interpretato correttamente:\n", json.dumps(response, indent=2))
    return response
# 
def single_query_top10_wot(historical_data, X):
    """
    Make a single query of 10 most likely places, without time information
    param: 
    X: one single sample containing context_stay and target_stay
    """
    
    sanitized_history = convert_to_serializable(historical_data)
    sanitized_context = convert_to_serializable(X['context_stay'])
    
    prompt = f"""
    Your task is to predict a user's next location based on his/her activity pattern.
    You will be provided with <history> which is a list containing this user's historical stays, then <context> which provide contextual information 
    about where and when this user has been to recently. Stays in both <history> and <context> are in chronological order.
    Each stay takes on such form as (start_time, day_of_week, duration, place_id). The detailed explanation of each element is as follows:
    start_time: the start time of the stay in 12h clock format.
    day_of_week: indicating the day of the week.
    duration: an integer indicating the duration (in minute) of each stay. 
    place_id: an integer representing the unique place ID, which indicates where the stay is.

    Please infer what the <next_place_id> might be (please output the 10 most likely places which are ranked in descending order in terms of probability), considering the following aspects:
    1. the activity pattern of this user that you leared from <history>, e.g., repeated visits to certain places during certain times;
    2. the context stays in <context>, which provide more recent activities of this user.
  
    Respond ONLY with a single-line JSON object with the following keys:
    - "prediction": a list of integers (place IDs)
    - "reason": a short string explaining your prediction

    Do NOT include any code, explanation, or markdown formatting. Only output valid JSON.    

    <history>: {json.dumps(sanitized_history)}
    <context>: {json.dumps(sanitized_context)}
    """

    response = get_chat_completion(prompt)

    # DEBUG:
    #print("📦 Risposta JSON (grezza o meno):", response)

    # La funzione get_chat_completion restituisce già un dizionario o None
    if response is None or not isinstance(response, dict):
        print("⚠️ Risposta non valida o assente.")
        return {"prediction": None, "reason": "Invalid JSON format from model"}

    print("✅ JSON interpretato correttamente:\n", json.dumps(response, indent=2))
    return response

def single_query_top1_fsq(historical_data, X):
    """
    Make a single query.
    param: 
    X: one single sample containing context_stay and target_stay
    """
    
    sanitized_history = convert_to_serializable(historical_data)
    sanitized_context = convert_to_serializable(X['context_stay'])
    sanitized_target = convert_to_serializable(X['target_stay'])
    
    prompt = f"""
    Your task is to predict a user's next location based on his/her activity pattern.
    You will be provided with <history> which is a list containing this user's historical stays, then <context> which provide contextual information 
    about where and when this user has been to recently. Stays in both <history> and <context> are in chronological order.
    Each stay takes on such form as (start_time, day_of_week, place_id). The detailed explanation of each element is as follows:
    start_time: the start time of the stay in 12h clock format.
    day_of_week: indicating the day of the week.
    place_id: an integer representing the unique place ID, which indicates where the stay is.

    Then you need to do next location prediction on <target_stay> which is the prediction target with unknown place ID denoted as <next_place_id> and 
    unknown duration denoted as None, while temporal information is provided.      
    
    Please infer what the <next_place_id> is (i.e., the most likely place ID), considering the following aspects:
    1. the activity pattern of this user that you leared from <history>, e.g., repeated visit to a certain place during certain time.
    2. the context stays in <context>, which provide more recent activities of this user; 
    3. the temporal information (i.e., start_time and weekday) of target stay, which is important because people's activity varies during different time (e.g., nighttime versus daytime)
    and on different days (e.g., weekday versus weekend).

    Respond ONLY with a single-line JSON object with the following keys:
    - "prediction": a list of integers (place IDs)
    - "reason": a short string explaining your prediction

    Do NOT include any code, explanation, or markdown formatting. Only output valid JSON.

    The data are as follows:
    <history>: {json.dumps(sanitized_history)}
    <context>: {json.dumps(sanitized_context)}
    <target_stay>: {json.dumps(sanitized_target)}
    """
    completion = get_chat_completion(prompt, json_mode=True)

    if completion is None:
        return None

    if isinstance(completion, str):
        try:
            completion = json.loads(completion)
        except json.JSONDecodeError:
            print("⚠️ JSON malformato:", completion)
            return None

    return completion

# Make a single query of 10 most likely places 
def single_query_top1_wot_fsq(historical_data, X):
    """
    Make a single query.
    param: 
    X: one single sample containing context_stay and target_stay
    """
    
    sanitized_history = convert_to_serializable(historical_data)
    sanitized_context = convert_to_serializable(X['context_stay'])
    
    prompt = f"""
    Your task is to predict a user's next location based on his/her activity pattern.
    You will be provided with <history> which is a list containing this user's historical stays, then <context> which provide contextual information 
    about where and when this user has been to recently. Stays in both <history> and <context> are in chronological order.
    Each stay takes on such form as (start_time, day_of_week, place_id). The detailed explanation of each element is as follows:
    start_time: the start time of the stay in 12h clock format.
    day_of_week: indicating the day of the week.
    place_id: an integer representing the unique place ID, which indicates where the stay is.    
    
    Please infer what the <next_place_id> is (i.e., the most likely place ID), considering the following aspects:
    1. the activity pattern of this user that you leared from <history>, e.g., repeated visit to a certain place during certain time;
    2. the context stays in <context>, which provide more recent activities of this user.

    Respond ONLY with a single-line JSON object with the following keys:
    - "prediction": a list of integers (place IDs)
    - "reason": a short string explaining your prediction

    Do NOT include any code, explanation, or markdown formatting. Only output valid JSON.

    The data are as follows:
    <history>: {json.dumps(sanitized_history)}
    <context>: {json.dumps(sanitized_context)}
    """
    completion = get_chat_completion(prompt, json_mode=True)

    if completion is None:
        return None

    if isinstance(completion, str):
        try:
            completion = json.loads(completion)
        except json.JSONDecodeError:
            print("⚠️ JSON malformato:", completion)
            return None

    return completion
# 
def single_query_top10_fsq(historical_data, X):
    """
    Make a single query.
    param: 
    X: one single sample containing context_stay and target_stay
    """
    
    sanitized_history = convert_to_serializable(historical_data)
    sanitized_context = convert_to_serializable(X['context_stay'])
    sanitized_target = convert_to_serializable(X['target_stay'])
    
    prompt = f"""
    Your task is to predict a user's next location based on his/her activity pattern.
    You will be provided with <history> which is a list containing this user's historical stays, then <context> which provide contextual information 
    about where and when this user has been to recently. Stays in both <history> and <context> are in chronological order.
    Each stay takes on such form as (start_time, day_of_week, duration, place_id). The detailed explanation of each element is as follows:
    start_time: the start time of the stay in 12h clock format.
    day_of_week: indicating the day of the week.
    duration: an integer indicating the duration (in minute) of each stay. Note that this will be None in the <target_stay> introduced later.
    place_id: an integer representing the unique place ID, which indicates where the stay is.

    Then you need to do next location prediction on <target_stay> which is the prediction target with unknown place ID denoted as <next_place_id> and 
    unknown duration denoted as None, while temporal information is provided.      
    
    Please infer what the <next_place_id> might be (please output the 10 most likely places which are ranked in descending order in terms of probability), considering the following aspects:
    1. the activity pattern of this user that you leared from <history>, e.g., repeated visits to certain places during certain times.
    2. the context stays in <context>, which provide more recent activities of this user; 
    3. the temporal information (i.e., start_time and weekday) of target stay, which is important because people's activity varies during different time (e.g., nighttime versus daytime)
    and on different days (e.g., weekday versus weekend).

    Respond ONLY with a single-line JSON object with the following keys:
    - "prediction": a list of integers (place IDs)
    - "reason": a short string explaining your prediction

    Do NOT include any code, explanation, or markdown formatting. Only output valid JSON.

    The data are as follows:
    <history>: {json.dumps(sanitized_history)}
    <context>: {json.dumps(sanitized_context)}
    <target_stay>: {json.dumps(sanitized_target)}
    """
    completion = get_chat_completion(prompt, json_mode=True)

    if completion is None:
        return None

    if isinstance(completion, str):
        try:
            completion = json.loads(completion)
        except json.JSONDecodeError:
            print("⚠️ JSON malformato:", completion)
            return None

    return completion

def single_query_top10_wot_fsq(historical_data, X):
    """
    Make a single query for top-10 most likely places, without using duration or time info.
    param: 
    historical_data: list of past stays (start_time, day_of_week, place_id)
    X: dict with key 'context_stay' (same structure as historical_data)
    """
    sanitized_history = convert_to_serializable(historical_data)
    sanitized_context = convert_to_serializable(X['context_stay'])

    prompt = f"""
    Your task is to predict a user's next location based on his/her activity pattern.
    You will be provided with <history> which is a list containing this user's historical stays, and <context> which provides recent contextual information.
    Stays in both <history> and <context> are in chronological order and have the format: (start_time, day_of_week, place_id), where:
    - start_time: start time of the stay in 12h clock format
    - day_of_week: the day of the week
    - place_id: an integer representing the location ID

    Your goal is to predict the next likely place ID. Output the 10 most probable places (ranked from most to least likely), considering:
    1. patterns in <history> (e.g., repeated visits to places at certain times/days),
    2. recent behavior in <context>.

    Respond ONLY with a single-line JSON object with the following keys:
    - "prediction": a list of integers (place IDs)
    - "reason": a short string explaining your prediction

    Do NOT include any code, explanation, or markdown formatting. Only output valid JSON.

    <history>: {json.dumps(sanitized_history)}
    <context>: {json.dumps(sanitized_context)}
    <next_place_id>: 
    """

    response = get_chat_completion(prompt)

    # DEBUG:
    #print("📦 Risposta JSON (grezza o meno):", response)

    # La funzione get_chat_completion restituisce già un dizionario o None
    if response is None or not isinstance(response, dict):
        print("⚠️ Risposta non valida o assente.")
        return {"prediction": None, "reason": "Invalid JSON format from model"}

    print("✅ JSON interpretato correttamente:\n", json.dumps(response, indent=2))
    return response

def load_results(filename):
    # Load previously saved results from a CSV file    
    results = pd.read_csv(filename)
    return results

def single_user_query(dataname, uid, historical_data, predict_X, predict_y, logger, top_k, is_wt, output_dir, sleep_query, sleep_crash):
    # Numero totale di query da eseguire
    total_queries = len(predict_X)
    logger.info(f"Total_queries: {total_queries}")

    # Inizializzazione del dataframe per i risultati
    current_results = pd.DataFrame(columns=['user_id', 'ground_truth', 'prediction', 'reason'])

    out_filename = f"{uid:02d}.csv"
    out_filepath = os.path.join(output_dir, out_filename)

    # Carica i risultati esistenti (se presenti)
    try:
        current_results = load_results(out_filepath)
        processed_queries = len(current_results)
        logger.info(f"Loaded {processed_queries} previous results.")
    except FileNotFoundError:
        logger.info("No previous results found. Starting from scratch.")
        processed_queries = 0

    # Loop sulle query non ancora processate
    for i in range(processed_queries, total_queries):
        logger.info(f'The {i+1}th sample:')

        try:
            # Seleziona la funzione di query corretta
            if dataname == 'geolife':
                res_dict = (
                    single_query_top1(historical_data, predict_X[i])
                    if is_wt and top_k == 1 else
                    single_query_top10(historical_data, predict_X[i])
                    if is_wt else
                    single_query_top1_wot(historical_data, predict_X[i])
                    if top_k == 1 else
                    single_query_top10_wot(historical_data, predict_X[i])
                )
            elif dataname == 'fsq':
                res_dict = (
                    single_query_top1_fsq(historical_data, predict_X[i])
                    if is_wt and top_k == 1 else
                    single_query_top10_fsq(historical_data, predict_X[i])
                    if is_wt else
                    single_query_top1_wot_fsq(historical_data, predict_X[i])
                    if top_k == 1 else
                    single_query_top10_wot_fsq(historical_data, predict_X[i])
                )
            else:
                raise ValueError(f"Unsupported dataset name: {dataname}")

            logger.info(f"Ground truth: {predict_y[i]}")
            logger.info(f"Pred results: {res_dict}")

            if not isinstance(res_dict, dict) or 'prediction' not in res_dict:
                raise ValueError("Risposta malformata: manca la chiave 'prediction'.")

            if top_k != 1:
                res_dict['prediction'] = str(res_dict['prediction'])  # salva la lista come stringa

            res_dict['user_id'] = uid
            res_dict['ground_truth'] = int(predict_y[i])

        except Exception as e:
            logger.warning(f"⚠️ Errore nella predizione/parsing: {e}")
            res_dict = {
                'user_id': uid,
                'ground_truth': int(predict_y[i]),
                'prediction': -100,
                'reason': str(e)
            }

        # Aggiunta del nuovo risultato
        new_row = pd.DataFrame([res_dict])
        current_results = pd.concat([current_results, new_row], ignore_index=True)

        # 💾 Salvataggio immediato ad ogni iterazione
        current_results.to_csv(out_filepath, index=False)
        logger.info(f"✅ Salvato parziale dopo sample {i+1}/{total_queries} → {out_filepath}")

    logger.info(f"🎉 Completate tutte le predizioni per user {uid}. Risultati salvati in: {out_filepath}")
    
def query_all_user(dataname, uid_list, logger, train_data, num_historical_stay,
                   num_context_stay, test_file, top_k, is_wt, output_dir, sleep_query, sleep_crash):
    for uid in uid_list:
        logger.info(f"=================Processing user {uid}==================")

        user_train = get_user_data(train_data, uid, num_historical_stay, logger)
        historical_data, predict_X, predict_y = organise_data(
            dataname, user_train, test_file, uid, logger, num_context_stay)

        single_user_query(
            dataname, uid, historical_data, predict_X, predict_y,
            logger, top_k=top_k, is_wt=is_wt, output_dir=output_dir,
            sleep_query=sleep_query, sleep_crash=sleep_crash
        )

# Get the remaning user
def get_unqueried_user(dataname, output_dir='output/'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if dataname == "geolife":
        all_user_id = [i+1 for i in range(45)]
    elif dataname == "fsq":
        all_user_id = [i+1 for i in range(535)]
    processed_id = [int(file.split('.')[0]) for file in os.listdir(output_dir) if file.endswith('.csv')]
    remain_id = [i for i in all_user_id if i not in processed_id]
    
    #DEBUG:
    #print(remain_id)
    #print(f"Number of the remaining id: {len(remain_id)}")
    return remain_id

def main():
    # Parameters
    dataname = "geolife"  # specify the dataset, geolife or fsq.
    num_historical_stay = 30  # M
    num_context_stay = 5  # N
    top_k = 10  # the number of output places k
    with_time = False  # whether incorporate temporal information for target stay
    sleep_single_query = 1  # the sleep time between queries
    sleep_if_crash = 5  # the sleep time if the server crashes
    output_dir = f"output/{dataname}/top10_wot"  # the output path
    log_dir = f"logs/{dataname}/top10_wot"  # the log dir

    # Step 1: Get dataset
    try:
        tv_data, test_file = get_dataset(dataname)
        # DEBUG:
        #print("✅ Dataset loaded successfully.")
        # DEBUG:
        #print(f"Number of total training and validation samples: {len(tv_data)}")
    except Exception as e:
        print(f"❌ Errore nel caricamento del dataset: {e}")
        return  # Stop the function if dataset loading fails

    # Step 2: Set up logging
    try:
        logger = get_logger('my_logger', log_dir=log_dir)
        #DEBUG:
        #print("✅ Logger initialized successfully.")
    except Exception as e:
        print(f"❌ Errore nell'inizializzazione del logger: {e}")
        return  # Stop the function if logger initialization fails

    # Step 3: Get unqueried users
    try:
        uid_list = get_unqueried_user(dataname, output_dir)
        # DEBUG:
        #print(f"✅ Unqueried user list: {uid_list}")
        #print(f"Number of unqueried users: {len(uid_list)}")
    except FileNotFoundError:
        print(f"❌ Errore: File non trovato. Assicurati che il percorso sia corretto.")
        return
    except pd.errors.EmptyDataError:
        print(f"❌ Errore: Il file CSV è vuoto. Assicurati che il file contenga dati.")
        return
    except pd.errors.ParserError:
        print(f"❌ Errore: Errore di parsing del file CSV. Controlla il formato del file.")
        return
    except Exception as e:
        print(f"❌ Errore nel recupero degli utenti non interrogati: {e}")
        return  # Stop the function if fetching users fails

    # Step 4: Run query for all users
    try:
        query_all_user(
            dataname, uid_list, logger, tv_data, num_historical_stay, num_context_stay,
            test_file, output_dir=output_dir, top_k=top_k, is_wt=with_time,
            sleep_query=sleep_single_query, sleep_crash=sleep_if_crash
        )
        print("✅ Query to all users completed successfully.")
    except Exception as e:
        print(f"❌ Errore durante l'esecuzione della query per gli utenti: {e}")
        return  # Stop the function if the query fails

    print("Query done")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n\nInterruzione manuale rilevata. Uscita in modo sicuro...")