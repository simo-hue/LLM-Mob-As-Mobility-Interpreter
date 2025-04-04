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

    print("Number of total test sample: ", len(test_file))
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
def single_query_top1(historical_data, X):
    """
    Make a single query.
    param: 
    X: one single sample containing context_stay and target_stay
    """
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
    
    Please infer what the <next_place_id> is (i.e., the most likely place ID), considering the following aspects:
    1. the activity pattern of this user that you leared from <history>, e.g., repeated visit to a certain place during certain time;
    2. the context stays in <context>, which provide more recent activities of this user; 
    3. the temporal information (i.e., start_time and day_of_week) of target stay, which is important because people's activity varies during different times (e.g., nighttime versus daytime)
    and on different days (e.g., weekday versus weekend).

    Please organize your answer in a JSON object containing following keys: "prediction" (place ID) and "reason" (a concise explanation that supports your prediction). Do not include line breaks in your output.

    The data are as follows:
    <history>: {historical_data}
    <context>: {X['context_stay']}
    <target_stay>: {X['target_stay']}
    """
    completion = get_chat_completion(prompt)
    return completion

# Make a single query of 10 most likely places
def single_query_top10(historical_data, X):
    """
    Make a single query.
    param: 
    X: one single sample containing context_stay and target_stay
    """
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
    1. the activity pattern of this user that you leared from <history>, e.g., repeated visits to certain places during certain times;
    2. the context stays in <context>, which provide more recent activities of this user; 
    3. the temporal information (i.e., start_time and day_of_week) of target stay, which is important because people's activity varies during different time (e.g., nighttime versus daytime)
    and on different days (e.g., weekday versus weekend).

    Please organize your answer in a JSON object containing following keys:
    "prediction" (the ID of the ten most probable places in descending order of probability) and "reason" (a concise explanation that supports your prediction). Do not include line breaks in your output.

    The data are as follows:
    <history>: {historical_data}
    <context>: {X['context_stay']}
    <target_stay>: {X['target_stay']}
    """
    completion = get_chat_completion(prompt)
    return completion

# Make a single query of 10 most likely places without time information
def single_query_top1_wot(historical_data, X):
    """
    Make a single query.
    param: 
    X: one single sample containing context_stay and target_stay
    """
    prompt = f"""
    Your task is to predict a user's next location based on his/her activity pattern.
    You will be provided with <history> which is a list containing this user's historical stays, then <context> which provide contextual information 
    about where and when this user has been to recently. Stays in both <history> and <context> are in chronological order.
    Each stay takes on such form as (start_time, day_of_week, duration, place_id). The detailed explanation of each element is as follows:
    start_time: the start time of the stay in 12h clock format.
    day_of_week: indicating the day of the week.
    duration: an integer indicating the duration (in minute) of each stay. 
    place_id: an integer representing the unique place ID, which indicates where the stay is.    
    
    Please infer what the <next_place_id> is (i.e., the most likely place ID), considering the following aspects:
    1. the activity pattern of this user that you leared from <history>, e.g., repeated visit to a certain place during certain time;
    2. the context stays in <context>, which provide more recent activities of this user.

    Please organize your answer in a JSON object containing following keys: "prediction" (place ID) and "reason" (a concise explanation that supports your prediction). Do not include line breaks in your output.

    The data are as follows:
    <history>: {historical_data}
    <context>: {X['context_stay']}
    """
    completion = get_chat_completion(prompt)
    return completion

# 
def single_query_top10_wot(historical_data, X):
    """
    Make a single query of 10 most likely places, without time information
    param: 
    X: one single sample containing context_stay and target_stay
    """
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
  
    Please organize your answer in a JSON object containing following keys:
    "prediction" (the ID of the ten most probable places in descending order of probability) and "reason" (a concise explanation that supports your prediction). Do not use line breaks in the reason.

    The data are as follows:
    <history>: {historical_data}
    <context>: {X['context_stay']}
    """
    completion = get_chat_completion(prompt)
    return completion


def single_query_top1_fsq(historical_data, X):
    """
    Make a single query.
    param: 
    X: one single sample containing context_stay and target_stay
    """
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

    Please organize your answer in a JSON object containing following keys:
    "prediction" (place ID) and "reason" (a concise explanation that supports your prediction)

    The data are as follows:
    <history>: {historical_data}
    <context>: {X['context_stay']}
    <target_stay>: {X['target_stay']}
    """
    completion = get_chat_completion(prompt)
    return completion

# Make a single query of 10 most likely places 
def single_query_top1_wot_fsq(historical_data, X):
    """
    Make a single query.
    param: 
    X: one single sample containing context_stay and target_stay
    """
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

    Please organize your answer in a JSON object containing following keys: "prediction" (place ID) and "reason" (a concise explanation that supports your prediction). Do not include line breaks in your output.

    The data are as follows:
    <history>: {historical_data}
    <context>: {X['context_stay']}
    """
    completion = get_chat_completion(prompt)
    return completion

# 
def single_query_top10_fsq(historical_data, X):
    """
    Make a single query.
    param: 
    X: one single sample containing context_stay and target_stay
    """
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

    Please organize your answer in a JSON object containing following keys:
    "prediction" (the ID of the ten most probable places in descending order of probability) and "reason" (a concise explanation that supports your prediction)

    The data are as follows:
    <history>: {historical_data}
    <context>: {X['context_stay']}
    <target_stay>: {X['target_stay']}
    """
    completion = get_chat_completion(prompt)
    return completion


def single_query_top10_wot_fsq(historical_data, X):
    """
    Make a single query of 10 most likely places, without time information
    param: 
    X: one single sample containing context_stay and target_stay
    """
    prompt = f"""
    Your task is to predict a user's next location based on his/her activity pattern.
    You will be provided with <history> which is a list containing this user's historical stays, then <context> which provide contextual information 
    about where and when this user has been to recently. Stays in both <history> and <context> are in chronological order.
    Each stay takes on such form as (start_time, day_of_week, place_id). The detailed explanation of each element is as follows:
    start_time: the start time of the stay in 12h clock format.
    day_of_week: indicating the day of the week.
    place_id: an integer representing the unique place ID, which indicates where the stay is.

    Please infer what the <next_place_id> might be (please output the 10 most likely places which are ranked in descending order in terms of probability), considering the following aspects:
    1. the activity pattern of this user that you leared from <history>, e.g., repeated visits to certain places during certain times.
    2. the context stays in <context>, which provide more recent activities of this user.
  
    Please organize your answer in a JSON object containing following keys:
    "prediction" (the ID of the ten most probable places in descending order of probability) and "reason" (a concise explanation that supports your prediction). Do not use line breaks in the reason.

    The data are as follows:
    <history>: {historical_data}
    <context>: {X['context_stay']}
    <next_place_id>: 
    """
    completion = get_chat_completion(prompt)
    return completion

def load_results(filename):
    # Load previously saved results from a CSV file    
    results = pd.read_csv(filename)
    return results

def single_user_query(dataname, uid, historical_data, predict_X, predict_y, logger, 
                      top_k, is_wt, output_dir, sleep_query, sleep_crash):
    # Initialize variables
    total_queries = len(predict_X)
    logger.info(f"Total_queries: {total_queries}")

    processed_queries = 0
    current_results = pd.DataFrame({
        'user_id': None,
        'ground_truth': None,
        'prediction': None,
        'reason': None
    }, index=[])

    out_filename = f"{uid:02d}" + ".csv"
    out_filepath = os.path.join(output_dir, out_filename)

    try:
        # Attempt to load previous results if available
        current_results = load_results(out_filepath)
        processed_queries = len(current_results)
        logger.info(f"Loaded {processed_queries} previous results.")
    except FileNotFoundError:
        logger.info("No previous results found. Starting from scratch.")

    # Process remaining queries
    for i in range(processed_queries, total_queries):
        logger.info(f'The {i+1}th sample: ')

        if dataname == 'geolife':
            if is_wt is True:
                if top_k == 1:
                    prompt = single_query_top1(historical_data, predict_X[i])
                elif top_k == 10:
                    prompt = single_query_top10(historical_data, predict_X[i])
                else:
                    raise ValueError(f"The top_k must be one of 1, 10. However, {top_k} was provided")
            else:
                if top_k == 1:
                    prompt = single_query_top1_wot(historical_data, predict_X[i])
                elif top_k == 10:
                    prompt = single_query_top10_wot(historical_data, predict_X[i])
                else:
                    raise ValueError(f"The top_k must be one of 1, 10. However, {top_k} was provided")
        elif dataname == 'fsq':
            if is_wt is True:
                if top_k == 1:
                    prompt = single_query_top1_fsq(historical_data, predict_X[i])
                elif top_k == 10:
                    prompt = single_query_top10_fsq(historical_data, predict_X[i])
                else:
                    raise ValueError(f"The top_k must be one of 1, 10. However, {top_k} was provided")
            else:
                if top_k == 1:
                    prompt = single_query_top1_wot_fsq(historical_data, predict_X[i])
                elif top_k == 10:
                    prompt = single_query_top10_wot_fsq(historical_data, predict_X[i])
                else:
                    raise ValueError(f"The top_k must be one of 1, 10. However, {top_k} was provided")

        # Usa il modello locale
        response = get_chat_completion(prompt)  

        # Log della risposta
        logger.info(f"Pred results: {response}")
        logger.info(f"Ground truth: {predict_y[i]}")

        try:
            res_dict = json.loads(response)  # Usa JSON parsing più sicuro
            if top_k != 1:
                res_dict['prediction'] = str(res_dict['prediction'])
                res_dict['user_id'] = uid
                res_dict['ground_truth'] = predict_y[i]
        except Exception as e:
            res_dict = {'user_id': uid, 'ground_truth': predict_y[i], 'prediction': -100, 'reason': response}
            logger.info(e)
            logger.info(f"API request failed for the {i+1}th query")

        # Aggiungi il risultato ai dati attuali
        new_row = pd.DataFrame(res_dict, index=[0])
        current_results = pd.concat([current_results, new_row], ignore_index=True)

    # Salva i risultati
    current_results.to_csv(out_filepath, index=False)
    logger.info(f"Saved {len(current_results)} results to {out_filepath}")

    # Continua se ci sono ancora query da processare ( ma con dei limiti )
    max_retries = 3
    retry_count = 0
    while len(current_results) < total_queries and retry_count < max_retries:
        logger.info(f"Restarting queries (attempt {retry_count + 1}/{max_retries})...")
        retry_count += 1
        single_user_query(dataname, uid, historical_data, predict_X, predict_y,
                      logger, top_k, is_wt, output_dir, sleep_query, sleep_crash)

def query_all_user(dataname, uid_list, logger, train_data, num_historical_stay,
                   num_context_stay, test_file, top_k, is_wt, output_dir, sleep_query, sleep_crash):
    for uid in uid_list:
        logger.info(f"=================Processing user {uid}==================")
        user_train = get_user_data(train_data, uid, num_historical_stay, logger)
        historical_data, predict_X, predict_y = organise_data(dataname, user_train, test_file, uid, logger, num_context_stay)
        single_user_query(dataname, uid, historical_data, predict_X, predict_y, logger, top_k=top_k, 
                          is_wt=is_wt, output_dir=output_dir, sleep_query=sleep_query, sleep_crash=sleep_crash)

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
    print(remain_id)
    print(f"Number of the remaining id: {len(remain_id)}")
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
        print("✅ Dataset loaded successfully.")
        print(f"Number of total training and validation samples: {len(tv_data)}")
    except Exception as e:
        print(f"❌ Errore nel caricamento del dataset: {e}")
        return  # Stop the function if dataset loading fails

    # Step 2: Set up logging
    try:
        logger = get_logger('my_logger', log_dir=log_dir)
        print("✅ Logger initialized successfully.")
    except Exception as e:
        print(f"❌ Errore nell'inizializzazione del logger: {e}")
        return  # Stop the function if logger initialization fails

    # Step 3: Get unqueried users
    try:
        uid_list = get_unqueried_user(dataname, output_dir)
        print(f"✅ Unqueried user list: {uid_list}")
        print(f"Number of unqueried users: {len(uid_list)}")
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