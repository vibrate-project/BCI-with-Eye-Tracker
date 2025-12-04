# -*- coding: utf-8 -*-
"""
Reproducible Code - Uses saved randomness state for exact reproduction
Modified to make final emotional state predictions by averaging across problems 8, 10, and 12
"""

import numpy as np
import pickle
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt

def load_complete_randomness_state(filename='complete_randomness_state.pkl'):
    import random
    with open(filename, 'rb') as f:
        randomness_state = pickle.load(f)
    np.random.set_state(randomness_state['numpy_random_state'])
    tf.random.set_seed(randomness_state['tensorflow_random_seed'])
    random.setstate(randomness_state['python_random_state'])
    import os
    os.environ['PYTHONHASHSEED'] = randomness_state['PYTHONHASHSEED']
    return randomness_state

def load_data():
    # Load eye-tracking data
    with open('Pickle/SegmentsSeparatedOneByOne_AllPeople.pkl', 'rb') as f:
        eye_data = pickle.load(f)
    
    # Load parameter data
    with open('Pickle/ParData_AllPeople.pkl', 'rb') as f:
        par_data = pickle.load(f)
    
    return eye_data, par_data


problem_lengths_8_13 = [1200, 1200, 840, 600, 840, 600] 

# Constants
N_PARTICIPANTS = 19
N_EMOTIONAL_STATES = 2
N_PROBLEMS = 6  
N_PARAMETERS = 23
PROBLEM_INDICES = [7, 8, 9, 10, 11, 12]  
MAX_LENGTH = max(problem_lengths_8_13)  

def augment_time_series(series, max_shift=50, noise_factor=0.01):
    shift = np.random.randint(-max_shift, max_shift)
    if shift > 0:
        augmented = np.concatenate([series[shift:], np.zeros((shift, series.shape[1]))])
    elif shift < 0:
        augmented = np.concatenate([np.zeros((-shift, series.shape[1])), series[:shift]])
    else:
        augmented = series.copy()
    noise = np.random.normal(0, noise_factor, augmented.shape)
    augmented += noise
    return augmented

def augment_dataset(X, y, augment_factor=3):
    X_augmented = []
    y_augmented = []
    for i in range(len(X)):
        X_augmented.append(X[i])
        y_augmented.append(y[i])
        for _ in range(augment_factor):
            X_augmented.append(augment_time_series(X[i]))
            y_augmented.append(y[i])
    return np.array(X_augmented), np.array(y_augmented)

def create_multimodal_model(eye_input_shape, par_input_shape):
    # Eye-tracking data branch
    eye_input = layers.Input(shape=eye_input_shape, name='eye_input')
    x = layers.Masking(mask_value=0.0)(eye_input)
    x = layers.Conv1D(32, 5, activation='relu')(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Conv1D(64, 5, activation='relu')(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(0.3)(x)
    x = layers.LSTM(64, return_sequences=False)(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(32, activation='relu')(x)
    x = models.Model(inputs=eye_input, outputs=x)
    
    # Parameter data branch
    par_input = layers.Input(shape=par_input_shape, name='par_input')
    y = layers.Dense(16, activation='relu')(par_input)
    y = layers.Dropout(0.2)(y)
    y = layers.Dense(8, activation='relu')(y)
    y = models.Model(inputs=par_input, outputs=y)
    
    # Combined model
    combined = layers.concatenate([x.output, y.output])
    z = layers.Dense(32, activation='relu')(combined)
    z = layers.Dropout(0.4)(z)
    z = layers.Dense(1, activation='sigmoid')(z)
    
    model = models.Model(inputs=[x.input, y.input], outputs=z)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

def prepare_data(eye_data, par_data):
    # Reshape eye data to [participants, emotional_state, problems, time, coordinates]
    eye_data_reshaped = np.zeros((N_PARTICIPANTS, N_EMOTIONAL_STATES, N_PROBLEMS, MAX_LENGTH, 2))
    for problem_idx, original_problem_idx in enumerate(PROBLEM_INDICES):
        for exp_idx in range(N_PARTICIPANTS * 2):
            participant_idx = exp_idx // 2
            emotional_state = exp_idx % 2
            seq_length = problem_lengths_8_13[problem_idx]
            exp_eye_data = np.array(eye_data[original_problem_idx][exp_idx])
            padded_sequence = np.zeros((MAX_LENGTH, 2))
            actual_length = min(seq_length, exp_eye_data.shape[0])
            padded_sequence[:actual_length, :] = exp_eye_data[:actual_length, :]
            eye_data_reshaped[participant_idx, emotional_state, problem_idx, :, :] = padded_sequence
    for problem_idx in range(N_PROBLEMS):
        problem_data = eye_data_reshaped[:, :, problem_idx, :, :]
        mean = np.mean(problem_data)
        std = np.std(problem_data)
        eye_data_reshaped[:, :, problem_idx, :, :] = (problem_data - mean) / (std + 1e-8)
    if isinstance(par_data, list):
        par_data = np.array(par_data)
    par_data_normalized = np.zeros_like(par_data)
    for problem_idx in range(N_PROBLEMS): 
        for param_idx in range(N_PARAMETERS):
            param_data = par_data[:, :, param_idx, problem_idx]
            mean = np.mean(param_data)
            std = np.std(param_data)
            par_data_normalized[:, :, param_idx, problem_idx] = (param_data - mean) / (std + 1e-8)
    
    return eye_data_reshaped, par_data_normalized

# Train and evaluate models
def train_and_evaluate_reproducible(eye_data, par_data):
    eye_data_prepared, par_data_prepared = prepare_data(eye_data, par_data)
    all_results = []
    
    for problem_idx in range(N_PROBLEMS):  # Only 6 problems now
        original_problem_num = PROBLEM_INDICES[problem_idx] + 1  # Convert to 1-based numbering
        X_eye = []
        y = []
        X_par = []
        
        for participant in range(N_PARTICIPANTS):
            for state in range(N_EMOTIONAL_STATES):
                eye_sample = eye_data_prepared[participant, state, problem_idx, :, :]
                X_eye.append(eye_sample)
                y.append(state)
                par_sample = par_data_prepared[participant, state, :, problem_idx]
                X_par.append(par_sample)
        
        X_eye = np.array(X_eye)
        X_par = np.array(X_par)
        y = np.array(y)
        (X_eye_train, X_eye_test, 
         X_par_train, X_par_test, 
         y_train, y_test) = train_test_split(X_eye, X_par, y, test_size=0.2, 
                                            random_state=42, stratify=y)
        
        indices = np.arange(len(y))
        train_indices, test_indices = train_test_split(indices, test_size=0.2, random_state=42, stratify=y)
        X_eye_train_aug, y_train_aug = augment_dataset(X_eye_train, y_train)
        X_par_train_aug = np.repeat(X_par_train, 4, axis=0)  # Match augmentation factor
        model = create_multimodal_model((MAX_LENGTH, 2), (N_PARAMETERS,))
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=15, restore_best_weights=True, verbose=1
        )
        
        history = model.fit(
            [X_eye_train_aug, X_par_train_aug], y_train_aug,
            epochs=100, batch_size=16, validation_split=0.2, 
            callbacks=[early_stopping], verbose=1)
        test_loss, test_acc = model.evaluate(
            [X_eye_test, X_par_test], y_test, verbose=0)
        y_pred_proba = model.predict([X_eye_test, X_par_test], verbose=0)
        y_pred = (y_pred_proba > 0.5).astype(int)
        auc = roc_auc_score(y_test, y_pred_proba)
        problem_results = {
            'problem_idx': problem_idx,
            'original_problem_num': original_problem_num,
            'model': model,
            'test_acc': test_acc,
            'test_loss': test_loss,
            'auc': auc,
            'y_test': y_test,
            'y_pred_proba': y_pred_proba,
            'y_pred': y_pred,
            'train_indices': train_indices,
            'test_indices': test_indices,
            'model_history': history.history
        }
        
        all_results.append(problem_results)
    
    return all_results

def make_final_predictions(results, target_problems=[8, 10, 12]):
    target_results = {}
    for res in results:
        if res['original_problem_num'] in target_problems:
            target_results[res['original_problem_num']] = res
    
    if not target_results:
        print("No results found for the specified problems!")
        return None
    first_problem = list(target_results.values())[0]
    test_indices = first_problem['test_indices']
    index_to_participant = {}
    for test_idx in test_indices:
        participant_idx = test_idx // 2
        emotional_state = test_idx % 2   
        index_to_participant[test_idx] = (participant_idx, emotional_state)
    participant_predictions = {}
    
    for test_idx in test_indices:
        participant_idx, true_state = index_to_participant[test_idx]
        position_in_test = np.where(first_problem['test_indices'] == test_idx)[0][0]
        
        if participant_idx not in participant_predictions:
            participant_predictions[participant_idx] = {
                'true_state': true_state,
                'problem_predictions': {problem: [] for problem in target_problems},
                'problem_classes': {problem: [] for problem in target_problems}
            }
        for problem_num in target_problems:
            if problem_num in target_results:
                res = target_results[problem_num]
                pred_proba = res['y_pred_proba'][position_in_test][0]
                pred_class = res['y_pred'][position_in_test][0]
                
                participant_predictions[participant_idx]['problem_predictions'][problem_num].append(pred_proba)
                participant_predictions[participant_idx]['problem_classes'][problem_num].append(pred_class)
    
    print("\nFinal Emotional State Predictions:")
    print("Participant | True State | P8 Pred | P10 Pred | P12 Pred | Majority Vote | Correct")
    print("-" * 85)
    
    correct_predictions = 0
    total_predictions = 0
    
    for participant_idx in sorted(participant_predictions.keys()):
        data = participant_predictions[participant_idx]
        true_state = data['true_state']
        
        p8_pred = data['problem_classes'][8][0] if 8 in data['problem_classes'] else -1
        p10_pred = data['problem_classes'][10][0] if 10 in data['problem_classes'] else -1
        p12_pred = data['problem_classes'][12][0] if 12 in data['problem_classes'] else -1
        
        votes = [p8_pred, p10_pred, p12_pred]
        positive_votes = sum(1 for vote in votes if vote == 1)
        negative_votes = sum(1 for vote in votes if vote == 0)
        
        if positive_votes > negative_votes:
            final_prediction = 1
        elif negative_votes > positive_votes:
            final_prediction = 0
        else:
            avg_proba = np.mean([
                data['problem_predictions'][8][0] if 8 in data['problem_predictions'] else 0.5,
                data['problem_predictions'][10][0] if 10 in data['problem_predictions'] else 0.5,
                data['problem_predictions'][12][0] if 12 in data['problem_predictions'] else 0.5
            ])
            final_prediction = 1 if avg_proba > 0.5 else 0
        
        correct = final_prediction == true_state
        if correct:
            correct_predictions += 1
        total_predictions += 1
        
        print(f"P{participant_idx+1:2d}       | "
              f"{'Positive' if true_state == 1 else 'Negative':10s} | "
              f"{'Positive' if p8_pred == 1 else 'Negative':7s} | "
              f"{'Positive' if p10_pred == 1 else 'Negative':8s} | "
              f"{'Positive' if p12_pred == 1 else 'Negative':8s} | "
              f"{'Positive' if final_prediction == 1 else 'Negative':12s} | "
              f"{'Yes' if correct else 'No':7s}")
    
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    print("-" * 85)
    print(f"Overall Accuracy: {correct_predictions}/{total_predictions} ({accuracy:.4f})")
    return participant_predictions, accuracy


if __name__ == "__main__":
    load_complete_randomness_state('randomness_state_before_training.pkl')
    eye_data, par_data = load_data()
    results = train_and_evaluate_reproducible(eye_data, par_data)
    participant_predictions, final_accuracy = make_final_predictions(results, [8, 10, 12])
    print("\nAnalysis completed successfully!")