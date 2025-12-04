# -*- coding: utf-8 -*-
"""
Created on Sat Nov 15 14:30:21 2025

@author: teodo
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

# Function to load and restore randomness state
def load_randomness_state(filename='randomness_state.pkl'):
    """Load and restore randomness state from file"""
    with open(filename, 'rb') as f:
        randomness_state = pickle.load(f)
    
    # Restore numpy random state
    np.random.set_state(randomness_state['numpy_random_state'])
    
    # Restore tensorflow random seed
    tf.random.set_seed(randomness_state['tensorflow_random_seed'])
    
    print(f"Randomness state loaded from {filename}")
    return randomness_state

# Load data
def load_data():
    # Load eye-tracking data
    with open('Pickle/SegmentsSeparatedOneByOne_AllPeople.pkl', 'rb') as f:
        eye_data = pickle.load(f)
    
    # Load parameter data
    with open('Pickle/ParData_AllPeople.pkl', 'rb') as f:
        par_data = pickle.load(f)
    
    return eye_data, par_data

# Problem lengths for problems 8-13 only (indices 7-12 in 0-based)
problem_lengths_8_13 = [1200, 1200, 840, 600, 840, 600]  # Problems 8-13 lengths

# Constants
N_PARTICIPANTS = 19
N_EMOTIONAL_STATES = 2
N_PROBLEMS = 6  # Only problems 8-13
N_PARAMETERS = 23
PROBLEM_INDICES = [7, 8, 9, 10, 11, 12]  # 0-based indices for problems 8-13
MAX_LENGTH = max(problem_lengths_8_13)  # Use max length from our selected problems

# Data augmentation functions - MUST BE IDENTICAL TO ORIGINAL
def augment_time_series(series, max_shift=50, noise_factor=0.01):
    """Augment time series data with random shifts and noise - IDENTICAL TO ORIGINAL"""
    # Random shift
    shift = np.random.randint(-max_shift, max_shift)
    if shift > 0:
        augmented = np.concatenate([series[shift:], np.zeros((shift, series.shape[1]))])
    elif shift < 0:
        augmented = np.concatenate([np.zeros((-shift, series.shape[1])), series[:shift]])
    else:
        augmented = series.copy()
    
    # Add random noise
    noise = np.random.normal(0, noise_factor, augmented.shape)
    augmented += noise
    
    return augmented

def augment_dataset(X, y, augment_factor=3):
    """Augment dataset by creating modified copies of each sample - IDENTICAL TO ORIGINAL"""
    X_augmented = []
    y_augmented = []
    
    for i in range(len(X)):
        X_augmented.append(X[i])
        y_augmented.append(y[i])
        
        for _ in range(augment_factor):
            X_augmented.append(augment_time_series(X[i]))
            y_augmented.append(y[i])
    
    return np.array(X_augmented), np.array(y_augmented)

# Multimodal model for problems 8-13 - MUST BE IDENTICAL TO ORIGINAL
def create_multimodal_model(eye_input_shape, par_input_shape):
    """Create model for problems with parameters (8-13) - IDENTICAL TO ORIGINAL"""
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

# Prepare data for training - Only problems 8-13 - MUST BE IDENTICAL TO ORIGINAL
def prepare_data(eye_data, par_data):
    # Reshape eye data to [participants, emotional_state, problems, time, coordinates]
    # Only for problems 8-13 (6 problems)
    eye_data_reshaped = np.zeros((N_PARTICIPANTS, N_EMOTIONAL_STATES, N_PROBLEMS, MAX_LENGTH, 2))
    
    # Process only problems 8-13
    for problem_idx, original_problem_idx in enumerate(PROBLEM_INDICES):
        for exp_idx in range(N_PARTICIPANTS * 2):
            participant_idx = exp_idx // 2
            emotional_state = exp_idx % 2
            seq_length = problem_lengths_8_13[problem_idx]
            
            # Get the eye data for this problem and experiment
            exp_eye_data = np.array(eye_data[original_problem_idx][exp_idx])
            
            # Pad the sequence
            padded_sequence = np.zeros((MAX_LENGTH, 2))
            actual_length = min(seq_length, exp_eye_data.shape[0])
            padded_sequence[:actual_length, :] = exp_eye_data[:actual_length, :]
            
            eye_data_reshaped[participant_idx, emotional_state, problem_idx, :, :] = padded_sequence
    
    # Normalize eye data per problem
    for problem_idx in range(N_PROBLEMS):
        problem_data = eye_data_reshaped[:, :, problem_idx, :, :]
        mean = np.mean(problem_data)
        std = np.std(problem_data)
        eye_data_reshaped[:, :, problem_idx, :, :] = (problem_data - mean) / (std + 1e-8)
    
    # Prepare parameter data - Only problems 8-13
    # Convert par_data to numpy array if it's a list
    if isinstance(par_data, list):
        par_data = np.array(par_data)
    
    # par_data should be in shape [19, 2, 23, 6] - we use all 6 parameter problems
    # Normalize parameter data
    par_data_normalized = np.zeros_like(par_data)
    for problem_idx in range(N_PROBLEMS):  # All 6 problems are parameter problems
        for param_idx in range(N_PARAMETERS):
            param_data = par_data[:, :, param_idx, problem_idx]
            mean = np.mean(param_data)
            std = np.std(param_data)
            par_data_normalized[:, :, param_idx, problem_idx] = (param_data - mean) / (std + 1e-8)
    
    return eye_data_reshaped, par_data_normalized

# Train and evaluate models - Only problems 8-13 - IDENTICAL TO ORIGINAL
def train_and_evaluate(eye_data, par_data):
    """Train models - MUST BE CALLED AFTER LOADING RANDOMNESS STATE"""
    
    # Prepare data
    eye_data_prepared, par_data_prepared = prepare_data(eye_data, par_data)
    
    # Store results
    all_results = []
    
    # For each problem in 8-13, create and train a model
    for problem_idx in range(N_PROBLEMS):  # Only 6 problems now
        original_problem_num = PROBLEM_INDICES[problem_idx] + 1  # Convert to 1-based numbering
        print(f"\nTraining model for Problem {original_problem_num} (index {PROBLEM_INDICES[problem_idx]})")
        
        # Prepare problem-specific data
        X_eye = []
        y = []
        X_par = []
        
        for participant in range(N_PARTICIPANTS):
            for state in range(N_EMOTIONAL_STATES):
                # Eye data for this problem
                eye_sample = eye_data_prepared[participant, state, problem_idx, :, :]
                X_eye.append(eye_sample)
                y.append(state)
                
                # Parameter data for problems 8-13
                par_sample = par_data_prepared[participant, state, :, problem_idx]  # problem_idx is 0-5 for our 6 problems
                X_par.append(par_sample)
        
        X_eye = np.array(X_eye)
        X_par = np.array(X_par)
        y = np.array(y)
        
        # Split data into train and test - MUST USE SAME random_state=42
        (X_eye_train, X_eye_test, 
         X_par_train, X_par_test, 
         y_train, y_test) = train_test_split(X_eye, X_par, y, test_size=0.2, 
                                            random_state=42, stratify=y)
        
        # Generate the indices used for train/test split - IDENTICAL TO ORIGINAL
        indices = np.arange(len(y))
        train_indices, test_indices = train_test_split(indices, test_size=0.2, random_state=42, stratify=y)
        
        # Augment training data - WILL USE SAME RANDOMNESS AS SAVED STATE
        X_eye_train_aug, y_train_aug = augment_dataset(X_eye_train, y_train)
        X_par_train_aug = np.repeat(X_par_train, 4, axis=0)  # Match augmentation factor
        
        # Create and train multimodal model
        model = create_multimodal_model((MAX_LENGTH, 2), (N_PARAMETERS,))
        
        # Add early stopping - IDENTICAL TO ORIGINAL
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=15, restore_best_weights=True, verbose=1
        )
        
        history = model.fit(
            [X_eye_train_aug, X_par_train_aug], y_train_aug,
            epochs=100, batch_size=16, validation_split=0.2, 
            callbacks=[early_stopping], verbose=1)
        
        # Evaluate
        test_loss, test_acc = model.evaluate(
            [X_eye_test, X_par_test], y_test, verbose=0)
        
        # Predict probabilities
        y_pred_proba = model.predict([X_eye_test, X_par_test], verbose=0)
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        # Calculate additional metrics
        auc = roc_auc_score(y_test, y_pred_proba)
        
        print(f"Problem {original_problem_num} - Test Accuracy: {test_acc:.4f}, AUC: {auc:.4f}")
        
        # Store results
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

# Enhanced evaluation function
def enhanced_evaluation(results):
    print("\n" + "="*50)
    print("DETAILED RESULTS FOR PROBLEMS 8-13")
    print("="*50)
    
    # Individual problem results
    print("\n--- Individual Problem Performance ---")
    problem_accuracies = []
    problem_aucs = []
    
    for res in results:
        print(f"\nProblem {res['original_problem_num']}:")
        print(f"  Accuracy: {res['test_acc']:.4f}")
        print(f"  AUC: {res['auc']:.4f}")
        print(f"  Loss: {res['test_loss']:.4f}")
        
        # Classification report
        report = classification_report(res['y_test'], res['y_pred'], 
                                     target_names=['Negative', 'Positive'])
        print("  Classification Report:")
        for line in report.split('\n'):
            if line.strip():
                print(f"    {line}")
        
        problem_accuracies.append(res['test_acc'])
        problem_aucs.append(res['auc'])
    
    # Average performance
    print(f"\n--- Average Performance ---")
    print(f"Mean Accuracy: {np.mean(problem_accuracies):.4f} (+/- {np.std(problem_accuracies):.4f})")
    print(f"Mean AUC: {np.mean(problem_aucs):.4f} (+/- {np.std(problem_aucs):.4f})")
    
    return problem_accuracies, problem_aucs

# Function to save complete experiment state
def save_experiment_state(results, filename='experiment_state.pkl'):
    """Save complete experiment state including models and results"""
    experiment_state = {
        'results': results,
        'constants': {
            'N_PARTICIPANTS': N_PARTICIPANTS,
            'N_EMOTIONAL_STATES': N_EMOTIONAL_STATES,
            'N_PROBLEMS': N_PROBLEMS,
            'N_PARAMETERS': N_PARAMETERS,
            'PROBLEM_INDICES': PROBLEM_INDICES,
            'MAX_LENGTH': MAX_LENGTH,
            'problem_lengths_8_13': problem_lengths_8_13
        }
    }
    
    with open(filename, 'wb') as f:
        pickle.dump(experiment_state, f)
    print(f"Complete experiment state saved to {filename}")

# Function to load experiment state
def load_experiment_state(filename='experiment_state.pkl'):
    """Load complete experiment state"""
    with open(filename, 'rb') as f:
        experiment_state = pickle.load(f)
    
    print(f"Experiment state loaded from {filename}")
    return experiment_state

# Main execution - REPRODUCIBLE VERSION
if __name__ == "__main__":
    # CRITICAL: First restore the randomness state from the saved file
    print("Restoring randomness state for exact reproducibility...")
    randomness_state = load_randomness_state('randomness_state_before_training.pkl')
    
    # Load data
    print("Loading data...")
    eye_data, par_data = load_data()
    
    # Check the structure of the loaded data
    print(f"Eye data type: {type(eye_data)}")
    if isinstance(eye_data, list):
        print(f"Eye data length: {len(eye_data)}")
    
    print(f"Parameter data type: {type(par_data)}")
    if isinstance(par_data, list):
        print(f"Parameter data length: {len(par_data)}")
    
    # Train and evaluate models for problems 8-13 only
    # This will now use the EXACT SAME randomness as the original run
    print(f"\nTraining multimodal models for Problems 8-13 with restored randomness...")
    results = train_and_evaluate(eye_data, par_data)
    
    # Save complete experiment state
    save_experiment_state(results, 'experiment_state_reproduced.pkl')
    
    # Enhanced evaluation
    problem_accuracies, problem_aucs = enhanced_evaluation(results)
    
    # Participant-level analysis
    print("\n" + "="*50)
    print("PARTICIPANT-LEVEL ANALYSIS")
    print("="*50)
    
    # Collect all test predictions and labels
    all_y_test = []
    all_y_pred_proba = []
    
    for res in results:
        all_y_test.append(res['y_test'])
        all_y_pred_proba.append(res['y_pred_proba'])
    
    # Since we have multiple problems per participant, average predictions
    all_y_test_combined = np.concatenate(all_y_test)
    all_y_pred_proba_combined = np.concatenate(all_y_pred_proba)
    
    print("Reproduction completed successfully!")
    
    # Verify reproducibility by comparing with original if available
    try:
        original_state = load_experiment_state('experiment_state.pkl')
        original_results = original_state['results']
        
        print("\n" + "="*50)
        print("REPRODUCIBILITY VERIFICATION")
        print("="*50)
        
        all_match = True
        for i, (orig, repro) in enumerate(zip(original_results, results)):
            orig_acc = orig['test_acc']
            repro_acc = repro['test_acc']
            orig_auc = orig['auc']
            repro_auc = repro['auc']
            
            acc_match = abs(orig_acc - repro_acc) < 1e-10
            auc_match = abs(orig_auc - repro_auc) < 1e-10
            
            print(f"Problem {orig['original_problem_num']}:")
            print(f"  Accuracy - Original: {orig_acc:.6f}, Reproduced: {repro_acc:.6f}, Match: {acc_match}")
            print(f"  AUC - Original: {orig_auc:.6f}, Reproduced: {repro_auc:.6f}, Match: {auc_match}")
            
            if not (acc_match and auc_match):
                all_match = False
        
        if all_match:
            print("\n✅ SUCCESS: All results match exactly!")
        else:
            print("\n❌ WARNING: Some results don't match exactly")
            
    except FileNotFoundError:
        print("\nNote: Original experiment_state.pkl not found for verification")
    
    print("\n" + "="*50)
    print("REPRODUCTION SUMMARY")
    print("="*50)
    print("1. Restored randomness state from randomness_state_before_training.pkl")
    print("2. Used identical data processing and model architecture")
    print("3. Used identical random seeds and split parameters")
    print("4. Saved reproduced results to experiment_state_reproduced.pkl")
    print("\nTo verify exact reproducibility, compare:")
    print("  - experiment_state.pkl (original)")
    print("  - experiment_state_reproduced.pkl (reproduced)")