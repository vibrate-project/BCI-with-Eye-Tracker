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
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight

# Set random seeds for maximum reproducibility
np.random.seed(42)
tf.random.set_seed(42)
tf.keras.utils.set_random_seed(42)

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

# Enhanced data augmentation functions
def augment_time_series(series, max_shift=20, noise_factor=0.005, scaling_factor=0.1):
    """Enhanced augmentation for time series data"""
    augmented = series.copy()
    
    # Random time warping (small shifts)
    shift = np.random.randint(-max_shift, max_shift)
    if shift > 0:
        augmented = np.concatenate([augmented[shift:], np.zeros((shift, augmented.shape[1]))])
    elif shift < 0:
        augmented = np.concatenate([np.zeros((-shift, augmented.shape[1])), augmented[:shift]])
    
    # Random scaling
    scale = 1 + np.random.uniform(-scaling_factor, scaling_factor)
    augmented *= scale
    
    # Add Gaussian noise
    noise = np.random.normal(0, noise_factor, augmented.shape)
    augmented += noise
    
    return augmented

def augment_dataset(X, y, groups, augment_factor=2):
    """Augment dataset while preserving group structure"""
    X_augmented = []
    y_augmented = []
    groups_augmented = []
    
    for i in range(len(X)):
        X_augmented.append(X[i])
        y_augmented.append(y[i])
        groups_augmented.append(groups[i])
        
        for _ in range(augment_factor):
            X_augmented.append(augment_time_series(X[i]))
            y_augmented.append(y[i])
            groups_augmented.append(groups[i])  # Same group for augmented samples
    
    return np.array(X_augmented), np.array(y_augmented), np.array(groups_augmented)

# CORRECTED Enhanced multimodal model with regularization
def create_enhanced_multimodal_model(eye_input_shape, par_input_shape, dropout_rate=0.4):
    """Create enhanced model with better regularization - CORRECTED VERSION"""
    
    # Eye-tracking data branch - Enhanced architecture
    eye_input = layers.Input(shape=eye_input_shape, name='eye_input')
    x = layers.Masking(mask_value=0.0)(eye_input)
    
    # Multiple Conv1D layers with batch normalization
    x = layers.Conv1D(64, 7, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(dropout_rate)(x)
    
    x = layers.Conv1D(128, 5, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(dropout_rate)(x)
    
    x = layers.Conv1D(256, 3, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling1D()(x)  # Use GlobalAveragePooling instead of more MaxPooling
    x = layers.Dropout(dropout_rate)(x)
    
    # Additional dense layers for eye branch
    eye_branch_output = layers.Dense(128, activation='relu')(x)
    eye_branch_output = layers.BatchNormalization()(eye_branch_output)
    eye_branch_output = layers.Dropout(dropout_rate)(eye_branch_output)
    
    eye_branch_output = layers.Dense(64, activation='relu')(eye_branch_output)
    eye_branch_output = layers.Dropout(dropout_rate)(eye_branch_output)
    
    # Parameter data branch
    par_input = layers.Input(shape=par_input_shape, name='par_input')
    y = layers.Dense(64, activation='relu')(par_input)
    y = layers.BatchNormalization()(y)
    y = layers.Dropout(dropout_rate)(y)
    
    y = layers.Dense(32, activation='relu')(y)
    y = layers.BatchNormalization()(y)
    y = layers.Dropout(dropout_rate)(y)
    
    # Combine branches - FIXED: Use tensors directly, not .output
    combined = layers.concatenate([eye_branch_output, y])
    
    # Deep combined layers
    z = layers.Dense(128, activation='relu')(combined)
    z = layers.BatchNormalization()(z)
    z = layers.Dropout(dropout_rate)(z)
    
    z = layers.Dense(64, activation='relu')(z)
    z = layers.BatchNormalization()(z)
    z = layers.Dropout(dropout_rate)(z)
    
    z = layers.Dense(32, activation='relu')(z)
    z = layers.Dropout(dropout_rate)(z)
    
    # Output layer
    output = layers.Dense(1, activation='sigmoid', kernel_regularizer=tf.keras.regularizers.l2(0.001))(z)
    
    model = models.Model(inputs=[eye_input, par_input], outputs=output)
    
    # Custom learning rate
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)
    model.compile(optimizer=optimizer, 
                  loss='binary_crossentropy', 
                  metrics=['accuracy', 
                          tf.keras.metrics.AUC(name='auc'),
                          tf.keras.metrics.Precision(name='precision'),
                          tf.keras.metrics.Recall(name='recall')])
    
    return model

# Prepare data for training - Only problems 8-13
def prepare_data(eye_data, par_data):
    # Reshape eye data to [participants, emotional_state, problems, time, coordinates]
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
    
    # Normalize eye data per participant and problem
    for participant_idx in range(N_PARTICIPANTS):
        for problem_idx in range(N_PROBLEMS):
            problem_data = eye_data_reshaped[participant_idx, :, problem_idx, :, :]
            if np.std(problem_data) > 0:  # Avoid division by zero
                eye_data_reshaped[participant_idx, :, problem_idx, :, :] = (
                    (problem_data - np.mean(problem_data)) / np.std(problem_data)
                )
    
    # Prepare parameter data - Only problems 8-13
    if isinstance(par_data, list):
        par_data = np.array(par_data)
    
    # Check the shape of parameter data and handle accordingly
    print(f"Parameter data shape: {par_data.shape}")
    
    # Normalize parameter data per parameter across all participants and states
    par_data_normalized = np.zeros_like(par_data)
    for param_idx in range(par_data.shape[2]):  # Use actual number of parameters
        param_data = par_data[:, :, param_idx, :]
        mean = np.mean(param_data)
        std = np.std(param_data)
        if std > 0:
            par_data_normalized[:, :, param_idx, :] = (param_data - mean) / std
    
    return eye_data_reshaped, par_data_normalized

# Enhanced training with proper cross-validation
def train_and_evaluate_enhanced(eye_data, par_data, n_splits=5):
    # Prepare data
    eye_data_prepared, par_data_prepared = prepare_data(eye_data, par_data)
    
    # Store results
    all_results = []
    
    # For each problem in 8-13, create and train a model
    for problem_idx in range(N_PROBLEMS):
        original_problem_num = PROBLEM_INDICES[problem_idx] + 1
        print(f"\n{'='*60}")
        print(f"Training model for Problem {original_problem_num}")
        print(f"{'='*60}")
        
        # Prepare problem-specific data
        X_eye = []
        X_par = []
        y = []
        groups = []  # Participant groups for cross-validation
        
        for participant in range(N_PARTICIPANTS):
            for state in range(N_EMOTIONAL_STATES):
                # Eye data for this problem
                eye_sample = eye_data_prepared[participant, state, problem_idx, :, :]
                X_eye.append(eye_sample)
                
                # Parameter data - handle different possible shapes
                if par_data_prepared.ndim == 4:
                    par_sample = par_data_prepared[participant, state, :, problem_idx]
                else:
                    # Fallback for different data structure
                    par_sample = par_data_prepared[participant, state, :]
                
                X_par.append(par_sample)
                
                y.append(state)
                groups.append(participant)  # Use participant as group
        
        X_eye = np.array(X_eye)
        X_par = np.array(X_par)
        y = np.array(y)
        groups = np.array(groups)
        
        print(f"Data shapes - X_eye: {X_eye.shape}, X_par: {X_par.shape}, y: {y.shape}")
        
        # Use Group K-Fold to avoid data leakage
        group_kfold = GroupKFold(n_splits=min(n_splits, len(np.unique(groups))))
        fold_results = []
        
        for fold, (train_idx, test_idx) in enumerate(group_kfold.split(X_eye, y, groups)):
            print(f"\n--- Fold {fold+1}/{n_splits} ---")
            
            # Split data
            X_eye_train, X_eye_test = X_eye[train_idx], X_eye[test_idx]
            X_par_train, X_par_test = X_par[train_idx], X_par[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            groups_train = groups[train_idx]
            
            # Augment training data (only training set!)
            X_eye_train_aug, y_train_aug, groups_train_aug = augment_dataset(
                X_eye_train, y_train, groups_train, augment_factor=2)
            X_par_train_aug = np.repeat(X_par_train, 3, axis=0)  # Match augmentation factor
            
            # Compute class weights for imbalanced data
            class_weights = compute_class_weight('balanced', classes=np.unique(y_train_aug), y=y_train_aug)
            class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}
            
            # Create enhanced model
            model = create_enhanced_multimodal_model((MAX_LENGTH, 2), (X_par.shape[1],))
            
            # Enhanced callbacks
            callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss', patience=20, restore_best_weights=True, verbose=1
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss', factor=0.5, patience=10, min_lr=1e-7, verbose=1
                )
            ]
            
            # Train with validation split
            history = model.fit(
                [X_eye_train_aug, X_par_train_aug], y_train_aug,
                epochs=150, batch_size=8, validation_split=0.2,
                callbacks=callbacks, class_weight=class_weight_dict, verbose=1
            )
            
            # Evaluate
            test_results = model.evaluate([X_eye_test, X_par_test], y_test, verbose=0)
            y_pred_proba = model.predict([X_eye_test, X_par_test], verbose=0).flatten()
            y_pred = (y_pred_proba > 0.5).astype(int)
            
            # Calculate metrics
            auc = roc_auc_score(y_test, y_pred_proba)
            
            fold_results.append({
                'fold': fold,
                'test_accuracy': test_results[1],
                'test_auc': auc,
                'test_loss': test_results[0],
                'y_test': y_test,
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba,
                'model': model
            })
            
            print(f"Fold {fold+1} - Accuracy: {test_results[1]:.4f}, AUC: {auc:.4f}")
        
        # Aggregate fold results
        avg_accuracy = np.mean([r['test_accuracy'] for r in fold_results])
        avg_auc = np.mean([r['test_auc'] for r in fold_results])
        std_accuracy = np.std([r['test_accuracy'] for r in fold_results])
        std_auc = np.std([r['test_auc'] for r in fold_results])
        
        print(f"\nProblem {original_problem_num} - Cross-validation Results:")
        print(f"Average Accuracy: {avg_accuracy:.4f} (±{std_accuracy:.4f})")
        print(f"Average AUC: {avg_auc:.4f} (±{std_auc:.4f})")
        
        problem_results = {
            'problem_idx': problem_idx,
            'original_problem_num': original_problem_num,
            'fold_results': fold_results,
            'avg_accuracy': avg_accuracy,
            'avg_auc': avg_auc,
            'std_accuracy': std_accuracy,
            'std_auc': std_auc
        }
        
        all_results.append(problem_results)
    
    return all_results

# Comprehensive evaluation
def comprehensive_evaluation(results):
    print("\n" + "="*80)
    print("COMPREHENSIVE EVALUATION RESULTS FOR PROBLEMS 8-13")
    print("="*80)
    
    # Individual problem results
    print("\n--- Individual Problem Performance ---")
    problem_accuracies = []
    problem_aucs = []
    
    for res in results:
        print(f"\nProblem {res['original_problem_num']}:")
        print(f"  Average Accuracy: {res['avg_accuracy']:.4f} (±{res['std_accuracy']:.4f})")
        print(f"  Average AUC: {res['avg_auc']:.4f} (±{res['std_auc']:.4f})")
        
        # Detailed fold results
        for fold_res in res['fold_results']:
            print(f"    Fold {fold_res['fold']+1}: Acc={fold_res['test_accuracy']:.4f}, "
                  f"AUC={fold_res['test_auc']:.4f}")
        
        problem_accuracies.append(res['avg_accuracy'])
        problem_aucs.append(res['avg_auc'])
    
    # Overall performance
    print(f"\n--- Overall Performance Across All Problems ---")
    print(f"Mean Accuracy: {np.mean(problem_accuracies):.4f} (±{np.std(problem_accuracies):.4f})")
    print(f"Mean AUC: {np.mean(problem_aucs):.4f} (±{np.std(problem_aucs):.4f})")
    
    # Statistical analysis
    print(f"\n--- Statistical Analysis ---")
    print(f"Accuracy Range: {np.min(problem_accuracies):.4f} - {np.max(problem_accuracies):.4f}")
    print(f"AUC Range: {np.min(problem_aucs):.4f} - {np.max(problem_aucs):.4f}")
    
    return problem_accuracies, problem_aucs

# Main execution
if __name__ == "__main__":
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
        # Convert to numpy array for easier handling
        par_data = np.array(par_data)
        print(f"Parameter data shape after conversion: {par_data.shape}")
    
    # Train and evaluate models with enhanced methodology
    print(f"\nTraining enhanced multimodal models for Problems 8-13...")
    results = train_and_evaluate_enhanced(eye_data, par_data, n_splits=5)
    
    # Comprehensive evaluation
    problem_accuracies, problem_aucs = comprehensive_evaluation(results)
    
    # Final summary
    print(f"\n{'>'*50} FINAL SUMMARY {'<'*50}")
    print(f"Overall Performance: {np.mean(problem_accuracies):.4f} accuracy, {np.mean(problem_aucs):.4f} AUC")
    print(f"Consistency: ±{np.std(problem_accuracies):.4f} accuracy, ±{np.std(problem_aucs):.4f} AUC")