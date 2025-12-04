import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import pandas as pd

# Set random seeds for reproducibility
np.random.seed(42)

# Load data
def load_data():
    # Load eye-tracking data
    with open('Pickle/SegmentsSeparatedOneByOne_AllPeople.pkl', 'rb') as f:
        eye_data = pickle.load(f)
    
    # Load parameter data
    with open('Pickle/ParData_AllPeople.pkl', 'rb') as f:
        par_data = pickle.load(f)
    
    return eye_data, par_data

# Constants
N_PARTICIPANTS = 19
N_EMOTIONAL_STATES = 2
N_PROBLEMS = 6  # Problems 8-13
N_PARAMETERS = 23
PROBLEM_NAMES = ['Problem 8', 'Problem 9', 'Problem 10', 'Problem 11', 'Problem 12', 'Problem 13']
EMOTIONAL_STATES = ['Negative', 'Positive']

def extract_first_pca_component(par_data):
    """
    Extract the first PCA component and compute averages for each problem
    """
    # Convert to numpy array if it's a list
    if isinstance(par_data, list):
        par_data = np.array(par_data)
    
    print(f"Original parameter data shape: {par_data.shape}")
    
    # Reshape parameter data: [participants, emotional_states, parameters, problems] -> [samples, parameters]
    n_samples = N_PARTICIPANTS * N_EMOTIONAL_STATES * N_PROBLEMS
    par_reshaped = par_data.transpose(0, 1, 3, 2).reshape(n_samples, N_PARAMETERS)
    
    print(f"Reshaped parameter data for PCA: {par_reshaped.shape}")
    
    # Standardize the data (important for PCA)
    scaler = StandardScaler()
    par_scaled = scaler.fit_transform(par_reshaped)
    
    # Apply PCA and keep only the first principal component
    pca = PCA(n_components=1)
    par_pca = pca.fit_transform(par_scaled)
    
    # Reshape back to [participants, emotional_states, problems]
    first_pc_values = par_pca.reshape(N_PARTICIPANTS, N_EMOTIONAL_STATES, N_PROBLEMS)
    
    print(f"\nFirst PCA component shape: {first_pc_values.shape}")
    print(f"Explained variance by first PC: {pca.explained_variance_ratio_[0]:.4f} ({pca.explained_variance_ratio_[0]:.2%})")
    
    return first_pc_values, pca, scaler

def compute_averages_and_visualize(first_pc_values):
    """
    Compute averages for each problem and create visualizations
    """
    # Compute various averages
    print("\n" + "="*60)
    print("FIRST PCA COMPONENT AVERAGES BY PROBLEM")
    print("="*60)
    
    # 1. Average across all participants and emotional states for each problem
    overall_means = np.mean(first_pc_values, axis=(0, 1))
    overall_stds = np.std(first_pc_values, axis=(0, 1))
    
    print("\nOverall averages (across all participants and emotional states):")
    for i, (mean, std) in enumerate(zip(overall_means, overall_stds)):
        print(f"{PROBLEM_NAMES[i]}: {mean:.4f} ± {std:.4f}")
    
    # 2. Average by emotional state for each problem
    negative_means = np.mean(first_pc_values[:, 0, :], axis=0)  # Negative emotional state
    positive_means = np.mean(first_pc_values[:, 1, :], axis=0)  # Positive emotional state
    
    print("\nAverages by emotional state:")
    print("Problem          Negative        Positive        Difference")
    print("-" * 55)
    for i in range(N_PROBLEMS):
        diff = positive_means[i] - negative_means[i]
        print(f"{PROBLEM_NAMES[i]:12} {negative_means[i]:12.4f} {positive_means[i]:12.4f} {diff:12.4f}")
    
    # 3. Average by participant (across all problems and emotional states)
    participant_means = np.mean(first_pc_values, axis=(1, 2))
    print(f"\nParticipant averages (across all problems and emotional states):")
    for i, mean in enumerate(participant_means):
        print(f"Participant {i+1:2d}: {mean:.4f}")
    
    # Create visualizations - FIXED: Pass overall_stds to the function
    create_average_visualizations(first_pc_values, overall_means, overall_stds, negative_means, positive_means, participant_means)
    
    return {
        'overall_means': overall_means,
        'overall_stds': overall_stds,
        'negative_means': negative_means,
        'positive_means': positive_means,
        'participant_means': participant_means,
        'first_pc_values': first_pc_values
    }

def create_average_visualizations(first_pc_values, overall_means, overall_stds, negative_means, positive_means, participant_means):
    """
    Create visualizations of the averaged first PCA component
    """
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create a figure with multiple subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Bar plot of overall averages by problem
    ax1 = axes[0, 0]
    x_pos = np.arange(len(PROBLEM_NAMES))
    bars = ax1.bar(x_pos, overall_means, yerr=overall_stds, 
                   capsize=5, alpha=0.7, color='skyblue', edgecolor='black')
    
    ax1.set_xlabel('Problems')
    ax1.set_ylabel('First PCA Component Value')
    ax1.set_title('Average First PCA Component by Problem\n(All Participants & Emotional States)')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(PROBLEM_NAMES, rotation=45)
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1 * np.sign(height),
                f'{height:.3f}', ha='center', va='bottom' if height > 0 else 'top', fontsize=9)
    
    # 2. Line plot comparing emotional states
    ax2 = axes[0, 1]
    x_pos = np.arange(len(PROBLEM_NAMES))
    ax2.plot(x_pos, negative_means, 'o-', linewidth=2, markersize=8, label='Negative', color='blue')
    ax2.plot(x_pos, positive_means, 's-', linewidth=2, markersize=8, label='Positive', color='red')
    
    ax2.set_xlabel('Problems')
    ax2.set_ylabel('First PCA Component Value')
    ax2.set_title('First PCA Component by Problem and Emotional State')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(PROBLEM_NAMES, rotation=45)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Heatmap of participant averages across problems
    ax3 = axes[1, 0]
    
    # Average across emotional states for heatmap
    participant_problem_means = np.mean(first_pc_values, axis=1)
    
    im = ax3.imshow(participant_problem_means, cmap='coolwarm', aspect='auto')
    ax3.set_xlabel('Problems')
    ax3.set_ylabel('Participants')
    ax3.set_title('First PCA Component - Participant Averages\n(Across Emotional States)')
    ax3.set_xticks(range(len(PROBLEM_NAMES)))
    ax3.set_xticklabels(PROBLEM_NAMES, rotation=45)
    ax3.set_yticks(range(N_PARTICIPANTS))
    ax3.set_yticklabels([f'P{i+1}' for i in range(N_PARTICIPANTS)])
    
    # Add colorbar
    plt.colorbar(im, ax=ax3, label='First PC Value')
    
    # 4. Box plot of first PC values by problem
    ax4 = axes[1, 1]
    
    # Prepare data for boxplot - flatten by problem
    box_data = []
    for problem_idx in range(N_PROBLEMS):
        problem_data = first_pc_values[:, :, problem_idx].flatten()
        box_data.append(problem_data)
    
    box_plot = ax4.boxplot(box_data, labels=PROBLEM_NAMES, patch_artist=True)
    
    # Color the boxes
    colors = plt.cm.Set3(np.linspace(0, 1, len(PROBLEM_NAMES)))
    for patch, color in zip(box_plot['boxes'], colors):
        patch.set_facecolor(color)
    
    ax4.set_xlabel('Problems')
    ax4.set_ylabel('First PCA Component Value')
    ax4.set_title('Distribution of First PCA Component by Problem')
    ax4.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.show()
    
    # Create additional detailed plots
    create_detailed_average_plots(first_pc_values, overall_means, overall_stds, negative_means, positive_means)

def create_detailed_average_plots(first_pc_values, overall_means, overall_stds, negative_means, positive_means):
    """
    Create additional detailed plots for averages
    """
    # Plot 1: Problem ranking by first PCA component
    plt.figure(figsize=(10, 6))
    
    # Sort problems by their average first PC value
    sorted_indices = np.argsort(overall_means)[::-1]  # Descending order
    sorted_means = overall_means[sorted_indices]
    sorted_names = [PROBLEM_NAMES[i] for i in sorted_indices]
    
    bars = plt.bar(range(len(sorted_means)), sorted_means, 
                   color=plt.cm.viridis(np.linspace(0, 1, len(sorted_means))))
    
    plt.xlabel('Problems (Ranked)')
    plt.ylabel('First PCA Component Value')
    plt.title('Problem Ranking by First PCA Component Value')
    plt.xticks(range(len(sorted_means)), sorted_names, rotation=45)
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01 * np.sign(height),
                f'{height:.3f}', ha='center', va='bottom' if height > 0 else 'top')
    
    plt.tight_layout()
    plt.show()
    
    # Plot 2: Emotional state differences by problem
    plt.figure(figsize=(10, 6))
    
    differences = positive_means - negative_means
    colors = ['green' if diff > 0 else 'red' for diff in differences]
    
    bars = plt.bar(PROBLEM_NAMES, differences, color=colors, alpha=0.7)
    
    plt.xlabel('Problems')
    plt.ylabel('Difference (Positive - Negative)')
    plt.title('Emotional State Differences in First PCA Component')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01 * np.sign(height),
                f'{height:.3f}', ha='center', va='bottom' if height > 0 else 'top')
    
    # Add horizontal line at zero
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Plot 3: Statistical summary table
    create_summary_table(overall_means, overall_stds, negative_means, positive_means, differences)

def create_summary_table(overall_means, overall_stds, negative_means, positive_means, differences):
    """
    Create a summary table of the averages
    """
    # Create a DataFrame for nice formatting
    summary_data = {
        'Problem': PROBLEM_NAMES,
        'Overall_Mean': overall_means,
        'Overall_Std': overall_stds,
        'Negative_Mean': negative_means,
        'Positive_Mean': positive_means,
        'Difference': differences
    }
    
    df = pd.DataFrame(summary_data)
    
    print("\n" + "="*80)
    print("SUMMARY TABLE OF FIRST PCA COMPONENT AVERAGES")
    print("="*80)
    print(df.round(4))
    
    # Additional statistics
    print(f"\nAdditional Statistics:")
    print(f"Overall mean across all problems: {np.mean(overall_means):.4f}")
    print(f"Overall standard deviation: {np.std(overall_means):.4f}")
    print(f"Range: {np.min(overall_means):.4f} to {np.max(overall_means):.4f}")
    print(f"Average emotional state difference: {np.mean(differences):.4f}")
    
    # Problems with largest emotional state differences
    largest_diff_idx = np.argmax(np.abs(differences))
    print(f"Problem with largest emotional state difference: {PROBLEM_NAMES[largest_diff_idx]} ({differences[largest_diff_idx]:.4f})")

# Main execution
if __name__ == "__main__":
    # Load data
    print("Loading data...")
    eye_data, par_data = load_data()
    
    print(f"Parameter data type: {type(par_data)}")
    if isinstance(par_data, list):
        print(f"Parameter data length: {len(par_data)}")
    elif hasattr(par_data, 'shape'):
        print(f"Parameter data shape: {par_data.shape}")
    
    # Extract first PCA component
    print("\nExtracting first PCA component...")
    first_pc_values, pca_model, scaler = extract_first_pca_component(par_data)
    
    # Compute and display averages
    averages = compute_averages_and_visualize(first_pc_values)
    
    # Save the results to files if needed
    print("\n" + "="*60)
    print("EXPORTING RESULTS")
    print("="*60)
    
    # Save first PC values to numpy file
    np.save('first_pca_component_values.npy', first_pc_values)
    print("First PCA component values saved to 'first_pca_component_values.npy'")
    
    # Save averages to CSV
    averages_df = pd.DataFrame({
        'Problem': PROBLEM_NAMES,
        'Overall_Mean': averages['overall_means'],
        'Overall_Std': averages['overall_stds'],
        'Negative_Mean': averages['negative_means'],
        'Positive_Mean': averages['positive_means'],
        'Difference': averages['positive_means'] - averages['negative_means']
    })
    averages_df.to_csv('pca_component_averages.csv', index=False)
    print("Averages saved to 'pca_component_averages.csv'")
    
    # Print final summary
    print(f"\nFINAL SUMMARY:")
    print(f"The first PCA component explains {pca_model.explained_variance_ratio_[0]:.2%} of the variance")
    print(f"Problems are ranked by their average first PC value:")
    
    sorted_indices = np.argsort(averages['overall_means'])[::-1]
    for i, idx in enumerate(sorted_indices):
        print(f"  {i+1}. {PROBLEM_NAMES[idx]}: {averages['overall_means'][idx]:.4f}")