import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score, StratifiedKFold, cross_val_predict
from pathlib import Path
from sklearn.preprocessing import LabelEncoder

def visualize_lda_decoding(npz_file, cv_folds=5, random_state=42, threshold=0.2):
    """
    Visualizes LDA decoding results while excluding units labeled as 'noise'
    and filtering out units with gOSI below a threshold.
    
    This function creates three figures:
      1. Overall CV accuracy and LDA projection with a discrete orientation legend.
      2. Per-orientation decoding accuracy.
      3. Comparison of decoding accuracy between real and shuffled data.
    
    The figure titles include the animal id and session id extracted from the NPZ file path.
    Figures are saved to the 'embedding' folder in the NPZ file's directory, with filenames 
    that include the filtering threshold.
    
    Parameters:
      npz_file (str or Path): Path to the NPZ file containing:
          - all_units_responses: neural responses, shape (n_units, n_ori, n_phase, n_sf, n_repeats)
          - unique_orientation: list/array of grating orientations
          - unit_qualities: list/array of quality labels (e.g., 'noise' or good unit)
          - gOSI: array of gOSI values (one per unit)
      cv_folds (int): Number of cross-validation folds.
      random_state (int): Random seed for reproducibility.
      threshold (float): gOSI threshold; only include units with gOSI > threshold.
    """
    npz_file = Path(npz_file)
    # Extract animal id and session id from npz_file path.
    # Assumes folder structure: .../sortout/{animal_id}/{session_id}/static_grating_tuning_metrics.npz
    session_folder = npz_file.parent
    animal_id = session_folder.parent.name
    session_id = session_folder.name

    # Load data from the NPZ file
    data = np.load(npz_file, allow_pickle=True)
    # Expected keys: 'all_units_responses', 'unique_orientation', 'unit_qualities', 'gOSI'
    all_units_responses = data['all_units_responses']  # shape: (n_units, n_ori, n_phase, n_sf, n_repeats)
    unique_orientation = data['unique_orientation']
    unit_qualities = np.array(data['unit_qualities'])
    metrics_file = npz_file.parent /"static_grating_tuning_metrics.npz"
    data = np.load(metrics_file, allow_pickle=True)
    # The file was saved with a key "all_shank_info" that is a dictionary.
    all_shank_info = data["all_shank_info"].item()  

    # Extract gOSI values from the dictionary.
    all_gOSI = []

    # Suppose all_shank_info is your dictionary, 
    # e.g. {'0': {'1': {'OSI':..., 'gOSI':...}, '2': {...}}, '1': {...}, ...}
    for shank_id, units_dict in all_shank_info.items():
        for unit_id, metrics_dict in units_dict.items():
            # Extract the gOSI value for this unit
            gOSI_value = metrics_dict['OSI']
            all_gOSI.append(gOSI_value)
    # Filter: remove units labeled as 'noise' and units with gOSI <= threshold.
    print(f"Total units before filtering: {all_units_responses.shape[0]}")
    #FIXME: gOSI size and unit_qualities size should be the same.
    all_gOSI = np.array(all_gOSI)
    valid_units_mask = (unit_qualities != 'noise') & (all_gOSI > threshold)
    all_units_responses = all_units_responses[valid_units_mask, ...]
    print(f"Total units after filtering (noise removed and gOSI > {threshold}): {all_units_responses.shape[0]}")

    n_units, n_ori, n_phase, n_sf, n_repeats = all_units_responses.shape
    n_trials = n_ori * n_phase * n_sf * n_repeats

    # Reshape responses into a 2D feature matrix X: each trial is one row and each unit is a feature.
    responses_transposed = np.transpose(all_units_responses, (1, 2, 3, 4, 0))
    X = responses_transposed.reshape(n_trials, n_units)

    # Create label vector y for grating orientations and encode them to discrete labels.
    y = np.repeat(unique_orientation, n_phase * n_sf * n_repeats)
    y = np.array(y)
    le = LabelEncoder()
    y_class = le.fit_transform(y)

    # Initialize LDA classifier and compute overall cross-validated decoding accuracy.
    clf = LinearDiscriminantAnalysis()
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    cv_scores = cross_val_score(clf, X, y_class, cv=skf)

    # Fit LDA on the entire dataset and project onto LDA components.
    clf.fit(X, y_class)
    X_lda = clf.transform(X)

    # ---- Figure 1: Overall CV accuracy and LDA projection ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot cross-validation scores.
    axes[0].bar(range(1, cv_folds + 1), cv_scores, color='skyblue', edgecolor='k')
    axes[0].set_xlabel("CV Fold")
    axes[0].set_ylabel("Decoding Accuracy")
    axes[0].set_title("LDA Cross-Validated Decoding Accuracy")
    axes[0].set_ylim([0, 1])
    
    # Scatter plot for LDA projection (using first two components).
    if X_lda.shape[1] >= 2:
        unique_orientations_encoded = np.unique(y_class)
        cmap = plt.get_cmap('tab10')
        for i, ori_enc in enumerate(unique_orientations_encoded):
            idx = np.where(y_class == ori_enc)[0]
            color = cmap(i / len(unique_orientations_encoded))
            ori_val = le.inverse_transform([ori_enc])[0]
            axes[1].scatter(
                X_lda[idx, 0],
                X_lda[idx, 1],
                color=color,
                label=f"{ori_val}°",
                alpha=0.7
            )
        axes[1].set_xlabel("LDA Component 1")
        axes[1].set_ylabel("LDA Component 2")
        axes[1].set_title("LDA Projection")
        axes[1].legend(title="Orientation", loc="best")
    else:
        axes[1].hist(X_lda[:, 0], bins=30, color='gray', alpha=0.7)
        axes[1].set_xlabel("LDA Component 1")
        axes[1].set_ylabel("Count")
        axes[1].set_title("LDA Projection Histogram")
    
    fig.suptitle(f"LDA Decoding for Animal: {animal_id}, Session: {session_id}", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save Figure 1 with threshold in the filename.
    embedding_folder = session_folder / "embedding"
    embedding_folder.mkdir(parents=True, exist_ok=True)
    save_path = embedding_folder / f"LDA_decoding_threshold_{threshold}.png"
    plt.savefig(save_path)
    plt.show()
    print(f"Overall decoding accuracy (mean ± std): {np.mean(cv_scores):.3f} ± {np.std(cv_scores):.3f}")
    print(f"Figure 1 saved to {save_path}")
    
    # ---- Figure 2: Per-orientation decoding accuracy ----
    y_pred = cross_val_predict(clf, X, y_class, cv=skf)
    unique_orientations_encoded = np.unique(y_class)
    per_orientation_accuracy = []
    ori_labels = []
    for ori_enc in unique_orientations_encoded:
        idx = np.where(y_class == ori_enc)[0]
        acc = np.mean(y_pred[idx] == y_class[idx])
        per_orientation_accuracy.append(acc)
        ori_val = le.inverse_transform([ori_enc])[0]
        ori_labels.append(ori_val)
    
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    ax2.bar(ori_labels, per_orientation_accuracy, color='lightgreen', edgecolor='k')
    ax2.set_xlabel("Orientation (°)")
    ax2.set_ylabel("Decoding Accuracy")
    ax2.set_title(f"Per-Orientation Decoding Accuracy\nAnimal: {animal_id}, Session: {session_id}")
    ax2.set_ylim([0, 1])
    plt.tight_layout()
    
    save_path2 = embedding_folder / f"LDA_per_orientation_accuracy_threshold_{threshold}.png"
    plt.savefig(save_path2)
    plt.show()
    print(f"Figure 2 saved to {save_path2}")
    
    # ---- Figure 3: Compare real vs. shuffled data ----
    X_shuffled = X[np.random.permutation(X.shape[0]), :]
    cv_scores_shuffled = cross_val_score(clf, X_shuffled, y_class, cv=skf)
    
    mean_real = np.mean(cv_scores)
    std_real = np.std(cv_scores)
    mean_shuffled = np.mean(cv_scores_shuffled)
    std_shuffled = np.std(cv_scores_shuffled)
    
    fig3, ax3 = plt.subplots(figsize=(8, 6))
    ax3.bar(['Real', 'Shuffled'], [mean_real, mean_shuffled],
            yerr=[std_real, std_shuffled], color=['dodgerblue', 'salmon'],
            edgecolor='k', capsize=8)
    ax3.set_ylabel("Decoding Accuracy")
    ax3.set_title(f"Real vs Shuffled Decoding Accuracy\nAnimal: {animal_id}, Session: {session_id}")
    ax3.set_ylim([0, 1])
    plt.tight_layout()
    
    save_path3 = embedding_folder / f"LDA_real_vs_shuffled_accuracy_threshold_{threshold}.png"
    plt.savefig(save_path3)
    plt.show()
    print(f"Real vs Shuffled accuracy: Real = {mean_real:.3f} ± {std_real:.3f}, Shuffled = {mean_shuffled:.3f} ± {std_shuffled:.3f}")
    print(f"Figure 3 saved to {save_path3}")
    
    return cv_scores

if __name__ == '__main__':
    npz_file_path = '/Volumes/xieluanlabs/xl_cl/code/sortout/CnL22/250314_174049/static_grating_responses.npz'
    visualize_lda_decoding(npz_file_path)