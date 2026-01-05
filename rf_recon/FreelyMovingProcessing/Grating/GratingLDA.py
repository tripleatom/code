"""
Simplified LDA Analysis for Grating Orientation Neural Data
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle
import h5py
import json
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_validate
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# DATA LOADING
# =============================================================================

def load_neural_data(filepath):
    """Load neural data from pickle, HDF5, or NPZ format."""
    filepath = Path(filepath)
    loaders = {
        '.pkl': _load_pickle,
        '.h5': _load_hdf5,
        '.npz': _load_npz
    }
    
    loader = loaders.get(filepath.suffix)
    if not loader:
        raise ValueError(f"Unsupported file format: {filepath.suffix}")
    
    print(f"Loading data from {filepath.suffix}: {filepath}")
    return loader(filepath)


def _load_pickle(filepath):
    """Load from pickle format."""
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def _load_hdf5(filepath):
    """Load from HDF5 format."""
    data = {
        'metadata': {},
        'experiment_parameters': {},
        'trial_info': {},
        'spike_data': {},
        'unit_info': {},
        'extraction_params': {}
    }
    
    with h5py.File(filepath, 'r') as f:
        # Helper to load group
        def load_group(group, target_dict):
            for key in group.attrs.keys():
                target_dict[key] = group.attrs[key]
            for key in group.keys():
                target_dict[key] = group[key][()]
        
        # Load metadata and parameters
        if 'metadata' in f:
            load_group(f['metadata'], data['metadata'])
        if 'experiment_parameters' in f:
            load_group(f['experiment_parameters'], data['experiment_parameters'])
        if 'extraction_params' in f:
            load_group(f['extraction_params'], data['extraction_params'])
        
        # Load trial info
        if 'trial_info' in f:
            trial_grp = f['trial_info']
            for key in ['orientations', 'unique_orientations', 'trial_windows']:
                if key in trial_grp:
                    data['trial_info'][key] = trial_grp[key][()].tolist()
            if 'all_trial_parameters' in trial_grp.attrs:
                data['trial_info']['all_trial_parameters'] = json.loads(
                    trial_grp.attrs['all_trial_parameters']
                )
        
        # Load spike data
        if 'spike_data' in f:
            for unit_id in f['spike_data'].keys():
                unit_grp = f['spike_data'][unit_id]
                trials_data = []
                
                for trial_key in sorted([k for k in unit_grp.keys() if k.startswith('trial_')]):
                    tgrp = unit_grp[trial_key]
                    trials_data.append({
                        'trial_index': tgrp.attrs['trial_index'],
                        'orientation': tgrp.attrs['orientation'] if tgrp.attrs['orientation'] != -999 else None,
                        'spike_count': tgrp.attrs['spike_count'],
                        'trial_start': tgrp.attrs['trial_start'],
                        'trial_end': tgrp.attrs['trial_end'],
                        'spike_times': tgrp['spike_times'][()].tolist()
                    })
                
                data['spike_data'][unit_id] = trials_data
        
        # Load unit info
        if 'unit_info' in f:
            for unit_id in f['unit_info'].keys():
                data['unit_info'][unit_id] = {}
                load_group(f['unit_info'][unit_id], data['unit_info'][unit_id])
    
    return data


def _load_npz(filepath):
    """Load from NPZ format (with companion pickle for complex data)."""
    data_npz = np.load(filepath, allow_pickle=True)
    pickle_path = filepath.with_suffix('.complex.pkl')
    complex_data = _load_pickle(pickle_path) if pickle_path.exists() else {}
    
    data = {
        'metadata': {},
        'experiment_parameters': {},
        'trial_info': {},
        'extraction_params': {},
        'spike_data': complex_data.get('spike_data', {}),
        'unit_info': complex_data.get('unit_info', {})
    }
    
    # Parse NPZ keys
    key_mapping = {
        'metadata_': ('metadata', 9),
        'exp_': ('experiment_parameters', 4),
        'params_': ('extraction_params', 7)
    }
    
    for key, value in data_npz.items():
        for prefix, (target, offset) in key_mapping.items():
            if key.startswith(prefix):
                data[target][key[offset:]] = value
                break
        
        # Handle trial info
        if key == 'trial_orientations':
            data['trial_info']['orientations'] = value.tolist()
        elif key == 'trial_unique_orientations':
            data['trial_info']['unique_orientations'] = value.tolist()
        elif key == 'trial_windows':
            data['trial_info']['trial_windows'] = value.tolist()
    
    if 'trial_parameters' in complex_data:
        data['trial_info']['all_trial_parameters'] = complex_data['trial_parameters']
    
    return data


# =============================================================================
# ANALYSIS
# =============================================================================

def calculate_firing_rates(neural_data, time_window=(0.07, 0.16)):
    """
    Calculate firing rates for each unit in each trial.
    
    Returns:
        firing_rates: (n_trials, n_units) array
        orientation_labels: (n_trials,) array
        unit_ids: list of unit identifiers
        trial_info: dict with experiment metadata
    """
    window_start, window_end = time_window
    window_duration = window_end - window_start
    
    unit_ids = list(neural_data['spike_data'].keys())
    orientations = neural_data['trial_info']['orientations']
    unique_orientations = neural_data['trial_info']['unique_orientations']
    n_trials, n_units = len(orientations), len(unit_ids)
    
    # Print summary
    print(f"\nCalculating firing rates:")
    print(f"  Units: {n_units} | Trials: {n_trials}")
    print(f"  Window: {window_start:.3f}-{window_end:.3f}s ({window_duration:.3f}s)")
    print(f"  Orientations: {unique_orientations}")
    
    for ori in unique_orientations:
        print(f"    {ori}°: {orientations.count(ori)} trials")
    
    # Calculate firing rates
    firing_rates = np.full((n_trials, n_units), np.nan)
    
    for unit_idx, unit_id in enumerate(unit_ids):
        for trial_data in neural_data['spike_data'][unit_id]:
            trial_idx = int(trial_data['trial_index'])
            if 0 <= trial_idx < n_trials:
                spike_times = np.array(trial_data['spike_times'])
                spikes_in_window = np.sum((spike_times >= window_start) & 
                                         (spike_times < window_end))
                firing_rates[trial_idx, unit_idx] = spikes_in_window / window_duration
    
    # Remove trials with missing data
    valid_mask = ~np.isnan(firing_rates).any(axis=1)
    firing_rates_clean = firing_rates[valid_mask]
    orientation_labels = np.array(orientations)[valid_mask]
    
    print(f"  Valid trials: {np.sum(valid_mask)}/{n_trials}")
    print(f"  Mean firing rate: {np.mean(firing_rates_clean):.2f} Hz")
    
    trial_info = {
        'valid_trials_mask': valid_mask,
        'unique_orientations': unique_orientations,
        'experiment_parameters': neural_data.get('experiment_parameters', {}),
        'n_trials_per_orientation': {
            ori: int(np.sum(orientation_labels == ori))
            for ori in unique_orientations
        }
    }
    
    return firing_rates_clean, orientation_labels, unit_ids, trial_info


def perform_lda_analysis(firing_rates, orientation_labels, n_components=None):
    """
    Perform LDA analysis with cross-validation.
    
    Returns:
        Dictionary with LDA results including:
        - transformed_data, predictions, cv_scores, confusion_matrix, etc.
    """
    unique_orientations = np.unique(orientation_labels)
    n_orientations = len(unique_orientations)
    n_features = firing_rates.shape[1]
    
    print(f"\nLDA Analysis:")
    print(f"  Orientations: {n_orientations} ({unique_orientations}°)")
    print(f"  Features (units): {n_features}")
    print(f"  Trials: {len(orientation_labels)}")
    
    # Determine components
    min_trials = min(np.sum(orientation_labels == ori) for ori in unique_orientations)
    max_components = min(n_orientations - 1, n_features)
    n_components = min(n_components or 3, max_components)
    
    print(f"  Min trials per class: {int(min_trials)}")
    print(f"  LDA components: {n_components}")
    
    # Standardize features
    scaler = StandardScaler()
    firing_rates_scaled = scaler.fit_transform(firing_rates)
    
    # Fit LDA for dimensionality reduction
    lda = LinearDiscriminantAnalysis(n_components=n_components)
    lda_transformed = lda.fit_transform(firing_rates_scaled, orientation_labels)
    
    # Cross-validation
    cv_folds = max(2, min(5, int(min_trials)))
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    
    lda_full = LinearDiscriminantAnalysis()
    cv_scores = cross_val_score(lda_full, firing_rates_scaled, orientation_labels,
                                cv=cv, scoring='accuracy')
    cv_results = cross_validate(lda_full, firing_rates_scaled, orientation_labels,
                                cv=cv, scoring=['accuracy', 'f1_macro'],
                                return_train_score=True, return_estimator=True)
    
    # Full model predictions
    lda_full.fit(firing_rates_scaled, orientation_labels)
    predictions = lda_full.predict(firing_rates_scaled)
    prediction_proba = lda_full.predict_proba(firing_rates_scaled)
    
    # Performance metrics
    conf_matrix = confusion_matrix(orientation_labels, predictions, 
                                   labels=unique_orientations)
    orientation_accuracies = conf_matrix.diagonal() / conf_matrix.sum(axis=1)
    chance_accuracy = 1.0 / n_orientations
    overall_accuracy = accuracy_score(orientation_labels, predictions)
    
    print(f"  CV accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    print(f"  Overall accuracy: {overall_accuracy:.3f}")
    print(f"  Chance level: {chance_accuracy:.3f}")
    print(f"  Above chance: {overall_accuracy - chance_accuracy:+.3f}")
    
    return {
        'lda_model': lda,
        'lda_full': lda_full,
        'scaler': scaler,
        'transformed_data': lda_transformed,
        'original_data': firing_rates_scaled,
        'orientation_labels': orientation_labels,
        'predictions': predictions,
        'prediction_proba': prediction_proba,
        'cv_scores': cv_scores,
        'cv_results': cv_results,
        'confusion_matrix': conf_matrix,
        'orientation_accuracies': orientation_accuracies,
        'unique_orientations': unique_orientations,
        'n_components': n_components,
        'chance_accuracy': chance_accuracy,
        'explained_variance_ratio': getattr(lda, 'explained_variance_ratio_', None)
    }


def calculate_orientation_selectivity(unit_ids, orientation_labels, firing_rates):
    """
    Calculate orientation selectivity index (OSI) using vector sum method.
    OSI = |Σ r(θ)·exp(i·2θ)| / Σ r(θ)
    
    Returns:
        Dictionary with unit_ids, osi, and preferred_orientation_deg
    """
    orientations = np.unique(orientation_labels)
    theta_rad = 2 * np.deg2rad(orientations)  # Double angle for orientation
    
    # Calculate mean firing rate per orientation
    mean_rates = np.array([
        firing_rates[orientation_labels == ori].mean(axis=0)
        for ori in orientations
    ])  # Shape: (n_orientations, n_units)
    
    # Vector sum
    complex_exp = np.exp(1j * theta_rad)[:, None]
    vector_sum = (mean_rates * complex_exp).sum(axis=0)
    
    osi = np.abs(vector_sum) / (mean_rates.sum(axis=0) + 1e-12)
    pref_orientation_deg = (np.angle(vector_sum) / 2.0) % np.pi
    pref_orientation_deg = np.rad2deg(pref_orientation_deg)
    
    print(f"\nOrientation Selectivity:")
    print(f"  Mean OSI: {osi.mean():.3f}")
    print(f"  Top units:")
    
    for idx in np.argsort(osi)[::-1][:min(10, len(unit_ids))]:
        print(f"    {unit_ids[idx]}: OSI={osi[idx]:.3f}, Pref={pref_orientation_deg[idx]:.1f}°")
    
    return {
        'unit_ids': unit_ids,
        'osi': osi,
        'preferred_orientation_deg': pref_orientation_deg
    }


# =============================================================================
# VISUALIZATION
# =============================================================================

def create_analysis_figure(results, unit_ids, trial_info, save_path=None):
    """Create comprehensive LDA analysis visualization."""
    plt.style.use('default')
    sns.set_palette("husl")
    
    fig = plt.figure(figsize=(20, 16))
    
    # Extract commonly used data
    transformed = results['transformed_data']
    labels = results['orientation_labels']
    unique_ori = results['unique_orientations']
    n_comp = results['n_components']
    colors = plt.cm.hsv(np.linspace(0, 1, len(unique_ori) + 1)[:-1])
    
    # Create subplots
    _plot_3d_scatter(fig, transformed, labels, unique_ori, colors, n_comp)
    _plot_2d_scatter(fig, transformed, labels, unique_ori, colors, n_comp)
    _plot_confusion_matrix(fig, results['confusion_matrix'], unique_ori)
    _plot_cv_scores(fig, results)
    _plot_per_orientation_accuracy(fig, results, unique_ori, colors)
    _plot_polar_accuracy(fig, results, unique_ori)
    _plot_lda_coefficients(fig, results, unit_ids, unique_ori)
    _plot_trial_distribution(fig, trial_info, unique_ori, colors)
    _plot_summary_text(fig, results, labels, unit_ids, trial_info)
    _plot_feature_importance(fig, results, unit_ids)
    _plot_prediction_confidence(fig, results, labels)
    
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"Saved figure to: {save_path}")
    
    return fig


def _plot_3d_scatter(fig, data, labels, orientations, colors, n_comp):
    """Plot 3D LDA scatter."""
    ax = fig.add_subplot(3, 4, 1, projection='3d')
    
    if n_comp >= 3:
        for i, ori in enumerate(orientations):
            mask = labels == ori
            ax.scatter(data[mask, 0], data[mask, 1], data[mask, 2],
                      c=[colors[i]], label=f'{ori}°', alpha=0.7, s=30)
        ax.set_xlabel('LD1')
        ax.set_ylabel('LD2')
        ax.set_zlabel('LD3')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    else:
        ax.text(0.5, 0.5, 0.5, 'Need ≥3 components\nfor 3D visualization',
                ha='center', va='center', transform=ax.transAxes)
    
    ax.set_title('LDA 3D Projection', fontsize=14, fontweight='bold')


def _plot_2d_scatter(fig, data, labels, orientations, colors, n_comp):
    """Plot 2D LDA scatter or 1D jitter."""
    ax = fig.add_subplot(3, 4, 2)
    
    if n_comp >= 2:
        for i, ori in enumerate(orientations):
            mask = labels == ori
            ax.scatter(data[mask, 0], data[mask, 1],
                      c=[colors[i]], label=f'{ori}°', alpha=0.7, s=30)
        ax.set_xlabel('LD1')
        ax.set_ylabel('LD2')
        ax.set_title('LDA 2D Projection', fontsize=14, fontweight='bold')
    else:
        for i, ori in enumerate(orientations):
            mask = labels == ori
            y_jitter = np.random.normal(0, 0.1, np.sum(mask))
            ax.scatter(data[mask, 0], y_jitter,
                      c=[colors[i]], label=f'{ori}°', alpha=0.7, s=30)
        ax.set_xlabel('LD1')
        ax.set_ylabel('Random jitter')
        ax.set_title('LDA 1D Projection', fontsize=14, fontweight='bold')
    
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


def _plot_confusion_matrix(fig, conf_matrix, orientations):
    """Plot confusion matrix."""
    ax = fig.add_subplot(3, 4, 3)
    
    im = ax.imshow(conf_matrix, interpolation='nearest', cmap='Blues')
    fig.colorbar(im, ax=ax)
    
    labels = [f'{ori}°' for ori in orientations]
    ax.set(xticks=np.arange(len(labels)), yticks=np.arange(len(labels)),
           xticklabels=labels, yticklabels=labels,
           title='Confusion Matrix', ylabel='True', xlabel='Predicted')
    
    # Add text annotations
    thresh = conf_matrix.max() / 2
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            color = 'white' if conf_matrix[i, j] > thresh else 'black'
            ax.text(j, i, int(conf_matrix[i, j]), ha='center', va='center',
                   color=color, fontsize=10)
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')


def _plot_cv_scores(fig, results):
    """Plot cross-validation scores."""
    ax = fig.add_subplot(3, 4, 4)
    
    scores = results['cv_scores']
    bars = ax.bar(range(len(scores)), scores, alpha=0.7, color='skyblue')
    ax.axhline(scores.mean(), color='red', linestyle='--', 
               label=f'Mean: {scores.mean():.3f}')
    ax.axhline(results['chance_accuracy'], color='gray', linestyle=':', 
               label=f'Chance: {results["chance_accuracy"]:.3f}')
    
    ax.set(xlabel='CV Fold', ylabel='Accuracy', ylim=[0, 1],
           title='Cross-Validation Scores')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Add value labels
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.01, f'{h:.3f}',
                ha='center', va='bottom', fontsize=8)


def _plot_per_orientation_accuracy(fig, results, orientations, colors):
    """Plot per-orientation classification accuracy."""
    ax = fig.add_subplot(3, 4, 5)
    
    accuracies = results['orientation_accuracies']
    bars = ax.bar(range(len(accuracies)), accuracies, color=colors, alpha=0.7)
    ax.axhline(results['chance_accuracy'], color='gray', linestyle=':', 
               label='Chance')
    
    ax.set(xlabel='Orientation', ylabel='Accuracy', ylim=[0, 1],
           title='Per-Orientation Accuracy',
           xticks=range(len(orientations)),
           xticklabels=[f'{ori}°' for ori in orientations])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.01, f'{h:.3f}',
                ha='center', va='bottom', fontsize=8)
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')


def _plot_polar_accuracy(fig, results, orientations):
    """Plot decoding accuracy in polar coordinates."""
    ax = fig.add_subplot(3, 4, 6, projection='polar')
    
    theta = 2 * np.deg2rad(orientations)  # Double for orientation
    accuracies = results['orientation_accuracies']
    
    ax.plot(theta, accuracies, 'o-', linewidth=2, markersize=8)
    ax.fill(theta, accuracies, alpha=0.25)
    ax.set(ylim=[0, 1], title='Polar Decoding Accuracy')
    ax.set_thetagrids(np.arange(0, 360, 45), 
                      [f'{int(a/2)}°' for a in np.arange(0, 360, 45)])
    ax.grid(True)


def _plot_lda_coefficients(fig, results, unit_ids, orientations):
    """Plot LDA coefficient heatmap."""
    ax = fig.add_subplot(3, 4, (7, 8))
    
    if hasattr(results['lda_full'], 'coef_'):
        im = ax.imshow(results['lda_full'].coef_, cmap='RdBu_r', aspect='auto')
        fig.colorbar(im, ax=ax)
        
        ax.set(ylabel='Discriminant', xlabel='Units',
               title='LDA Coefficients',
               yticks=range(len(orientations)),
               yticklabels=[f'{ori}°' for ori in orientations])
        
        if len(unit_ids) <= 20:
            ax.set_xticks(range(len(unit_ids)))
            ax.set_xticklabels([uid.split('_')[-1] for uid in unit_ids], 
                              rotation=45, ha='right')


def _plot_trial_distribution(fig, trial_info, orientations, colors):
    """Plot trial count distribution."""
    ax = fig.add_subplot(3, 4, 9)
    
    counts = [trial_info['n_trials_per_orientation'][ori] for ori in orientations]
    bars = ax.bar(range(len(counts)), counts, color=colors, alpha=0.7)
    
    ax.set(xlabel='Orientation', ylabel='Number of Trials',
           title='Trial Distribution',
           xticks=range(len(orientations)),
           xticklabels=[f'{ori}°' for ori in orientations])
    ax.grid(True, alpha=0.3, axis='y')
    
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.5, int(h),
                ha='center', va='bottom', fontsize=10)
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')


def _plot_summary_text(fig, results, labels, unit_ids, trial_info):
    """Plot summary statistics text."""
    ax = fig.add_subplot(3, 4, 10)
    ax.axis('off')
    
    exp_params = trial_info.get('experiment_parameters', {})
    unique_ori = results['unique_orientations']
    
    summary = f"""
    Classification Summary
    
    Overall Accuracy: {accuracy_score(labels, results['predictions']):.3f}
    CV Accuracy: {results['cv_scores'].mean():.3f} ± {results['cv_scores'].std():.3f}
    Chance Level: {results['chance_accuracy']:.3f}
    
    Experiment Info:
    • Total trials: {len(labels)}
    • Orientations: {len(unique_ori)} ({min(unique_ori)}° - {max(unique_ori)}°)
    • Units: {len(unit_ids)}
    • LDA components: {results['n_components']}
    • Stimulus duration: {exp_params.get('stimulus_duration', 'N/A')}s
    • ITI duration: {exp_params.get('iti_duration', 'N/A')}s
    """
    
    ax.text(0.05, 0.95, summary, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))


def _plot_feature_importance(fig, results, unit_ids):
    """Plot feature importance based on LDA coefficients."""
    ax = fig.add_subplot(3, 4, 11)
    
    if hasattr(results['lda_full'], 'coef_'):
        importance = np.mean(np.abs(results['lda_full'].coef_), axis=0)
        top_idx = np.argsort(importance)[::-1][:15]
        
        ax.barh(range(len(top_idx)), importance[top_idx], alpha=0.7, color='orange')
        ax.set(yticks=range(len(top_idx)),
               yticklabels=[unit_ids[i].split('_')[-1] for i in top_idx],
               xlabel='Mean |Coefficient|',
               title='Top Discriminative Units')
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3, axis='x')


def _plot_prediction_confidence(fig, results, labels):
    """Plot distribution of prediction confidence."""
    ax = fig.add_subplot(3, 4, 12)
    
    confidence = np.max(results['prediction_proba'], axis=1)
    correct = results['predictions'] == labels
    
    ax.hist(confidence[correct], bins=20, alpha=0.7, label='Correct',
            color='green', density=True)
    ax.hist(confidence[~correct], bins=20, alpha=0.7, label='Incorrect',
            color='red', density=True)
    
    ax.set(xlabel='Prediction Confidence', ylabel='Density',
           title='Prediction Confidence Distribution')
    ax.legend()


# =============================================================================
# MAIN WORKFLOW
# =============================================================================

def run_analysis(data_path, time_window=(0.07, 0.16), save_plots=True, 
                output_path=None):
    """
    Complete analysis pipeline: load → analyze → visualize.
    
    Args:
        data_path: Path to neural data file
        time_window: Tuple of (start, end) time in seconds
        save_plots: Whether to save the figure
        output_path: Custom output path (defaults to data_path with .png suffix)
    
    Returns:
        Tuple of (lda_results, firing_rates, orientation_labels, unit_ids)
    """
    # Load and process data
    data = load_neural_data(data_path)
    firing_rates, orientation_labels, unit_ids, trial_info = calculate_firing_rates(
        data, time_window=time_window
    )
    
    if len(orientation_labels) == 0:
        raise ValueError("No valid trials found. Check data and time_window.")
    
    # Perform LDA analysis
    lda_results = perform_lda_analysis(firing_rates, orientation_labels)
    
    # Create visualization
    if output_path is None and save_plots:
        output_path = Path(data_path).with_suffix('.lda_analysis.png')
    
    fig = create_analysis_figure(
        lda_results, unit_ids, trial_info,
        save_path=output_path if save_plots else None
    )
    
    # Optional: Calculate orientation selectivity
    print("\nCalculating orientation selectivity...")
    selectivity = calculate_orientation_selectivity(
        unit_ids, orientation_labels, firing_rates
    )
    
    return lda_results, firing_rates, orientation_labels, unit_ids


# =============================================================================
# SCRIPT ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    # Configure your data path here
    DATA_PATH = "/Volumes/xieluanlabs/xl_cl/sortout/CnL39SG/CnL39SG_20250921_230747/embedding_analysis/CnL39SG_20250921_230747_DriftingGrating_data.pkl"
    
    try:
        results = run_analysis(
            data_path=DATA_PATH,
            time_window=(0.07, 0.16),
            save_plots=True
        )
        print("\nAnalysis complete!")
        
    except FileNotFoundError:
        print(f"Error: Data file not found at {DATA_PATH}")
        print("Please update the DATA_PATH variable.")
    except Exception as e:
        print(f"Error during analysis: {e}")
        raise