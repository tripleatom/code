"""
Plot Orientation Tuning Curves for All Neurons

This script loads neural data and generates individual tuning curve plots
for each unit, saved to a dedicated folder.
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
from scipy import stats
from matplotlib.gridspec import GridSpec
import warnings
import argparse
import sys
warnings.filterwarnings('ignore')


# =============================================================================
# DATA LOADING (reusing from main script)
# =============================================================================

def load_neural_data(filepath):
    """Load neural data from pickle format."""
    filepath = Path(filepath)

    if filepath.suffix != '.pkl':
        raise ValueError(f"Unsupported file format: {filepath.suffix}. Only .pkl files are supported.")

    print(f"Loading data from {filepath.suffix}: {filepath.name}")
    return _load_pickle(filepath)


def _load_pickle(filepath):
    """Load from pickle format."""
    with open(filepath, 'rb') as f:
        return pickle.load(f)


# =============================================================================
# TUNING CURVE CALCULATION
# =============================================================================

def calculate_tuning_curves(neural_data, time_window=(0.07, 0.16)):
    """
    Calculate tuning curves for all units.
    
    Returns:
        Dictionary containing:
        - unit_tuning_data: dict with unit_id as key, tuning data as value
        - unique_orientations: list of tested orientations
        - experiment_info: dict with experimental parameters
    """
    window_start, window_end = time_window
    window_duration = window_end - window_start
    
    unit_ids = list(neural_data['spike_data'].keys())
    orientations = neural_data['trial_info']['orientations']
    unique_orientations = sorted(neural_data['trial_info']['unique_orientations'])
    
    print(f"\nCalculating tuning curves:")
    print(f"  Units: {len(unit_ids)}")
    print(f"  Orientations: {unique_orientations}")
    print(f"  Time window: {window_start:.3f}-{window_end:.3f}s ({window_duration:.3f}s)")
    
    unit_tuning_data = {}
    
    for unit_id in unit_ids:
        # Collect firing rates per trial
        unit_trials = neural_data['spike_data'][unit_id]
        trial_rates = {ori: [] for ori in unique_orientations}
        
        for trial_data in unit_trials:
            orientation = trial_data['orientation']
            if orientation in unique_orientations:
                spike_times = np.array(trial_data['spike_times'])
                spikes_in_window = np.sum((spike_times >= window_start) & 
                                         (spike_times < window_end))
                firing_rate = spikes_in_window / window_duration
                trial_rates[orientation].append(firing_rate)
        
        # Calculate statistics per orientation
        mean_rates = []
        sem_rates = []
        std_rates = []
        trial_counts = []
        
        for ori in unique_orientations:
            rates = trial_rates[ori]
            if len(rates) > 0:
                mean_rates.append(np.mean(rates))
                sem_rates.append(stats.sem(rates))
                std_rates.append(np.std(rates))
                trial_counts.append(len(rates))
            else:
                mean_rates.append(0)
                sem_rates.append(0)
                std_rates.append(0)
                trial_counts.append(0)
        
        # Calculate tuning metrics
        mean_rates_arr = np.array(mean_rates)
        
        # Orientation selectivity index (OSI) - vector sum method
        theta_rad = 2 * np.deg2rad(unique_orientations)
        complex_sum = np.sum(mean_rates_arr * np.exp(1j * theta_rad))
        osi = np.abs(complex_sum) / (np.sum(mean_rates_arr) + 1e-12)
        preferred_ori = (np.angle(complex_sum) / 2.0) % np.pi
        preferred_ori_deg = np.rad2deg(preferred_ori)
        
        # Modulation index
        max_rate = np.max(mean_rates_arr)
        min_rate = np.min(mean_rates_arr)
        modulation_index = (max_rate - min_rate) / (max_rate + min_rate + 1e-12)
        
        # Baseline firing rate (mean across all orientations)
        baseline_rate = np.mean(mean_rates_arr)
        
        unit_tuning_data[unit_id] = {
            'orientations': unique_orientations,
            'mean_rates': mean_rates,
            'sem_rates': sem_rates,
            'std_rates': std_rates,
            'trial_counts': trial_counts,
            'trial_rates': trial_rates,
            'osi': osi,
            'preferred_orientation_deg': preferred_ori_deg,
            'modulation_index': modulation_index,
            'max_rate': max_rate,
            'min_rate': min_rate,
            'baseline_rate': baseline_rate
        }
    
    experiment_info = {
        'time_window': time_window,
        'experiment_parameters': neural_data.get('experiment_parameters', {}),
        'n_units': len(unit_ids)
    }
    
    return {
        'unit_tuning_data': unit_tuning_data,
        'unique_orientations': unique_orientations,
        'experiment_info': experiment_info
    }


# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================

def plot_single_tuning_curve(unit_id, tuning_data, save_path=None):
    """
    Create a comprehensive tuning curve plot for a single unit.
    
    Includes:
    - Cartesian tuning curve with error bars
    - Polar tuning curve
    - Raster plot of individual trials
    - Summary statistics
    """
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    orientations = tuning_data['orientations']
    mean_rates = np.array(tuning_data['mean_rates'])
    sem_rates = np.array(tuning_data['sem_rates'])
    
    # Main title
    fig.suptitle(f'Orientation Tuning Curve: {unit_id}', 
                 fontsize=16, fontweight='bold')
    
    # 1. Cartesian tuning curve (large, left side)
    ax1 = fig.add_subplot(gs[0:2, 0:2])
    ax1.errorbar(orientations, mean_rates, yerr=sem_rates, 
                 marker='o', markersize=8, linewidth=2, capsize=5,
                 color='#2E86AB', ecolor='#A23B72', capthick=2)
    ax1.fill_between(orientations, 
                      mean_rates - sem_rates, 
                      mean_rates + sem_rates,
                      alpha=0.2, color='#2E86AB')
    
    # Mark preferred orientation
    pref_idx = np.argmax(mean_rates)
    ax1.plot(orientations[pref_idx], mean_rates[pref_idx], 
             '*', color='red', markersize=20, 
             label=f'Preferred: {orientations[pref_idx]}°')
    
    ax1.set_xlabel('Orientation (degrees)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Firing Rate (Hz)', fontsize=12, fontweight='bold')
    ax1.set_title('Tuning Curve (Cartesian)', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(fontsize=10)
    ax1.set_xticks(orientations)
    
    # 2. Polar tuning curve (top right)
    ax2 = fig.add_subplot(gs[0, 2], projection='polar')
    theta = 2 * np.deg2rad(orientations)  # Double angle for orientation
    
    # Close the loop for polar plot
    theta_plot = np.concatenate([theta, [theta[0]]])
    rates_plot = np.concatenate([mean_rates, [mean_rates[0]]])
    
    ax2.plot(theta_plot, rates_plot, 'o-', linewidth=2, markersize=8, color='#2E86AB')
    ax2.fill(theta_plot, rates_plot, alpha=0.25, color='#2E86AB')
    
    # Mark preferred orientation
    pref_theta = 2 * np.deg2rad(tuning_data['preferred_orientation_deg'])
    pref_rate = np.max(mean_rates)
    ax2.plot(pref_theta, pref_rate, '*', color='red', markersize=15)
    
    ax2.set_title('Polar Tuning Curve', fontsize=12, fontweight='bold', pad=20)
    ax2.set_thetagrids(np.arange(0, 360, 45), 
                       [f'{int(a/2)}°' for a in np.arange(0, 360, 45)])
    ax2.grid(True)
    
    # 3. Box plot of firing rates (middle right)
    ax3 = fig.add_subplot(gs[1, 2])
    trial_rates_list = [tuning_data['trial_rates'][ori] for ori in orientations]
    
    bp = ax3.boxplot(trial_rates_list, labels=[f'{ori}°' for ori in orientations],
                      patch_artist=True, showmeans=True)
    
    # Color boxes
    for patch, ori in zip(bp['boxes'], orientations):
        color = plt.cm.hsv((ori % 180) / 180)
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    
    ax3.set_xlabel('Orientation', fontsize=10)
    ax3.set_ylabel('Firing Rate (Hz)', fontsize=10)
    ax3.set_title('Trial Distribution', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # 4. Raster plot (bottom left)
    ax4 = fig.add_subplot(gs[2, 0:2])
    
    y_pos = 0
    colors_raster = plt.cm.hsv(np.linspace(0, 1, len(orientations) + 1)[:-1])
    orientation_positions = {}
    
    for i, ori in enumerate(orientations):
        trials = tuning_data['trial_rates'][ori]
        n_trials = len(trials)
        
        if n_trials > 0:
            # Plot markers for each trial
            trial_indices = np.arange(y_pos, y_pos + n_trials)
            ax4.scatter([ori] * n_trials, trial_indices, 
                       s=50, c=[colors_raster[i]], alpha=0.6, marker='|')
            
            orientation_positions[ori] = y_pos + n_trials / 2
            y_pos += n_trials
    
    ax4.set_xlabel('Orientation (degrees)', fontsize=10, fontweight='bold')
    ax4.set_ylabel('Trial Number', fontsize=10, fontweight='bold')
    ax4.set_title('Trial Raster by Orientation', fontsize=12, fontweight='bold')
    ax4.set_xticks(orientations)
    ax4.grid(True, alpha=0.3, axis='x')
    
    # 5. Summary statistics (bottom right)
    ax5 = fig.add_subplot(gs[2, 2])
    ax5.axis('off')
    
    summary_text = f"""
    TUNING STATISTICS
    
    Orientation Selectivity:
    • OSI: {tuning_data['osi']:.3f}
    • Preferred: {tuning_data['preferred_orientation_deg']:.1f}°
    • Modulation Index: {tuning_data['modulation_index']:.3f}
    
    Firing Rates:
    • Max: {tuning_data['max_rate']:.2f} Hz
    • Min: {tuning_data['min_rate']:.2f} Hz
    • Baseline: {tuning_data['baseline_rate']:.2f} Hz
    • Range: {tuning_data['max_rate'] - tuning_data['min_rate']:.2f} Hz
    
    Trials:
    • Total: {sum(tuning_data['trial_counts'])}
    • Per orientation: {min(tuning_data['trial_counts'])}-{max(tuning_data['trial_counts'])}
    """
    
    ax5.text(0.05, 0.95, summary_text, transform=ax5.transAxes,
             fontsize=10, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    
    # Save if path provided
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    
    return fig


def plot_all_tuning_curves_summary(tuning_results, save_path=None, max_per_page=16):
    """
    Create summary plots showing all tuning curves in a grid.
    
    Args:
        tuning_results: Output from calculate_tuning_curves()
        save_path: Base path for saving (will add _page1.png, _page2.png, etc.)
        max_per_page: Maximum number of units per page (default 16 = 4x4 grid)
    """
    unit_tuning_data = tuning_results['unit_tuning_data']
    unique_orientations = tuning_results['unique_orientations']
    unit_ids = sorted(unit_tuning_data.keys())
    
    n_units = len(unit_ids)
    n_pages = int(np.ceil(n_units / max_per_page))
    
    figures = []
    
    for page in range(n_pages):
        start_idx = page * max_per_page
        end_idx = min((page + 1) * max_per_page, n_units)
        page_units = unit_ids[start_idx:end_idx]
        
        n_units_page = len(page_units)
        n_cols = 4
        n_rows = int(np.ceil(n_units_page / n_cols))
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows))
        fig.suptitle(f'Tuning Curves Summary (Page {page + 1}/{n_pages})', 
                     fontsize=16, fontweight='bold')
        
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        for idx, unit_id in enumerate(page_units):
            row = idx // n_cols
            col = idx % n_cols
            ax = axes[row, col]
            
            tuning_data = unit_tuning_data[unit_id]
            mean_rates = tuning_data['mean_rates']
            sem_rates = tuning_data['sem_rates']
            
            ax.errorbar(unique_orientations, mean_rates, yerr=sem_rates,
                       marker='o', markersize=5, linewidth=1.5, capsize=3)
            ax.fill_between(unique_orientations, 
                           np.array(mean_rates) - np.array(sem_rates),
                           np.array(mean_rates) + np.array(sem_rates),
                           alpha=0.2)
            
            ax.set_title(f'{unit_id}\nOSI: {tuning_data["osi"]:.2f}', fontsize=9)
            ax.set_xlabel('Orientation (°)', fontsize=8)
            ax.set_ylabel('Rate (Hz)', fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=7)
        
        # Hide unused subplots
        for idx in range(n_units_page, n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            axes[row, col].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            save_path_page = Path(save_path).parent / f"{Path(save_path).stem}_page{page + 1}.png"
            fig.savefig(save_path_page, dpi=150, bbox_inches='tight')
            print(f"Saved summary page {page + 1} to: {save_path_page}")
        
        figures.append(fig)
    
    return figures


# =============================================================================
# MAIN WORKFLOW
# =============================================================================

def generate_tuning_curves(data_path, time_window=(0.07, 0.16), 
                          output_folder=None, create_summary=True):
    """
    Complete pipeline: load data, calculate tuning curves, save all plots.
    
    Args:
        data_path: Path to neural data file
        time_window: Tuple of (start, end) time in seconds for analysis
        output_folder: Folder to save tuning curve plots (default: data_path_tuning_curves)
        create_summary: Whether to create summary plots with all units
    
    Returns:
        Dictionary with tuning results
    """
    # Load data
    data = load_neural_data(data_path)
    
    # Calculate tuning curves
    tuning_results = calculate_tuning_curves(data, time_window=time_window)
    unit_tuning_data = tuning_results['unit_tuning_data']
    
    # Set up output folder
    if output_folder is None:
        output_folder = Path(data_path).parent / f"{Path(data_path).stem}_tuning_curves"
    else:
        output_folder = Path(output_folder)
    
    output_folder.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving tuning curves to: {output_folder}")
    
    # Generate individual plots
    print(f"\nGenerating individual tuning curve plots...")
    unit_ids = sorted(unit_tuning_data.keys())
    
    for i, unit_id in enumerate(unit_ids, 1):
        # Clean unit_id for filename
        clean_id = unit_id.replace('/', '_').replace('\\', '_')
        save_path = output_folder / f"{clean_id}_tuning_curve.png"
        
        plot_single_tuning_curve(unit_id, unit_tuning_data[unit_id], 
                                save_path=save_path)
        
        if i % 10 == 0:
            print(f"  Processed {i}/{len(unit_ids)} units...")
    
    print(f"✓ Saved {len(unit_ids)} individual tuning curve plots")
    
    # Generate summary plots
    if create_summary:
        print(f"\nGenerating summary plots...")
        summary_path = output_folder / "tuning_curves_summary.png"
        plot_all_tuning_curves_summary(tuning_results, save_path=summary_path)
    
    # Save tuning statistics to CSV
    save_tuning_statistics(unit_tuning_data, output_folder / "tuning_statistics.csv")
    
    print(f"\n✓ All plots saved to: {output_folder}")
    
    return tuning_results


def save_tuning_statistics(unit_tuning_data, save_path):
    """Save tuning statistics to CSV file."""
    import csv
    
    save_path = Path(save_path)
    
    with open(save_path, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Header
        writer.writerow([
            'unit_id', 'osi', 'preferred_orientation_deg', 'modulation_index',
            'max_rate_hz', 'min_rate_hz', 'baseline_rate_hz', 'range_hz',
            'total_trials'
        ])
        
        # Data rows
        for unit_id in sorted(unit_tuning_data.keys()):
            data = unit_tuning_data[unit_id]
            writer.writerow([
                unit_id,
                f"{data['osi']:.4f}",
                f"{data['preferred_orientation_deg']:.2f}",
                f"{data['modulation_index']:.4f}",
                f"{data['max_rate']:.2f}",
                f"{data['min_rate']:.2f}",
                f"{data['baseline_rate']:.2f}",
                f"{data['max_rate'] - data['min_rate']:.2f}",
                sum(data['trial_counts'])
            ])
    
    print(f"✓ Saved tuning statistics to: {save_path}")


# =============================================================================
# SCRIPT ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    # Get data path from user input
    DATA_PATH = input("Enter path to neural data (.pkl file): ").strip().strip('"').strip("'")

    # Optional: customize output folder
    OUTPUT_FOLDER = None  # Set to None to use default (data_path_tuning_curves)

    try:
        tuning_results = generate_tuning_curves(
            data_path=DATA_PATH,
            time_window=(0.07, 0.16),
            output_folder=OUTPUT_FOLDER,
            create_summary=True
        )

        print("\n" + "="*60)
        print("Tuning curve analysis complete!")
        print("="*60)

    except FileNotFoundError:
        print(f"Error: Data file not found at {DATA_PATH}")
        print("Please check the file path and try again.")
    except Exception as e:
        print(f"Error during analysis: {e}")
        raise