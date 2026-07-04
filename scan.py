import os
import copy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pipeline import run_pipeline
from simulation_params import params as default_params

def generate_phase_plots(master_df: pd.DataFrame, output_dir: str):
    """
    Generate 2D heatmaps (phase plots) for various aggregated statistics.
    """
    metrics = [
        'frac_kinesin', 'frac_dynein', 'frac_pause', 
        'kinesin_duration', 'dynein_duration', 
        'kinesin_length', 'dynein_length', 
        'kinesin_events', 'dynein_events',
        'final_position'
    ]
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Needs to be a pivot table: index = n_dynein, columns = n_kinesin
    for metric in metrics:
        if metric not in master_df.columns:
            continue
            
        pivot_table = master_df.pivot(index='n_dynein', columns='n_kinesin', values=metric)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(pivot_table, annot=True, fmt=".2f", cmap="viridis", cbar_kws={'label': metric})
        
        # Invert Y axis so that 1 dynein is at the bottom, increasing upwards
        plt.gca().invert_yaxis()
        
        plt.title(f"Phase Plot: {metric}")
        plt.xlabel("Number of Kinesin")
        plt.ylabel("Number of Dynein")
        plt.tight_layout()
        
        save_path = os.path.join(output_dir, f"phase_plot_{metric}.png")
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"Saved phase plot: {save_path}")


def run_parameter_scan(n_kinesin_range: list, n_dynein_range: list, n_tracks_per_point: int = 10, output_dir: str = "scan_results"):
    os.makedirs(output_dir, exist_ok=True)
    
    all_summary_stats = []
    
    for k in n_kinesin_range:
        for d in n_dynein_range:
            print(f"n_kinesin = {k}, n_dynein = {d}")
            
            # Prepare configuration
            p = copy.deepcopy(default_params)
            p['n_kinesin'] = k
            p['n_dynein'] = d
            
            # Create a specific directory for the individual tracks output if needed,
            # or just write everything in a uniform place. Here we use a dedicated folder per state.
            state_out_dir = os.path.join(output_dir, f"traces_K{k}_D{d}")
            
            # Run pipeline (don't save individual tracks to save disk, but keep the overlay and postprocessed stats)
            results_df = run_pipeline(n_tracks=n_tracks_per_point, p=p, save_tracks=False, output_dir=state_out_dir)
            
            # Average the results over the N tracks to get single data point for the heatmap
            if not results_df.empty:
                mean_stats = results_df.mean(numeric_only=True).to_dict()
                std_stats = results_df.std(numeric_only=True).to_dict()
                
                combined_stats = {
                    'n_kinesin': k, 
                    'n_dynein': d, 
                    'N': len(results_df)
                }
                
                for col_name in mean_stats:
                    if col_name == 'track_id':
                        continue
                    combined_stats[col_name] = mean_stats[col_name]
                    combined_stats[f"{col_name}_std"] = std_stats[col_name]
                
                all_summary_stats.append(combined_stats)

    master_df = pd.DataFrame(all_summary_stats)
    master_csv_path = os.path.join(output_dir, "master_scan_results.csv")
    master_df.to_csv(master_csv_path, index=False)
    print(f"\nScan results saved to {master_csv_path}")
    
    # Generate Phase Plots
    generate_phase_plots(master_df, output_dir=os.path.join(output_dir, "phase_plots"))

if __name__ == "__main__":
    # Define ranges for the scan
    KINESIN_RANGE = [1, 2]
    DYNEIN_RANGE = [2, 3, 4, 5, 6]
    TRACKS_PER_POINT = 1000
    
    run_parameter_scan(
        n_kinesin_range=KINESIN_RANGE,
        n_dynein_range=DYNEIN_RANGE,
        n_tracks_per_point=TRACKS_PER_POINT,
        output_dir="parameter_scan_results"
    )
