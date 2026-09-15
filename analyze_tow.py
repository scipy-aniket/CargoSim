import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pipeline import run_pipeline
from simulation_params import params as default_params

def generate_comparative_plots(output_dir: str):
    # Load the generated event CSVs
    tow_file = os.path.join(output_dir, "ToW_events.csv")
    s_file = os.path.join(output_dir, "S_events.csv")
    
    if not os.path.exists(tow_file) or not os.path.exists(s_file):
        print("Event files not found. Ensure the pipeline ran successfully.")
        return
        
    df_tow = pd.read_csv(tow_file)
    df_s = pd.read_csv(s_file)
    
    # Add labels for combined plotting
    df_tow['Event_Type'] = 'Strict ToW (Avg K & D > 0)'
    df_s['Event_Type'] = 'S Segment (Velocity Pause)'
    
    # Calculate Catch Rate (Events / Second)
    df_tow['Catch_Rate'] = df_tow['Catch_Events'] / df_tow['Duration']
    df_s['Catch_Rate'] = df_s['Catch_Events'] / df_s['Duration']
    
    # Combine datasets for comparative plotting
    df_combined = pd.concat([df_tow, df_s], ignore_index=True)
    
    plots_dir = os.path.join(output_dir, "event_analysis_plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    sns.set_theme(style="whitegrid")
    
    # 1. Distribution of Catch Bond Events Triggered
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df_combined, x='Catch_Events', hue='Event_Type', 
                 bins=range(int(df_combined['Catch_Events'].max()) + 2), 
                 stat="density", common_norm=False,
                 multiple="dodge", shrink=0.8)
    plt.title("Distribution of Catch Bond Events Triggered")
    plt.xlabel("Number of Catch Bond Events")
    plt.ylabel("Density")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "dist_catch_events.png"), dpi=150)
    plt.close()
    
    # 2. Distribution of D* at the end of the event
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df_combined, x='D_Star_At_End', hue='Event_Type', 
                 bins=range(int(df_combined['D_Star_At_End'].max()) + 2), 
                 stat="density", common_norm=False,
                 multiple="dodge", shrink=0.8)
    plt.title("Distribution of D* (Inactive Dyneins) at end of Event")
    plt.xlabel("Number of D*")
    plt.ylabel("Density")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "dist_d_star.png"), dpi=150)
    plt.close()
    
    # Helper for binned scatter plots with SEM
    def plot_binned_sem(df, x_col, y_col, title, xlabel, ylabel, filename, bins=15):
        plt.figure(figsize=(10, 6))
        
        # Create uniform bins excluding NaNs
        bins_edges = np.linspace(df[x_col].min(), df[x_col].max(), bins + 1)
        df_binned = df.copy()
        df_binned['bin'] = pd.cut(df_binned[x_col], bins=bins_edges, include_lowest=True)
        
        # Calculate Mean and Standard Error of the Mean (SEM)
        grouped = df_binned.groupby('bin', observed=False)[y_col]
        means = grouped.mean()
        sems = grouped.sem()
        
        # Get bin centers
        bin_centers = [b.mid for b in means.index]
        
        plt.errorbar(bin_centers, means, yerr=sems, fmt='o', capsize=5, 
                     color="darkblue", ecolor="gray", markersize=6, alpha=0.8)
        
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, filename), dpi=150)
        plt.close()

    # 3. Catch Bond Events vs Maximum Kinesin Force (Strict ToW Only)
    plot_binned_sem(
        df=df_tow, x_col='Max_K_Force', y_col='Catch_Events',
        title="Catch Bond Events vs Max Kinesin Force (ToW Only, Mean ± SEM)",
        xlabel="Max Total Kinesin Force (pN)", ylabel="Catch Events Triggered",
        filename="scatter_max_force_vs_catch_events.png", bins=15
    )

    # 4. Catch Unbinds vs Maximum Kinesin Force (Strict ToW Only)
    plot_binned_sem(
        df=df_tow, x_col='Max_K_Force', y_col='Catch_Unbinds',
        title="Catch Unbinds vs Max Kinesin Force (ToW Only, Mean ± SEM)",
        xlabel="Max Total Kinesin Force (pN)", ylabel="Catch Unbinds Triggered",
        filename="scatter_max_force_vs_catch_unbinds.png", bins=15
    )

    # 7. Catch Bond Events vs Average Kinesin Force (Strict ToW Only)
    plot_binned_sem(
        df=df_tow, x_col='Avg_K_Force', y_col='Catch_Events',
        title="Catch Bond Events vs Average Kinesin Force (ToW Only, Mean ± SEM)",
        xlabel="Average Total Kinesin Force (pN)", ylabel="Catch Events Triggered",
        filename="scatter_avg_force_vs_catch_events.png", bins=15
    )

    # 8. Catch Unbinds vs Average Kinesin Force (Strict ToW Only)
    plot_binned_sem(
        df=df_tow, x_col='Avg_K_Force', y_col='Catch_Unbinds',
        title="Catch Unbinds vs Average Kinesin Force (ToW Only, Mean ± SEM)",
        xlabel="Average Total Kinesin Force (pN)", ylabel="Catch Unbinds Triggered",
        filename="scatter_avg_force_vs_catch_unbinds.png", bins=15
    )
    
    # 9. Catch Rate vs Average Kinesin Force (Strict ToW Only)
    plot_binned_sem(
        df=df_tow, x_col='Avg_K_Force', y_col='Catch_Rate',
        title="Catch Rate vs Average Kinesin Force (ToW Only, Mean ± SEM)",
        xlabel="Average Total Kinesin Force (pN)", ylabel="Catch Rate (Events/s)",
        filename="scatter_avg_force_vs_catch_rate.png", bins=15
    )

    # 10. Catch Rate vs Maximum Kinesin Force (Strict ToW Only)
    plot_binned_sem(
        df=df_tow, x_col='Max_K_Force', y_col='Catch_Rate',
        title="Catch Rate vs Maximum Kinesin Force (ToW Only, Mean ± SEM)",
        xlabel="Max Total Kinesin Force (pN)", ylabel="Catch Rate (Events/s)",
        filename="scatter_max_force_vs_catch_rate.png", bins=15
    )

    # 5. Catch Bond Events vs Event Duration (Strict ToW Only)
    plot_binned_sem(
        df=df_tow, x_col='Duration', y_col='Catch_Events',
        title="Catch Bond Events vs Event Duration (ToW Only, Mean ± SEM)",
        xlabel="Duration (s)", ylabel="Catch Events Triggered",
        filename="scatter_duration_vs_catch_events.png", bins=15
    )

    # 6. D* at End vs Event Duration (Strict ToW Only)
    plot_binned_sem(
        df=df_tow, x_col='Duration', y_col='D_Star_At_End',
        title="D* at End vs Event Duration (ToW Only, Mean ± SEM)",
        xlabel="Duration (s)", ylabel="D* (Inactive Dyneins) at End",
        filename="scatter_duration_vs_d_star.png", bins=15
    )
    
    print(f"All comparative plots generated and saved to {plots_dir}")


def main():
    N_TRACKS = 1000
    OUTPUT_DIR = "tow_analysis_results"
    
    print(f"Running pipeline for {N_TRACKS} tracks with default parameters...")
    # Generate tracks. Using save_tracks=False so it doesn't dump 100 massive individual CSVs.
    run_pipeline(
        n_tracks=N_TRACKS, 
        p=default_params, 
        save_tracks=False, 
        output_dir=OUTPUT_DIR,
        max_workers=14
    )
    
    print("\nGenerating comparative plots...")
    generate_comparative_plots(OUTPUT_DIR)
    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
