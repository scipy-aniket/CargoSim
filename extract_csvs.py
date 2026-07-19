import pandas as pd
import os

def main():
    input_file = "tow_analysis_results/ToW_events.csv"
    
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found. Ensure analyze_tow.py ran successfully.")
        return

    # Load the ToW events that were previously extracted
    df = pd.read_csv(input_file)
    
    # Filter for strict Tug-of-War just to be absolutely sure no S-phase slips in.
    # In earlier steps 'Is_Hidden_ToW' is evaluated for Avg_K > 0.1 and Avg_D > 0.1
    if 'Is_Hidden_ToW' in df.columns:
        tow_df = df[df['Is_Hidden_ToW'] == True].copy()
    else:
        # Fallback if that column isn't there, just taking strictly ToW designated states
        tow_df = df[df['State'] == 'ToW'].copy()
        
    print(f"Processing {len(tow_df)} Strict Tug-of-War events...")

    # 1. D* distribution at the end of tug of war 
    # Create a frequency/probability distribution
    d_star_dist = tow_df['D_Star_At_End'].value_counts().sort_index().reset_index()
    d_star_dist.columns = ['D_Star_At_End', 'Frequency']
    d_star_dist['Probability'] = d_star_dist['Frequency'] / d_star_dist['Frequency'].sum()
    
    d_star_dist.to_csv("d_star_distribution.csv", index=False)
    print("=> Saved 'd_star_distribution.csv' successfully.")
    
    # 2. Unbinned data (Duration | Max Kinesin | Avg Kinesin | Catch Events | D* at End)
    cols = ['Duration', 'Max_K_Force', 'Avg_K_Force', 'Catch_Events', 'D_Star_At_End']
    
    # Safeguard to ensure we don't crash if a column has a varied name
    existing_cols = [c for c in cols if c in tow_df.columns]
    unbinned_df = tow_df[existing_cols]
    
    unbinned_df.to_csv("unbinned_tow_data.csv", index=False)
    print("=> Saved 'unbinned_tow_data.csv' successfully.")

if __name__ == "__main__":
    main()