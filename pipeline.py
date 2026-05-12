import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Dict, Any, Tuple

from base import (
    Cargo, 
    Dynein, 
    DyneinActive, 
    Kinesin,
    create_stepping_function,
    create_kinesin_unbinding,
    create_dynein_unbinding_tfbd
)
from simulation_params import params as default_params

def segment_track_moving_window(track_df: pd.DataFrame, 
                                window_size: float = 0.5, 
                                step_size: float = 0.01, 
                                v_thresh: float = 150.0,
                                min_duration: float = 0.15) -> pd.DataFrame:
    """
    Segments track using a sliding window.
    'P' -> Plus ended/Kinesin driven
    'M' -> Minus ended/Dynein driven
    'S' -> Pause/Stationary
    """
    t_max = track_df['time'].iloc[-1]
    if t_max <= 0:
        return pd.DataFrame()
        
    t_reg = np.arange(0, t_max, step_size)
    
    pos_reg = np.interp(t_reg, track_df['time'], track_df['cargo_pos'])
    nk_reg = np.interp(t_reg, track_df['time'], track_df['n_kinesin_bound'])
    nd_reg = np.interp(t_reg, track_df['time'], track_df['n_dynein_bound'])
    nin_reg = np.interp(t_reg, track_df['time'], track_df['n_dynein_inactive'])
    
    s_pos = pd.Series(pos_reg)
    window_steps = max(1, int(window_size / step_size))
    
    disp = s_pos.diff(window_steps).shift(-window_steps // 2)
    vel_reg = disp / window_size
    vel_reg = vel_reg.fillna(0).values
    
    states = np.full(len(t_reg), 'S', dtype=object)
    states[vel_reg > v_thresh] = 'P'
    states[vel_reg < -v_thresh] = 'M'
    
    raw_segments = []
    if len(states) == 0: 
        return pd.DataFrame()

    current_state = states[0]
    start_idx = 0
    
    for i in range(1, len(states)):
        if states[i] != current_state:
            raw_segments.append({'s_idx': start_idx, 'e_idx': i, 'state': current_state})
            current_state = states[i]
            start_idx = i
            
    raw_segments.append({'s_idx': start_idx, 'e_idx': len(states)-1, 'state': current_state})
    
    cleaned_states = []
    if not raw_segments:
        return pd.DataFrame()

    current_seg = raw_segments[0]
    
    for i in range(1, len(raw_segments)):
        next_seg = raw_segments[i]
        next_duration = (next_seg['e_idx'] - next_seg['s_idx']) * step_size
        
        if next_duration < min_duration:
            current_seg['e_idx'] = next_seg['e_idx']
        else:
            if current_seg['state'] == next_seg['state']:
                current_seg['e_idx'] = next_seg['e_idx']
            else:
                cleaned_states.append(current_seg)
                current_seg = next_seg
                
    cleaned_states.append(current_seg)
    
    final_output = []
    for seg in cleaned_states:
        s_idx, e_idx = seg['s_idx'], seg['e_idx']
        
        t_start = t_reg[s_idx]
        t_end = t_reg[e_idx]
        duration = t_end - t_start
        
        x_start = pos_reg[s_idx]
        x_end = pos_reg[e_idx]
        
        avg_k = np.mean(nk_reg[s_idx:e_idx+1])
        avg_d = np.mean(nd_reg[s_idx:e_idx+1])
        avg_in = np.mean(nin_reg[s_idx:e_idx+1])
        is_tow = (avg_k > 0.1) and (avg_d > 0.1)
        
        final_output.append({
            'State': seg['state'],
            'Start_Time': t_start,
            'End_Time': t_end,
            'Duration': duration,
            'Start_Pos': x_start,
            'End_Pos': x_end,
            'Run_Length': x_end - x_start,
            'Velocity': (x_end - x_start) / duration if duration > 0 else 0,
            'Avg_K': avg_k,
            'Avg_D': avg_d,
            'Avg_D_Inactive': avg_in,
            'Is_Hidden_ToW': is_tow
        })
        
    return pd.DataFrame(final_output)

def generate_single_track(track_id: int, p: dict, save_tracks: bool, output_dir: str) -> Tuple[int, pd.DataFrame]:
    seed = p['base_seed'] + track_id
    np.random.seed(seed)

    dyn_step = create_stepping_function(p['v0_d'], p['f_stall_d'], p['stepsize_dynein'], w=p['w_dynein'])
    dyn_unbind = create_dynein_unbinding_tfbd(
        p['eps_d'], p['f_inhibition_trigger_d'], p['f_d_detach'], p['alpha_tfbd'], p['f_0_tfbd']
    )

    kin_step = create_stepping_function(p['v0_k'], p['f_stall_k'], p['stepsize_kinesin'], w=p['w_kinesin'])
    kin_unbind = create_kinesin_unbinding(p['eps_k'], p['f_k_detach'])

    cargo = Cargo(trap_stiffness=p['trap_stiffness'], trap_center=p['trap_center'])

    dynein_class = DyneinActive if p.get('dynein_motor_type') == 'DyneinActive' else Dynein
    
    for j in range(p['n_dynein']):
        d = dynein_class(
            motor_id=j, stiffness=p['k_stiffness_dynein'], rest_length=p['rest_length_dynein'],
            step_size=p['stepsize_dynein'], binding_rate=p['pi_d'], func_stepping_rate=dyn_step,
            func_unbinding_rate=dyn_unbind, stall_force=p['f_stall_d'],
            k_activation_0=p['act_k0'], beta_hindrance=p['beta'], inhibition_trigger_force=p['f_inhibition_trigger_d']
        )
        cargo.add_motor(d)

    for j in range(p['n_kinesin']):
        k = Kinesin(
            motor_id=p['n_dynein'] + j, stiffness=p['k_stiffness_kinesin'], rest_length=p['rest_length_kinesin'],
            step_size=p['stepsize_kinesin'], binding_rate=p['pi_k'], func_stepping_rate=kin_step,
            func_unbinding_rate=kin_unbind, stall_force=p['f_stall_k']
        )
        cargo.add_motor(k)

    while cargo.time < p['t_max']:
        cargo.step_gillespie()
        
    df = pd.DataFrame(cargo.history)
    
    if save_tracks:
        os.makedirs(output_dir, exist_ok=True)
        filename = os.path.join(output_dir, f"track_{track_id:03d}.csv")
        df.to_csv(filename, index=False)
        
    return track_id, df

def process_track(track_id: int, track_df: pd.DataFrame) -> dict:
    segments = segment_track_moving_window(track_df)
    
    final_pos = track_df['cargo_pos'].iloc[-1] if not track_df.empty else 0.0
    total_time = track_df['time'].iloc[-1] if not track_df.empty else 0.0
    
    if segments.empty or len(segments) <= 1:
        return {
            'track_id': track_id, 'final_position': final_pos, 'total_time': total_time,
            'kinesin_duration': 0, 'dynein_duration': 0, 'pause_duration': 0,
            'kinesin_length': 0, 'dynein_length': 0,
            'kinesin_events': 0, 'dynein_events': 0, 'pause_events': 0,
            'frac_kinesin': 0, 'frac_dynein': 0, 'frac_pause': 1.0,
            'num_segments': 0
        }
    
    # Exclude last event as it is usually incomplete
    seg_complete = segments.iloc[:-1]
    
    mask_p = seg_complete['State'] == 'P'
    mask_m = seg_complete['State'] == 'M'
    mask_s = seg_complete['State'] == 'S'
    
    k_dur = seg_complete[mask_p]['Duration'].sum()
    d_dur = seg_complete[mask_m]['Duration'].sum()
    s_dur = seg_complete[mask_s]['Duration'].sum()
    
    k_len = seg_complete[mask_p]['Run_Length'].abs().sum()
    d_len = seg_complete[mask_m]['Run_Length'].abs().sum()
    
    k_ev = mask_p.sum()
    d_ev = mask_m.sum()
    s_ev = mask_s.sum()
    
    tot_dur = k_dur + d_dur + s_dur
    if tot_dur == 0:
        tot_dur = 1
        
    return {
        'track_id': track_id,
        'final_position': final_pos,
        'total_time': total_time,
        'kinesin_duration': k_dur,
        'dynein_duration': d_dur,
        'pause_duration': s_dur,
        'kinesin_length': k_len,
        'dynein_length': d_len,
        'kinesin_events': k_ev,
        'dynein_events': d_ev,
        'pause_events': s_ev,
        'frac_kinesin': k_dur / tot_dur,
        'frac_dynein': d_dur / tot_dur,
        'frac_pause': s_dur / tot_dur,
        'num_segments': len(seg_complete)
    }

def plot_overlay(tracks_dict: dict, output_file: str):
    plt.figure(figsize=(10, 7))
    for track_id, track_data in tracks_dict.items():
        plt.plot(track_data['time'], track_data['cargo_pos'], alpha=0.6, linewidth=0.8)
    
    plt.xlabel('Time (s)', fontsize=12)
    plt.ylabel('Cargo Position (nm)', fontsize=12)
    plt.title(f'Overlay of {len(tracks_dict)} Trajectories', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close()

def run_pipeline(n_tracks: int, p: dict, save_tracks: bool = False, output_dir: str = "tracks_out", max_workers: int = 4) -> pd.DataFrame:
    print(f"Generating {n_tracks} tracks using {max_workers} processes...")
    
    tracks_dict = {}
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(generate_single_track, i, p, save_tracks, output_dir) for i in range(n_tracks)]
        for future in as_completed(futures):
            tid, df = future.result()
            tracks_dict[tid] = df
            
    print("Processing generated tracks...")
    results = []
    for tid, df in tracks_dict.items():
        res = process_track(tid, df)
        results.append(res)
        
    results_df = pd.DataFrame(results)
    
    os.makedirs(output_dir, exist_ok=True)
    plot_file = os.path.join(output_dir, "trajectories_overlay.png")
    plot_overlay(tracks_dict, plot_file)
    print(f"Overlay plot saved to {plot_file}")
    
    stats_file = os.path.join(output_dir, "post_processed_statistics.csv")
    results_df.to_csv(stats_file, index=False)
    print(f"Processing complete. Statistics saved to {stats_file}")
    
    return results_df

if __name__ == "__main__":
    print("Running Postprocessing Pipeline...")
    df = run_pipeline(n_tracks=4, p=default_params, save_tracks=True, output_dir="postprocessed_results")
    print(df.head())
