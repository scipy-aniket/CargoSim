# CargoSim

CargoSim is a stochastic simulation framework designed to model the bidirectional transport of a single cargo driven by antagonistic molecular motors (e.g., Kinesin and Dynein) along a one-dimensional track. The simulation relies on the Gillespie algorithm to handle the mechanochemical state transitions (stepping, binding, unbinding, and activation) of the motor ensemble.

## Repository Structure

The codebase consists of seven Python files:

* **`simulation_params.py`**: The centralized configuration file. It contains a single dictionary (`params`) defining all physical constants, kinetic rates, trap settings, and motor counts used across the simulation, including which Dynein model variant to use (`dynein_motor_type`).

* **`base.py`**: Contains the core object-oriented physics engine.
    * `GenericMotor`: Base class handling force calculations, binding, unbinding, and stepping.
    * `Kinesin`, `Dynein`, `DyneinActive`: Subclasses defining motor-specific kinetic rules (e.g., Dynein's inhibited/active state transitions).
    * `Cargo`: The central manager class that aggregates motors, tracks the current system state, updates the spatial position based on force balances, and executes the core `step_gillespie()` loop.
   * *Rate Functions*: Functions (e.g., `create_stepping_function`, `create_dynein_unbinding_tfbd`) that return force-dependent kinetic rates.

* **`pipeline.py`**: Handles track generation and post-processing.
    * Utilizes `ProcessPoolExecutor` to run multiple cargo trajectories in parallel.
    * Implements `segment_track_moving_window()`, which applies a sliding window calculation to segment raw trajectory data into continuous states (Plus-ended, Minus-ended, or Stationary/Pause).
    * Extracts trajectory-level statistics (durations, run lengths, event frequencies)

* **`scan.py`**: The execution script for parameter sweeps.

* **`analyze_tow.py`**: Runs a large batch pipeline (1000 tracks by default) with the default parameters and `save_tracks=False`, then compares the resulting `ToW_events.csv` ("Strict ToW", `Avg_K & Avg_D > 0`) against `S_events.csv` ("S Segment", velocity pause). Produces, under `<output_dir>/event_analysis_plots/`:

* **`extract_csvs.py`**: Extracts:
    * `d_star_distribution.csv` — frequency/probability distribution of `D_Star_At_End`.
    * `unbinned_tow_data.csv` — the raw per-event values of `Duration`, `Max_K_Force`, `Avg_K_Force`, `Catch_Events`, and `D_Star_At_End`.
    
    This script must be run *after* `analyze_tow.py` (it depends on `tow_analysis_results/ToW_events.csv` existing).

* **`fitting.py`**: Theoretical/analytical module. Uses continuous-time Markov chain (transition rate matrix) master equation solutions to compute, for an ensemble of `N` Dynein motors, the splitting probabilities over final active motor counts and the Global Mean Inactive Time (GMIT).

## Consistency Note

Note: the parameter beta used in all scripts is referred to as "phi" in the manuscript.

## Execution & Usage

### 1. Running a Single Pipeline
To test the simulation with the default parameters and generate a small batch of trajectories with an overlay plot, run the pipeline directly:
```bash
python pipeline.py
```
By default this uses `save_tracks=True`, outputting individual `track_XXX.csv` files, `post_processed_statistics.csv`, `trajectories_overlay.png`, and (if any qualifying segments exist) `S_events.csv` and `ToW_events.csv`, all under `postprocessed_results/`.

### 2. Running a Parameter Scan
To run a parameter sweep across different motor counts:
1. Adjust `KINESIN_RANGE`, `DYNEIN_RANGE`, and `TRACKS_PER_POINT` in `scan.py`.
2. Execute the script:
```bash
python scan.py
```
This generates `master_scan_results.csv` and a `phase_plots/` directory of 2D heatmaps under `parameter_scan_results/` (or whichever `output_dir` is configured). Individual tracks are **not** saved by default in the scan (`save_tracks=False`).

### 3. Analyzing Tug-of-War Events
To generate a large sample of tracks and compare Tug-of-War vs. paused segments:
```bash
python analyze_tow.py
```
This runs a 1000-track pipeline (adjust `N_TRACKS` / `OUTPUT_DIR` in the script) and writes comparative plots to `tow_analysis_results/event_analysis_plots/`.

### 4. Extracting Filtered ToW Data
After running `analyze_tow.py`, extract the strict Tug-of-War subset a CSVs for downstream analysis:
```bash
python extract_csvs.py
```
This reads `tow_analysis_results/ToW_events.csv` and writes `d_star_distribution.csv` and `unbinned_tow_data.csv` to the working directory.

### 5. Fitting the GMIT Model
To fit the hindrance parameter to experimental data and explore the (ε, π) parameter space independently of the simulation:
```bash
python fitting.py
```

## Data & Output Reference

**Track Data (`track_XXX.csv`, produced when `save_tracks=True`):**
* `time`: Time (s)
* `cargo_pos`: Cargo position (nm)
* `n_kinesin_bound`: Number of bound Kinesin
* `n_dynein_bound`: Number of bound Dynein (L)
* `n_dynein_active`: Unbound, active Dynein (j)
* `n_dynein_inactive`: Unbound, inhibited Dynein (D\*)
* `catch_bond_events`: Cumulative count of Dynein entering the catch-bond (high-force) regime while bound
* `catch_unbind_events`: Cumulative count of Dynein unbinding while in the catch-bond regime
* `total_kinesin_force`: Summed magnitude of force across bound Kinesin (pN)
* `total_dynein_force`: Summed magnitude of force across bound Dynein (pN)

**Segment/Event Data (`S_events.csv`, `ToW_events.csv`, one row per segmented interval):**
`State`, `Start_Time`, `End_Time`, `Duration`, `Start_Pos`, `End_Pos`, `Run_Length`, `Velocity`, `Avg_K`, `Avg_D`, `Avg_D_Inactive`, `Catch_Events`, `Catch_Unbinds`, `D_Star_At_End`, `Max_K_Force`, `Avg_K_Force`, `Is_Hidden_ToW`, `Track_ID`.

**Per-Track Summary (`post_processed_statistics.csv`):**
`track_id`, `final_position`, `total_time`, `kinesin_duration`, `dynein_duration`, `pause_duration`, `kinesin_length`, `dynein_length`, `kinesin_events`, `dynein_events`, `pause_events`, `frac_kinesin`, `frac_dynein`, `frac_pause`, `num_segments`.

## System Requirements
* Python 3.8+
* `numpy`, `pandas`, `matplotlib`, `seaborn`, `scipy`