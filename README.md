# CargoSim

CargoSim is a stochastic simulation framework designed to model the bidirectional transport of a single cargo driven by antagonistic molecular motors (e.g., Kinesin and Dynein) along a one-dimensional track. The simulation relies on the Gillespie algorithm to handle the mechanochemical state transitions (stepping, binding, unbinding, and activation) of the motor ensemble.

## Repository Structure

The codebase is organized into a modular, object-oriented structure consisting of four primary Python files:

* **`simulation_params.py`**: The centralized configuration file. It contains a single dictionary (`params`) defining all physical constants, kinetic rates, trap settings, and motor counts used across the simulation.
* **`base.py`**: Contains the core object-oriented physics engine.
    * `GenericMotor`: Base class handling force calculations, binding, unbinding, and stepping.
    * `Kinesin`, `Dynein`, `DyneinActive`: Subclasses defining motor-specific kinetic rules (e.g., Dynein's inhibited/active state transitions).
    * `Cargo`: The central manager class that aggregates motors, tracks the current system state, updates the spatial position based on force balances, and executes the core `step_gillespie()` loop.
    * *Rate Functions*: Functions (e.g., `create_stepping_function`, `create_dynein_unbinding_tfbd`) that return force-dependent kinetic rates.
* **`pipeline.py`**: Handles track generation and post-processing.
    * Utilizes `ProcessPoolExecutor` to run multiple cargo trajectories in parallel.
    * Implements `segment_track_moving_window()`, which applies a sliding window calculation to segment raw trajectory data into continuous states (Plus-ended, Minus-ended, or Stationary/Pause).
    * Extracts trajectory-level statistics (durations, run lengths, event frequencies).
* **`scan.py`**: The execution script for parameter sweeps.
    * Iterates over 2D grids of motor counts (e.g., sweeping combinations of $N$ Kinesin vs. $M$ Dynein).
    * Aggregates the statistical outputs from `pipeline.py` into a master dataframe.
    * Generates 2D heatmaps for various metrics.

## Execution & Usage

### 1. Running a Single Pipeline
To test the simulation with the default parameters and generate a small batch of trajectories with an overlay plot, run the pipeline directly:
```bash
python pipeline.py
```
This outputs individual CSV tracks (if enabled), `post_processed_statistics.csv`, and a `trajectories_overlay.png` plot.

### 2. Running a Parameter Scan
To run a parameter sweep across different motor counts:
1. Adjust `KINESIN_RANGE`, `DYNEIN_RANGE`, and `TRACKS_PER_POINT` in `scan.py`.
2. Execute the script:
```bash
python scan.py
```
This generates `master_scan_results.csv` and a directory of 2D phase plots for metrics like pause fractions and run lengths.

### 3. Extracting Raw Data
To perform custom analysis, you need the raw simulation tracks.

* **Single Runs:** `pipeline.py` defaults to `save_tracks=True`. Outputs `track_XXX.csv`.
* **Parameter Scans:** In `scan.py`, change `save_tracks=False` to `save_tracks=True` inside the `run_pipeline` call. 
*Note: Saving tracks during large sweeps consumes significant disk space.*

**Track Data Structure:**
Each `track_XXX.csv` contains:
* `time`: Time (s)
* `cargo_pos`: Position (nm)
* `n_kinesin_bound`: Attached Kinesin
* `n_dynein_bound`: Attached Dynein
* `n_dynein_active`: Unbound, active Dynein
* `n_dynein_inactive`: Unbound, inhibited Dynein

## System Requirements
* Python 3.8+
* `numpy`, `pandas`, `matplotlib`, `seaborn`