import os
import json
import pandas as pd
from pathlib import Path
import shutil

# Try different import paths for Analysis depending on Ray version
try:
    from ray.tune import Analysis
except ImportError:
    try:
        from ray.tune.analysis import Analysis
    except ImportError:
        try:
            from ray.tune.experiment_analysis import ExperimentAnalysis as Analysis
        except ImportError:
            Analysis = None
            print("Warning: Could not import Ray Analysis class. Will use manual analysis only.")

def analyze_existing_pbt_run(run_dir):
    """
    Analyze an existing PBT run to see what trials completed and find the best model.
    
    Parameters:
    -----------
    run_dir : str or Path
        Path to the existing PBT run directory
    """
    run_dir = Path(run_dir)
    
    print(f"Analyzing run directory: {run_dir}")
    print("=" * 60)
    
    # Check if this looks like a valid Ray Tune experiment
    experiment_state_files = list(run_dir.glob("experiment_state-*.json"))
    if not experiment_state_files:
        print("ERROR: No experiment state files found. This doesn't appear to be a valid Ray Tune experiment.")
        return None
    
    # Find all trial directories
    trial_dirs = [d for d in run_dir.iterdir() if d.is_dir() and d.name.startswith("run_model_")]
    print(f"Found {len(trial_dirs)} trial directories")
    
    # Analyze each trial
    trial_results = []
    for trial_dir in trial_dirs:
        trial_info = analyze_trial(trial_dir)
        if trial_info:
            trial_results.append(trial_info)
    
    if not trial_results:
        print("No completed trials found")
        return None
    
    # Convert to DataFrame for easier analysis
    df = pd.DataFrame(trial_results)
    print(f"\nTrial Summary:")
    print(f"Total trials: {len(df)}")
    print(f"Completed trials: {len(df[df['status'] == 'TERMINATED'])}")
    print(f"Running/Error trials: {len(df[df['status'] != 'TERMINATED'])}")
    
    # Find best trial based on the metric (assuming lower is better for recon_smth)
    completed_df = df[df['status'] == 'TERMINATED']
    if len(completed_df) > 0 and 'best_valid_recon_smth' in completed_df.columns:
        completed_df = completed_df.dropna(subset=['best_valid_recon_smth'])
        if len(completed_df) > 0:
            best_trial = completed_df.loc[completed_df['best_valid_recon_smth'].idxmin()]
            print(f"\nBest Trial: {best_trial['trial_id']}")
            print(f"Best valid/recon_smth: {best_trial['best_valid_recon_smth']:.6f}")
            print(f"Final epoch: {best_trial['epochs_completed']}")
            print(f"Trial directory: {best_trial['trial_dir']}")
            
            # Check if best model directory already exists
            best_model_dir = run_dir / "best_model"
            if not best_model_dir.exists():
                print(f"\nCopying best model to: {best_model_dir}")
                shutil.copytree(best_trial['trial_dir'], best_model_dir)
            else:
                print(f"Best model directory already exists: {best_model_dir}")
            
            return best_trial
    
    print("\nTrial Details:")
    print(df.to_string(index=False))
    
    return df

def analyze_trial(trial_dir):
    """Analyze a single trial directory"""
    trial_info = {
        'trial_id': trial_dir.name,
        'trial_dir': trial_dir,
        'status': 'UNKNOWN',
        'epochs_completed': 0,
        'best_valid_recon_smth': None,
        'has_checkpoints': False
    }
    
    # Check for result.json (Ray Tune's result file)
    result_file = trial_dir / "result.json"
    if result_file.exists():
        try:
            # Read all lines (Ray Tune writes one JSON object per line)
            with open(result_file, 'r') as f:
                lines = f.readlines()
            
            if lines:
                # Parse the last line to get final results
                last_result = json.loads(lines[-1])
                trial_info['status'] = last_result.get('trial_runner_status', 'UNKNOWN')
                trial_info['epochs_completed'] = last_result.get('training_iteration', 0)
                
                # Look for the metric we care about
                if 'valid/recon_smth' in last_result:
                    trial_info['best_valid_recon_smth'] = last_result['valid/recon_smth']
                
                # Check for best metric across all epochs
                best_metric = None
                for line in lines:
                    try:
                        result = json.loads(line)
                        if 'valid/recon_smth' in result:
                            metric_val = result['valid/recon_smth']
                            if best_metric is None or metric_val < best_metric:
                                best_metric = metric_val
                    except:
                        continue
                
                if best_metric is not None:
                    trial_info['best_valid_recon_smth'] = best_metric
        except Exception as e:
            print(f"Error reading result.json for {trial_dir.name}: {e}")
    
    # Check for checkpoints
    checkpoint_dirs = list(trial_dir.glob("checkpoint_*"))
    trial_info['has_checkpoints'] = len(checkpoint_dirs) > 0
    
    return trial_info

def try_ray_analysis(run_dir):
    """Try to use Ray's built-in analysis tools"""
    if Analysis is None:
        print("Ray Analysis not available in this version")
        return None
        
    try:
        analysis = Analysis(str(run_dir))
        print("Ray Analysis Summary:")
        print(f"Number of trials: {len(analysis.trials)}")
        
        # Get dataframe of results
        try:
            df = analysis.dataframe()
            if not df.empty:
                print("\nBest trial according to Ray Analysis:")
                best_trial = analysis.get_best_trial('valid/recon_smth', mode='min')
                best_config = analysis.get_best_config('valid/recon_smth', mode='min')
                print(f"Best trial: {best_trial}")
                print(f"Best config: {best_config}")
        except Exception as e:
            print(f"Could not get dataframe or best results: {e}")
        
        return analysis
    except Exception as e:
        print(f"Could not create Ray Analysis object: {e}")
        return None

if __name__ == "__main__":
    # Set the path to your existing run
    run_dir = "/mnt/g/To Process/PMA17/PMA17 2020-10-22 Session-1/Analysis/LFADS/runs/lfads-torch-2afc/2afc/250722_161929_2afc_single"
    
    print("Attempting Ray Analysis first...")
    analysis = try_ray_analysis(run_dir)
    
    print("\n" + "="*60)
    print("Manual Analysis...")
    analyze_existing_pbt_run(run_dir)