import os
import json
import pandas as pd
from pathlib import Path
import shutil

def analyze_pbt_run_manual(run_dir):
    """
    Manually analyze an existing PBT run without relying on Ray's Analysis class.
    Works with any Ray version.
    
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
    
    print(f"Found {len(experiment_state_files)} experiment state file(s)")
    
    # Find all trial directories
    trial_dirs = [d for d in run_dir.iterdir() if d.is_dir() and d.name.startswith("run_model_")]
    print(f"Found {len(trial_dirs)} trial directories")
    
    if not trial_dirs:
        print("No trial directories found!")
        return None
    
    # Analyze each trial
    trial_results = []
    for trial_dir in trial_dirs:
        trial_info = analyze_trial_manual(trial_dir)
        if trial_info:
            trial_results.append(trial_info)
    
    if not trial_results:
        print("No trials with results found")
        return None
    
    # Convert to DataFrame for easier analysis
    df = pd.DataFrame(trial_results)
    
    print(f"\nTrial Summary:")
    print(f"Total trials found: {len(df)}")
    print(f"Trials with results: {len(df[df['has_results'] == True])}")
    print(f"Trials with checkpoints: {len(df[df['has_checkpoints'] == True])}")
    
    # Show status breakdown
    status_counts = df['status'].value_counts()
    print(f"\nStatus breakdown:")
    for status, count in status_counts.items():
        print(f"  {status}: {count}")
    
    # Find best trial based on the metric (assuming lower is better for recon_smth)
    valid_trials = df[(df['has_results'] == True) & (df['best_valid_recon_smth'].notna())]
    
    if len(valid_trials) > 0:
        best_idx = valid_trials['best_valid_recon_smth'].idxmin()
        best_trial = df.loc[best_idx]
        
        print(f"\nBest Trial Found:")
        print(f"  Trial ID: {best_trial['trial_id']}")
        print(f"  Best valid/recon_smth: {best_trial['best_valid_recon_smth']:.6f}")
        print(f"  Epochs completed: {best_trial['epochs_completed']}") 
        print(f"  Final status: {best_trial['status']}")
        print(f"  Has checkpoints: {best_trial['has_checkpoints']}")
        print(f"  Trial directory: {best_trial['trial_dir']}")
        
        # Check if best model directory already exists
        best_model_dir = run_dir / "best_model"
        if not best_model_dir.exists():
            try:
                print(f"\nCopying best model to: {best_model_dir}")
                shutil.copytree(best_trial['trial_dir'], best_model_dir)
                print("✓ Best model copied successfully!")
            except Exception as e:
                print(f"✗ Error copying best model: {e}")
        else:
            print(f"\nBest model directory already exists: {best_model_dir}")
        
        return best_trial
    else:
        print("\\nNo trials with valid metrics found!")
    
    print(f"\nDetailed Trial Information:")
    print("-" * 80)
    for _, trial in df.iterrows():
        print(f"Trial: {trial['trial_id']}")
        print(f"  Status: {trial['status']}")
        print(f"  Epochs: {trial['epochs_completed']}")
        print(f"  Best metric: {trial['best_valid_recon_smth']}")
        print(f"  Has checkpoints: {trial['has_checkpoints']}")
        print()
    
    return df

def analyze_trial_manual(trial_dir):
    """Analyze a single trial directory manually"""
    trial_info = {
        'trial_id': trial_dir.name,
        'trial_dir': str(trial_dir),
        'status': 'UNKNOWN',
        'epochs_completed': 0,
        'best_valid_recon_smth': None,
        'has_checkpoints': False,
        'has_results': False
    }
    
    # Check for result.json (Ray Tune's result file)
    result_file = trial_dir / "result.json"
    if result_file.exists():
        try:
            # Read all lines (Ray Tune writes one JSON object per line)
            with open(result_file, 'r') as f:
                lines = f.readlines()
            
            if lines:
                trial_info['has_results'] = True
                
                # Parse the last line to get final results
                try:
                    last_result = json.loads(lines[-1].strip())
                    trial_info['status'] = last_result.get('trial_runner_status', 'UNKNOWN')
                    trial_info['epochs_completed'] = last_result.get('training_iteration', 0)
                except:
                    pass
                
                # Check for best metric across all epochs
                best_metric = None
                for line in lines:
                    try:
                        result = json.loads(line.strip())
                        if 'valid/recon_smth' in result:
                            metric_val = result['valid/recon_smth']
                            if best_metric is None or metric_val < best_metric:
                                best_metric = metric_val
                    except:
                        continue
                
                if best_metric is not None:
                    trial_info['best_valid_recon_smth'] = best_metric
                    
        except Exception as e:
            print(f"Warning: Error reading result.json for {trial_dir.name}: {e}")
    
    # Check for checkpoints
    checkpoint_dirs = list(trial_dir.glob("checkpoint_*"))
    trial_info['has_checkpoints'] = len(checkpoint_dirs) > 0
    
    # Also check for other common checkpoint patterns
    if not trial_info['has_checkpoints']:
        # Check for PyTorch Lightning checkpoints
        lightning_ckpts = list(trial_dir.glob("**/*.ckpt"))
        trial_info['has_checkpoints'] = len(lightning_ckpts) > 0
    
    return trial_info

def check_experiment_state(run_dir):
    """Check the experiment state file for additional information"""
    run_dir = Path(run_dir)
    
    experiment_state_files = list(run_dir.glob("experiment_state-*.json"))
    if not experiment_state_files:
        return None
    
    # Use the most recent state file
    latest_state_file = max(experiment_state_files, key=lambda x: x.stat().st_mtime)
    
    try:
        with open(latest_state_file, 'r') as f:
            state = json.load(f)
        
        print(f"\\nExperiment State (from {latest_state_file.name}):")
        print(f"  Experiment ID: {state.get('experiment_id', 'Unknown')}")
        print(f"  Runner state: {state.get('runner_state', 'Unknown')}")
        print(f"  Checkpointer state: {state.get('checkpointer', {}).get('latest_checkpoint_info', 'Unknown')}")
        
        return state
    except Exception as e:
        print(f"Could not read experiment state: {e}")
        return None

if __name__ == "__main__":
    # Set the path to your existing run
    run_dir = "/mnt/g/To Process/PMA17/PMA17 2020-10-22 Session-1/Analysis/LFADS/runs/lfads-torch-2afc/2afc/250722_161929_2afc_single"
    
    # Check experiment state
    check_experiment_state(run_dir)
    
    # Analyze the run
    result = analyze_pbt_run_manual(run_dir)
    
    if result is not None:
        print("\\n" + "="*60)
        print("Analysis complete! Check the output above for your best model.")
    else:
        print("\\n" + "="*60)
        print("Analysis failed. Check that the run directory path is correct.")