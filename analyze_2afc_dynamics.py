#!/usr/bin/env python3
"""
Analysis script for 2AFC LFADS results.

This script loads trained LFADS models and analyzes the learned dynamics
during the waiting period, comparing correct trials, errors, and timeouts.

Author: Assistant
Date: 2024
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import h5py
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import classification_report, confusion_matrix
from scipy.stats import ttest_ind
from scipy.spatial.distance import pdist, squareform
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class TwoAFCDynamicsAnalyzer:
    """
    Analyzer for 2AFC LFADS results focusing on waiting period dynamics.
    """
    
    def __init__(self, model_dir: str, data_path: str):
        """
        Initialize the analyzer.
        
        Parameters:
        -----------
        model_dir : str
            Directory containing trained LFADS model outputs
        data_path : str
            Path to the original prepared data file
        """
        self.model_dir = Path(model_dir)
        self.data_path = data_path
        
        # Load data and model outputs
        self.data = self.load_data()
        self.model_outputs = self.load_model_outputs()
        
        # Extract timing information
        self.extract_timing_info()
        
        print(f"Analyzer initialized:")
        print(f"  Model directory: {model_dir}")
        print(f"  Data file: {data_path}")
        print(f"  Time bins: {self.n_time_bins}")
        print(f"  Neurons: {self.n_neurons}")
        print(f"  Factors: {self.n_factors}")
    
    def load_data(self) -> Dict:
        """Load the original prepared data."""
        data = {}
        with h5py.File(self.data_path, 'r') as f:
            for key in f.keys():
                if hasattr(f[key], 'shape'):
                    data[key] = f[key][:]
                else:
                    data[key] = f[key]
            
            # Load attributes
            for key in f.attrs.keys():
                data[key] = f.attrs[key]
        
        return data
    
    def load_model_outputs(self) -> Dict:
        """Load LFADS model outputs."""
        outputs = {}
        
        # Look for output files
        output_files = list(self.model_dir.glob("*output*.h5"))
        if not output_files:
            raise FileNotFoundError(f"No LFADS output files found in {self.model_dir}")
        
        # Load the first output file (assuming single session)
        output_file = output_files[0]
        print(f"Loading model outputs from: {output_file}")
        
        with h5py.File(output_file, 'r') as f:
            for key in f.keys():
                if hasattr(f[key], 'shape'):
                    outputs[key] = f[key][:]
                else:
                    outputs[key] = f[key]
        
        return outputs
    
    def extract_timing_info(self):
        """Extract timing information from data."""
        self.bin_width_ms = self.data.get('bin_width_ms', 20.0)
        self.pre_cue_bins = self.data.get('pre_cue_bins', 25)
        self.waiting_period_bins = self.data.get('waiting_period_bins', 100)
        self.post_decision_bins = self.data.get('post_decision_bins', 25)
        
        # Get data dimensions
        train_data = self.data['train_encod_data']
        self.n_time_bins = train_data.shape[1]
        self.n_neurons = train_data.shape[2]
        
        # Get factor dimensions from model outputs
        if 'train_factors' in self.model_outputs:
            self.n_factors = self.model_outputs['train_factors'].shape[2]
        else:
            self.n_factors = 50  # Default
        
        # Create time axis
        total_time_ms = self.n_time_bins * self.bin_width_ms
        pre_cue_ms = self.pre_cue_bins * self.bin_width_ms
        self.time_axis = np.linspace(-pre_cue_ms, total_time_ms - pre_cue_ms, self.n_time_bins) / 1000
        
        # Define epoch boundaries
        self.cue_onset_bin = self.pre_cue_bins
        self.waiting_start_bin = self.pre_cue_bins
        self.waiting_end_bin = self.pre_cue_bins + self.waiting_period_bins
    
    def combine_data_splits(self) -> Dict:
        """Combine training and validation data for analysis."""
        combined = {}
        
        # Combine neural data
        combined['neural_data'] = np.concatenate([
            self.data['train_encod_data'],
            self.data['valid_encod_data']
        ], axis=0)
        
        # Combine factors if available
        if 'train_factors' in self.model_outputs and 'valid_factors' in self.model_outputs:
            combined['factors'] = np.concatenate([
                self.model_outputs['train_factors'],
                self.model_outputs['valid_factors']
            ], axis=0)
        
        # Combine rates if available  
        if 'train_rates' in self.model_outputs and 'valid_rates' in self.model_outputs:
            combined['rates'] = np.concatenate([
                self.model_outputs['train_rates'],
                self.model_outputs['valid_rates']
            ], axis=0)
        
        # Combine trial information
        combined['trial_outcomes'] = np.concatenate([
            self.data['trial_outcomes_train'],
            self.data['trial_outcomes_valid']
        ], axis=0)
        
        combined['evidence_levels'] = np.concatenate([
            self.data['evidence_levels_train'],
            self.data['evidence_levels_valid']
        ], axis=0)
        
        combined['choices'] = np.concatenate([
            self.data['choices_train'],
            self.data['choices_valid']
        ], axis=0)
        
        return combined
    
    def analyze_factor_dynamics(self, combined_data: Dict) -> Dict:
        """
        Analyze factor dynamics across trial types.
        
        Parameters:
        -----------
        combined_data : dict
            Combined data from train/validation splits
            
        Returns:
        --------
        dict: Analysis results
        """
        
        if 'factors' not in combined_data:
            print("No factor data available for analysis")
            return {}
        
        factors = combined_data['factors']
        outcomes = combined_data['trial_outcomes']
        
        # Define trial types
        trial_types = {
            'correct': outcomes == 1,
            'error': outcomes == 0,
            'timeout': outcomes == 2
        }
        
        results = {}
        
        # Compute average trajectories for each trial type
        results['avg_trajectories'] = {}
        results['std_trajectories'] = {}
        
        for trial_type, mask in trial_types.items():
            if np.sum(mask) > 0:
                type_factors = factors[mask]
                results['avg_trajectories'][trial_type] = np.mean(type_factors, axis=0)
                results['std_trajectories'][trial_type] = np.std(type_factors, axis=0)
                
                print(f"{trial_type.capitalize()} trials: {np.sum(mask)}")
        
        # Compute PCA on factors during waiting period
        waiting_factors = factors[:, self.waiting_start_bin:self.waiting_end_bin, :]
        waiting_factors_flat = waiting_factors.reshape(-1, waiting_factors.shape[-1])
        
        pca = PCA(n_components=3)
        waiting_pca = pca.fit_transform(waiting_factors_flat)
        waiting_pca = waiting_pca.reshape(factors.shape[0], -1, 3)
        
        results['waiting_pca'] = waiting_pca
        results['pca_explained_var'] = pca.explained_variance_ratio_
        
        print(f"PCA explained variance (first 3 PCs): {pca.explained_variance_ratio_[:3]}")
        
        # Compute trajectory distances between trial types
        results['trajectory_distances'] = self.compute_trajectory_distances(
            results['avg_trajectories']
        )
        
        return results
    
    def compute_trajectory_distances(self, avg_trajectories: Dict) -> Dict:
        """Compute distances between average trajectories."""
        distances = {}
        trial_types = list(avg_trajectories.keys())
        
        for i, type1 in enumerate(trial_types):
            for j, type2 in enumerate(trial_types):
                if i <= j:
                    continue
                
                traj1 = avg_trajectories[type1]
                traj2 = avg_trajectories[type2]
                
                # Compute Euclidean distance at each time point
                dist = np.linalg.norm(traj1 - traj2, axis=1)
                distances[f"{type1}_vs_{type2}"] = dist
        
        return distances
    
    def decode_trial_outcomes(self, combined_data: Dict) -> Dict:
        """
        Decode trial outcomes from factor dynamics.
        
        Parameters:
        -----------
        combined_data : dict
            Combined data
            
        Returns:
        --------
        dict: Decoding results
        """
        
        if 'factors' not in combined_data:
            return {}
        
        factors = combined_data['factors']
        outcomes = combined_data['trial_outcomes']
        
        # Focus on waiting period
        waiting_factors = factors[:, self.waiting_start_bin:self.waiting_end_bin, :]
        
        # Average across time for each trial
        trial_avg_factors = np.mean(waiting_factors, axis=1)
        
        # Only use trials with clear outcomes (exclude timeouts for now)
        clear_mask = (outcomes == 0) | (outcomes == 1)
        if np.sum(clear_mask) < 10:
            print("Not enough trials for decoding analysis")
            return {}
        
        X = trial_avg_factors[clear_mask]
        y = outcomes[clear_mask]
        
        # Split into train/test
        n_trials = len(X)
        train_size = int(0.7 * n_trials)
        indices = np.random.permutation(n_trials)
        
        train_idx = indices[:train_size]
        test_idx = indices[train_size:]
        
        # Fit LDA classifier
        lda = LinearDiscriminantAnalysis()
        lda.fit(X[train_idx], y[train_idx])
        
        # Predict on test set
        y_pred = lda.predict(X[test_idx])
        y_true = y[test_idx]
        
        # Compute performance metrics
        accuracy = np.mean(y_pred == y_true)
        
        results = {
            'accuracy': accuracy,
            'y_true': y_true,
            'y_pred': y_pred,
            'lda_weights': lda.coef_[0],
            'classification_report': classification_report(y_true, y_pred, output_dict=True)
        }
        
        print(f"Decoding accuracy: {accuracy:.3f}")
        
        return results
    
    def analyze_neural_reconstruction(self, combined_data: Dict) -> Dict:
        """Analyze quality of neural reconstruction."""
        
        if 'rates' not in combined_data:
            return {}
        
        neural_data = combined_data['neural_data']
        reconstructed = combined_data['rates']
        
        # Compute correlation between actual and reconstructed activity
        correlations = []
        for trial in range(neural_data.shape[0]):
            for neuron in range(neural_data.shape[2]):
                actual = neural_data[trial, :, neuron]
                recon = reconstructed[trial, :, neuron]
                if np.std(actual) > 0 and np.std(recon) > 0:
                    corr = np.corrcoef(actual, recon)[0, 1]
                    if not np.isnan(corr):
                        correlations.append(corr)
        
        results = {
            'neuron_correlations': np.array(correlations),
            'mean_correlation': np.mean(correlations),
            'std_correlation': np.std(correlations)
        }
        
        print(f"Mean reconstruction correlation: {results['mean_correlation']:.3f} ± {results['std_correlation']:.3f}")
        
        return results
    
    def plot_factor_trajectories(self, analysis_results: Dict, output_dir: str = None):
        """Plot factor trajectories for different trial types."""
        
        if 'avg_trajectories' not in analysis_results:
            return
        
        avg_traj = analysis_results['avg_trajectories']
        std_traj = analysis_results.get('std_trajectories', {})
        
        # Plot first few factors
        n_factors_to_plot = min(6, self.n_factors)
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.ravel()
        
        colors = ['green', 'red', 'orange', 'blue']
        trial_types = list(avg_traj.keys())
        
        for factor_idx in range(n_factors_to_plot):
            ax = axes[factor_idx]
            
            for i, trial_type in enumerate(trial_types):
                mean_traj = avg_traj[trial_type][:, factor_idx]
                
                ax.plot(self.time_axis, mean_traj, 
                       color=colors[i % len(colors)], 
                       label=f'{trial_type}', linewidth=2)
                
                if trial_type in std_traj:
                    std_traj_factor = std_traj[trial_type][:, factor_idx]
                    ax.fill_between(self.time_axis, 
                                  mean_traj - std_traj_factor,
                                  mean_traj + std_traj_factor,
                                  color=colors[i % len(colors)], alpha=0.2)
            
            ax.axvline(0, color='black', linestyle='--', alpha=0.5, label='Cue onset')
            ax.set_xlabel('Time (s)')
            ax.set_ylabel(f'Factor {factor_idx + 1}')
            ax.set_title(f'Factor {factor_idx + 1} Dynamics')
            ax.grid(True, alpha=0.3)
            if factor_idx == 0:
                ax.legend()
        
        plt.tight_layout()
        
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            plt.savefig(os.path.join(output_dir, 'factor_trajectories.png'), 
                       dpi=300, bbox_inches='tight')
            print(f"Factor trajectories saved to: {os.path.join(output_dir, 'factor_trajectories.png')}")
        
        plt.show()
    
    def plot_trajectory_distances(self, analysis_results: Dict, output_dir: str = None):
        """Plot distances between trial type trajectories over time."""
        
        if 'trajectory_distances' not in analysis_results:
            return
        
        distances = analysis_results['trajectory_distances']
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for comparison, dist in distances.items():
            ax.plot(self.time_axis, dist, label=comparison, linewidth=2)
        
        ax.axvline(0, color='black', linestyle='--', alpha=0.5, label='Cue onset')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Trajectory Distance')
        ax.set_title('Distance Between Trial Type Trajectories')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output_dir:
            plt.savefig(os.path.join(output_dir, 'trajectory_distances.png'), 
                       dpi=300, bbox_inches='tight')
            print(f"Trajectory distances saved to: {os.path.join(output_dir, 'trajectory_distances.png')}")
        
        plt.show()
    
    def plot_pca_trajectories(self, analysis_results: Dict, output_dir: str = None):
        """Plot PCA trajectories during waiting period."""
        
        if 'waiting_pca' not in analysis_results:
            return
        
        waiting_pca = analysis_results['waiting_pca']
        explained_var = analysis_results['pca_explained_var']
        
        # Get trial outcomes for coloring
        combined_data = self.combine_data_splits()
        outcomes = combined_data['trial_outcomes']
        
        # Define trial types and colors
        trial_types = {
            'correct': (outcomes == 1),
            'error': (outcomes == 0),
            'timeout': (outcomes == 2)
        }
        colors = {'correct': 'green', 'error': 'red', 'timeout': 'orange'}
        
        # 3D trajectory plot
        fig = plt.figure(figsize=(12, 5))
        
        # 3D plot
        ax1 = fig.add_subplot(121, projection='3d')
        for trial_type, mask in trial_types.items():
            if np.sum(mask) > 0:
                type_pca = waiting_pca[mask]
                # Plot average trajectory
                avg_traj = np.mean(type_pca, axis=0)
                ax1.plot(avg_traj[:, 0], avg_traj[:, 1], avg_traj[:, 2], 
                        color=colors[trial_type], linewidth=3, label=f'{trial_type}')
                
                # Plot individual trajectories (subset)
                n_show = min(5, np.sum(mask))
                for i in range(n_show):
                    ax1.plot(type_pca[i, :, 0], type_pca[i, :, 1], type_pca[i, :, 2], 
                            color=colors[trial_type], alpha=0.3, linewidth=1)
        
        ax1.set_xlabel(f'PC1 ({explained_var[0]:.1%})')
        ax1.set_ylabel(f'PC2 ({explained_var[1]:.1%})')
        ax1.set_zlabel(f'PC3 ({explained_var[2]:.1%})')
        ax1.set_title('Waiting Period Trajectories (PCA)')
        ax1.legend()
        
        # 2D projection
        ax2 = fig.add_subplot(122)
        for trial_type, mask in trial_types.items():
            if np.sum(mask) > 0:
                type_pca = waiting_pca[mask]
                avg_traj = np.mean(type_pca, axis=0)
                ax2.plot(avg_traj[:, 0], avg_traj[:, 1], 
                        color=colors[trial_type], linewidth=3, label=f'{trial_type}')
                
                # Plot start and end points
                ax2.scatter(avg_traj[0, 0], avg_traj[0, 1], 
                          color=colors[trial_type], s=100, marker='o', alpha=0.7)
                ax2.scatter(avg_traj[-1, 0], avg_traj[-1, 1], 
                          color=colors[trial_type], s=100, marker='s', alpha=0.7)
        
        ax2.set_xlabel(f'PC1 ({explained_var[0]:.1%})')
        ax2.set_ylabel(f'PC2 ({explained_var[1]:.1%})')
        ax2.set_title('PC1-PC2 Projection')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output_dir:
            plt.savefig(os.path.join(output_dir, 'pca_trajectories.png'), 
                       dpi=300, bbox_inches='tight')
            print(f"PCA trajectories saved to: {os.path.join(output_dir, 'pca_trajectories.png')}")
        
        plt.show()
    
    def plot_decoding_results(self, decoding_results: Dict, output_dir: str = None):
        """Plot decoding analysis results."""
        
        if not decoding_results:
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Confusion matrix
        y_true = decoding_results['y_true']
        y_pred = decoding_results['y_pred']
        cm = confusion_matrix(y_true, y_pred)
        
        ax1 = axes[0]
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
                   xticklabels=['Error', 'Correct'],
                   yticklabels=['Error', 'Correct'])
        ax1.set_title(f'Confusion Matrix\\nAccuracy: {decoding_results["accuracy"]:.3f}')
        ax1.set_xlabel('Predicted')
        ax1.set_ylabel('Actual')
        
        # LDA weights
        ax2 = axes[1]
        weights = decoding_results['lda_weights']
        ax2.bar(range(len(weights)), weights)
        ax2.set_xlabel('Factor')
        ax2.set_ylabel('LDA Weight')
        ax2.set_title('Discriminant Weights')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output_dir:
            plt.savefig(os.path.join(output_dir, 'decoding_results.png'), 
                       dpi=300, bbox_inches='tight')
            print(f"Decoding results saved to: {os.path.join(output_dir, 'decoding_results.png')}")
        
        plt.show()
    
    def plot_reconstruction_quality(self, recon_results: Dict, output_dir: str = None):
        """Plot reconstruction quality analysis."""
        
        if not recon_results:
            return
        
        correlations = recon_results['neuron_correlations']
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Histogram of correlations
        ax1 = axes[0]
        ax1.hist(correlations, bins=30, alpha=0.7, edgecolor='black')
        ax1.axvline(recon_results['mean_correlation'], color='red', linestyle='--', 
                   label=f'Mean: {recon_results["mean_correlation"]:.3f}')
        ax1.set_xlabel('Correlation')
        ax1.set_ylabel('Number of Neuron-Trial Pairs')
        ax1.set_title('Reconstruction Quality')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Box plot
        ax2 = axes[1]
        ax2.boxplot(correlations)
        ax2.set_ylabel('Correlation')
        ax2.set_title('Reconstruction Correlation Distribution')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output_dir:
            plt.savefig(os.path.join(output_dir, 'reconstruction_quality.png'), 
                       dpi=300, bbox_inches='tight')
            print(f"Reconstruction quality plot saved to: {os.path.join(output_dir, 'reconstruction_quality.png')}")
        
        plt.show()
    
    def run_full_analysis(self, output_dir: str = None):
        """Run complete analysis pipeline."""
        
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        print("Starting full analysis of 2AFC dynamics...")
        
        # Combine data splits
        print("\\n1. Combining data splits...")
        combined_data = self.combine_data_splits()
        
        # Analyze factor dynamics
        print("\\n2. Analyzing factor dynamics...")
        factor_results = self.analyze_factor_dynamics(combined_data)
        
        # Decode trial outcomes
        print("\\n3. Decoding trial outcomes...")
        decoding_results = self.decode_trial_outcomes(combined_data)
        
        # Analyze reconstruction quality
        print("\\n4. Analyzing reconstruction quality...")
        recon_results = self.analyze_neural_reconstruction(combined_data)
        
        # Generate plots
        print("\\n5. Generating plots...")
        if factor_results:
            self.plot_factor_trajectories(factor_results, output_dir)
            self.plot_trajectory_distances(factor_results, output_dir)
            self.plot_pca_trajectories(factor_results, output_dir)
        
        if decoding_results:
            self.plot_decoding_results(decoding_results, output_dir)
        
        if recon_results:
            self.plot_reconstruction_quality(recon_results, output_dir)
        
        # Save results summary
        if output_dir:
            results_summary = {
                'factor_analysis': factor_results,
                'decoding': decoding_results,
                'reconstruction': recon_results
            }
            
            # Save as text summary
            with open(os.path.join(output_dir, 'analysis_summary.txt'), 'w') as f:
                f.write("2AFC LFADS Analysis Summary\\n")
                f.write("=" * 50 + "\\n\\n")
                
                if decoding_results:
                    f.write(f"Decoding Accuracy: {decoding_results['accuracy']:.3f}\\n")
                
                if recon_results:
                    f.write(f"Mean Reconstruction Correlation: {recon_results['mean_correlation']:.3f}\\n")
                
                if factor_results and 'pca_explained_var' in factor_results:
                    pca_var = factor_results['pca_explained_var']
                    f.write(f"PCA Explained Variance (top 3): {pca_var[:3]}\\n")
            
            print(f"\\nAnalysis complete! Results saved to: {output_dir}")
        
        return {
            'factor_analysis': factor_results,
            'decoding': decoding_results, 
            'reconstruction': recon_results
        }


def main():
    """Main function for running 2AFC dynamics analysis."""
    
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze 2AFC LFADS results')
    parser.add_argument('--model_dir', type=str, required=True,
                       help='Directory containing trained LFADS model')
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to prepared data file')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Directory for analysis outputs')
    
    args = parser.parse_args()
    
    # Create analyzer
    analyzer = TwoAFCDynamicsAnalyzer(args.model_dir, args.data_path)
    
    # Run analysis
    results = analyzer.run_full_analysis(args.output_dir)
    
    print("\\nAnalysis completed successfully!")


if __name__ == "__main__":
    main()