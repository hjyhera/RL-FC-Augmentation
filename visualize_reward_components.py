import matplotlib.pyplot as plt
import numpy as np
import json
import glob
import re
import os
from pathlib import Path
from scipy.ndimage import uniform_filter1d

def parse_reward_components():
    """Parse reward components from selection history JSON files"""
    import glob
    import json
    
    fold_data = {}
    
    # Parse from selection history JSON files
    history_pattern = "logs/synthetic_info/selection_history_fold*.json"
    history_files = glob.glob(history_pattern)
    
    if history_files:
        print("Found selection history JSON files, parsing reward components...")
        for history_file in history_files:
            fold_num = int(re.search(r'fold(\d+)', history_file).group(1))
            
            try:
                with open(history_file, 'r') as f:
                    history_data = json.load(f)
                
                episodes = []
                rewards = []
                sel_ratios = []
                components = {
                    'immediate': [],
                    'shaped': [],
                    'base_quality': [],
                    'ratio_penalty': [],
                    'diversity_bonus': [],
                    'disc_mean': [],
                    'dcae_mean': [],
                    'likeness_score': [],
                    'validation_reward': [],
                    'final_reward': []
                }
                
                for entry in history_data:
                    episodes.append(entry['episode'])
                    rewards.append(entry['reward'])
                    sel_ratios.append(entry.get('sel_ratio', 0.0))
                    
                    # Extract reward components if available
                    if 'reward_components' in entry and entry['reward_components']:
                        comp = entry['reward_components']
                        components['immediate'].append(comp.get('immediate', 0.0))
                        components['shaped'].append(comp.get('shaped', 0.0))
                        components['base_quality'].append(comp.get('base_quality', 0.0))
                        components['ratio_penalty'].append(comp.get('ratio_penalty', 0.0))
                        components['diversity_bonus'].append(comp.get('diversity_bonus', 0.0))
                        components['disc_mean'].append(comp.get('disc_mean', 0.0))
                        components['dcae_mean'].append(comp.get('dcae_mean', 0.0))
                        components['likeness_score'].append(comp.get('likeness_score', 0.0))
                        components['validation_reward'].append(comp.get('validation_reward', 0.0))
                        components['final_reward'].append(comp.get('final_reward', entry['reward']))
                    else:
                        # Fill with zeros if no components available
                        for key in components:
                            components[key].append(0.0)
                
                fold_data[fold_num] = {
                    'episodes': episodes,
                    'rewards': rewards,
                    'sel_ratios': sel_ratios,
                    'components': components
                }
                print(f"Parsed Fold {fold_num}: {len(episodes)} entries with reward components")
            except Exception as e:
                print(f"Error parsing {history_file}: {e}")
    
    return fold_data

def smooth_data(data, window_size=5):
    """Apply smoothing to data using uniform filter"""
    if len(data) < window_size:
        return data
    return uniform_filter1d(np.array(data), size=window_size, mode='nearest')

def normalize_data(data, method='minmax'):
    """Normalize data to 0-1 range using min-max or z-score"""
    data = np.array(data)
    if method == 'minmax':
        min_val = np.min(data)
        max_val = np.max(data)
        if max_val == min_val:
            return np.zeros_like(data)
        return (data - min_val) / (max_val - min_val)
    elif method == 'zscore':
        mean_val = np.mean(data)
        std_val = np.std(data)
        if std_val == 0:
            return np.zeros_like(data)
        z_scores = (data - mean_val) / std_val
        # Convert z-scores to 0-1 range using sigmoid
        return 1 / (1 + np.exp(-z_scores))
    return data

def create_reward_components_plot():
    """Create visualization of reward components across epochs for each fold"""
    
    fold_data = parse_reward_components()
    
    if not fold_data:
        print("No reward component data found!")
        return None
    
    # Create figure with subplots (2x3 grid for individual folds + 1 average)
    fig, axes = plt.subplots(2, 3, figsize=(20, 14))
    fig.suptitle('Reward Components Across Episodes by Fold', fontsize=16)
    
    # Flatten axes for easier indexing
    axes = axes.flatten()
    
    colors = {
        'final_reward': '#000000',      # Black - final reward
        'disc_mean': '#1f77b4',         # Blue - discriminator mean
        'dcae_mean': '#ff7f0e',         # Orange - DCAE mean
        'shaped': '#2ca02c',            # Green - shaped reward
        'likeness_score': '#d62728',    # Red - likeness score
        'validation_reward': '#9467bd', # Purple - validation reward
        'base_quality': '#8c564b',      # Brown - base quality
        'diversity_bonus': '#e377c2'    # Pink - diversity bonus
    }
    
    # Plot individual folds
    for fold_idx, (fold, data) in enumerate(fold_data.items()):
        if fold_idx >= 5:  # Only plot first 5 folds
            break
            
        ax = axes[fold_idx]
        episodes = data['episodes']
        components = data['components']
        sel_ratios = data['sel_ratios']
        
        # Normalize and smooth reward components
        disc_norm = normalize_data(components['disc_mean'], method='minmax')
        dcae_norm = normalize_data(components['dcae_mean'], method='minmax')
        likeness_norm = normalize_data(components['likeness_score'], method='minmax')
        # Flip validation reward to make it symmetric (convert negative to positive)
        validation_flipped = [-x for x in components['validation_reward']]
        validation_norm = normalize_data(validation_flipped, method='minmax')
        
        disc_smooth = smooth_data(disc_norm, window_size=5)
        dcae_smooth = smooth_data(dcae_norm, window_size=5)
        likeness_smooth = smooth_data(likeness_norm, window_size=5)
        validation_smooth = smooth_data(validation_norm, window_size=5)
        sel_ratio_smooth = smooth_data(sel_ratios, window_size=5)
        
        # Plot reward components on left y-axis
        ax.plot(episodes, disc_smooth, color=colors['disc_mean'], 
                linewidth=2, label='Disc Mean (norm)', alpha=0.8)
        ax.plot(episodes, dcae_smooth, color=colors['dcae_mean'], 
                linewidth=2, label='DCAE Mean (norm)', alpha=0.8)
        ax.plot(episodes, likeness_smooth, color=colors['likeness_score'], 
                linewidth=2, label='Likeness Score (norm)', alpha=0.8)
        ax.plot(episodes, validation_smooth, color=colors['validation_reward'], 
                linewidth=2, label='Validation (flipped)', alpha=0.8)
        
        # Create second y-axis for selection ratio
        ax2 = ax.twinx()
        ax2.plot(episodes, sel_ratio_smooth, color='black', 
                linewidth=2, label='Selection Ratio', alpha=0.8, linestyle='--')
        ax2.set_ylabel('Selection Ratio', color='black')
        ax2.tick_params(axis='y', labelcolor='black')
        ax2.set_ylim(0, 1)
        
        # Skip ratio penalty plotting
        
        # Add zero line for reference
        ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
        
        # Formatting
        ax.set_title(f'Fold {fold}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Normalized Reward Value (0-1)')
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
        
        # Combine legends from both axes
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=8)
        
        # Add statistics text for selection ratio
        sel_ratio_mean = np.mean(sel_ratios)
        sel_ratio_std = np.std(sel_ratios)
        ax.text(0.02, 0.02, f'Sel Ratio: {sel_ratio_mean:.3f}±{sel_ratio_std:.3f}', 
                transform=ax.transAxes, verticalalignment='bottom',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Use the 6th subplot for average across all folds
    if len(fold_data) > 1:
        ax_avg = axes[5]
        
        # Calculate average components across all folds with normalization
        all_episodes = []
        component_names = ['disc_mean', 'dcae_mean', 'likeness_score', 'validation_reward']
        
        # Collect all data for global normalization
        all_data = {name: [] for name in component_names}
        all_sel_ratios = []
        for fold, data in fold_data.items():
            for name in component_names:
                if name == 'validation_reward':
                    # Flip validation reward values
                    all_data[name].extend([-x for x in data['components'][name]])
                else:
                    all_data[name].extend(data['components'][name])
            all_sel_ratios.extend(data['sel_ratios'])
        
        # Calculate global min/max for consistent normalization
        global_stats = {}
        for name in component_names:
            global_stats[name] = {
                'min': np.min(all_data[name]),
                'max': np.max(all_data[name])
            }
        
        # Find maximum episode range to show all episodes
        max_episodes = max(len(data['episodes']) for data in fold_data.values())
        
        avg_components = {name: [] for name in component_names}
        std_components = {name: [] for name in component_names}
        avg_sel_ratios = []
        std_sel_ratios = []
        
        for ep in range(1, max_episodes + 1):
            episode_values = {name: [] for name in component_names}
            episode_sel_ratios = []
            
            for fold, data in fold_data.items():
                if ep <= len(data['episodes']):
                    ep_idx = ep - 1
                    # Collect selection ratios
                    if ep_idx < len(data['sel_ratios']):
                        episode_sel_ratios.append(data['sel_ratios'][ep_idx])
                    
                    for name in component_names:
                        if name in data['components']:
                            # Normalize using global stats
                            raw_val = data['components'][name][ep_idx]
                            if name == 'validation_reward':
                                raw_val = -raw_val  # Flip validation reward
                            min_val = global_stats[name]['min']
                            max_val = global_stats[name]['max']
                            if max_val != min_val:
                                norm_val = (raw_val - min_val) / (max_val - min_val)
                            else:
                                norm_val = 0.0
                            episode_values[name].append(norm_val)
            
            all_episodes.append(ep)
            # Calculate averages for selection ratios
            if episode_sel_ratios:
                avg_sel_ratios.append(np.mean(episode_sel_ratios))
                std_sel_ratios.append(np.std(episode_sel_ratios))
            else:
                avg_sel_ratios.append(0.0)
                std_sel_ratios.append(0.0)
                
            for name in component_names:
                if episode_values[name]:
                    avg_components[name].append(np.mean(episode_values[name]))
                    std_components[name].append(np.std(episode_values[name]))
                else:
                    avg_components[name].append(0.0)
                    std_components[name].append(0.0)
        
        # Plot average components (excluding final and shaped) with smoothing
        disc_avg_smooth = smooth_data(avg_components['disc_mean'], window_size=5)
        dcae_avg_smooth = smooth_data(avg_components['dcae_mean'], window_size=5)
        likeness_avg_smooth = smooth_data(avg_components['likeness_score'], window_size=5)
        validation_avg_smooth = smooth_data(avg_components['validation_reward'], window_size=5)
        sel_ratio_avg_smooth = smooth_data(avg_sel_ratios, window_size=5)
        
        # Also smooth the standard deviations
        disc_std_smooth = smooth_data(std_components['disc_mean'], window_size=5)
        dcae_std_smooth = smooth_data(std_components['dcae_mean'], window_size=5)
        likeness_std_smooth = smooth_data(std_components['likeness_score'], window_size=5)
        validation_std_smooth = smooth_data(std_components['validation_reward'], window_size=5)
        sel_ratio_std_smooth = smooth_data(std_sel_ratios, window_size=5)
        
        # Plot reward components on left y-axis
        ax_avg.plot(all_episodes, disc_avg_smooth, color=colors['disc_mean'], 
                    linewidth=2, label='Disc Mean (avg)', alpha=0.9)
        
        ax_avg.plot(all_episodes, dcae_avg_smooth, color=colors['dcae_mean'], 
                    linewidth=2, label='DCAE Mean (avg)', alpha=0.9)
        
        ax_avg.plot(all_episodes, likeness_avg_smooth, color=colors['likeness_score'], 
                    linewidth=2, label='Likeness Score (avg)', alpha=0.9)
        
        ax_avg.plot(all_episodes, validation_avg_smooth, color=colors['validation_reward'], 
                    linewidth=2, label='Validation (avg)', alpha=0.9)
        
        # Create second y-axis for selection ratio
        ax_avg2 = ax_avg.twinx()
        ax_avg2.plot(all_episodes, sel_ratio_avg_smooth, color='black', 
                    linewidth=2, label='Selection Ratio (avg)', alpha=0.9, linestyle='--')
        ax_avg2.set_ylabel('Selection Ratio', color='black')
        ax_avg2.tick_params(axis='y', labelcolor='black')
        ax_avg2.set_ylim(0, 1)
        
        # Skip ratio penalty plotting for average
        
        # Add zero line
        ax_avg.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
        
        # Formatting for average plot
        ax_avg.set_title('Average Across All Folds', fontsize=12, fontweight='bold')
        ax_avg.set_xlabel('Episode')
        ax_avg.set_ylabel('Normalized Reward Value (0-1)')
        ax_avg.set_ylim(-0.05, 1.05)
        ax_avg.grid(True, alpha=0.3)
        
        # Combine legends from both axes
        lines1, labels1 = ax_avg.get_legend_handles_labels()
        lines2, labels2 = ax_avg2.get_legend_handles_labels()
        ax_avg.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=8)
        
        # Remove statistics text box as requested
    else:
        # Remove empty subplot if only one fold
        axes[5].remove()
    
    plt.tight_layout()
    plt.savefig('logs/reward_components_progression.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

def create_component_analysis():
    """Create detailed analysis of reward components"""
    fold_data = parse_reward_components()
    
    if not fold_data:
        print("No reward component data found!")
        return
    
    print("=== Reward Components Analysis ===")
    print(f"{'Fold':<6} {'Episodes':<10} {'Final':<12} {'Immediate':<12} {'Shaped':<12} {'Likeness':<12} {'Validation':<12} {'Penalty':<12}")
    print("-" * 90)
    
    for fold, data in fold_data.items():
        components = data['components']
        episodes = len(data['episodes'])
        
        final_mean = np.mean(components['final_reward'])
        immediate_mean = np.mean(components['immediate'])
        shaped_mean = np.mean(components['shaped'])
        likeness_mean = np.mean(components['likeness_score'])
        validation_mean = np.mean(components['validation_reward'])
        penalty_mean = np.mean(components['ratio_penalty'])
        
        print(f"{fold:<6} {episodes:<10} {final_mean:<12.3f} {immediate_mean:<12.3f} {shaped_mean:<12.3f} {likeness_mean:<12.3f} {validation_mean:<12.3f} {penalty_mean:<12.3f}")
    
    # Overall statistics
    all_final = []
    all_immediate = []
    all_penalty = []
    
    for data in fold_data.values():
        all_final.extend(data['components']['final_reward'])
        all_immediate.extend(data['components']['immediate'])
        all_penalty.extend(data['components']['ratio_penalty'])
    
    print("\n=== Overall Statistics ===")
    print(f"Total episodes: {sum(len(data['episodes']) for data in fold_data.values())}")
    print(f"Final reward: {np.mean(all_final):.3f} ± {np.std(all_final):.3f}")
    print(f"Immediate reward: {np.mean(all_immediate):.3f} ± {np.std(all_immediate):.3f}")
    print(f"Ratio penalty: {np.mean(all_penalty):.3f} ± {np.std(all_penalty):.3f}")
    
    # Component contribution analysis
    print("\n=== Component Contribution Analysis ===")
    for fold, data in fold_data.items():
        components = data['components']
        
        # Calculate correlation between components and final reward
        final_rewards = components['final_reward']
        
        print(f"\nFold {fold}:")
        for comp_name in ['disc_mean', 'dcae_mean', 'shaped', 'likeness_score', 'validation_reward']:
            if len(components[comp_name]) > 1:
                correlation = np.corrcoef(final_rewards, components[comp_name])[0, 1]
                print(f"  {comp_name}: correlation = {correlation:.3f}")

if __name__ == "__main__":
    # Create output directory
    os.makedirs("logs", exist_ok=True)
    
    # Generate visualizations and analysis
    print("Creating reward components visualization...")
    create_reward_components_plot()
    
    print("\nGenerating component analysis...")
    create_component_analysis()
    
    print(f"\nVisualization saved to: logs/reward_components_progression.png")
