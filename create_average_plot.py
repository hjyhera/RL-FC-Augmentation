import matplotlib.pyplot as plt
import numpy as np
import json
import glob
import re
import os
from pathlib import Path
from scipy.ndimage import uniform_filter1d
import seaborn as sns

# Set modern style and Times New Roman font
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Set serif font (fallback to available fonts if Times New Roman not available)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'Liberation Serif', 'serif']
plt.rcParams['mathtext.fontset'] = 'dejavuserif'  # For mathematical expressions
plt.rcParams['font.size'] = 12  # Base font size

def parse_reward_components():
    """Parse reward components from selection history JSON files"""
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
                    'fidelity_mean': [],
                    'alignment_mean': [],
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
                        components['fidelity_mean'].append(comp.get('fidelity_mean', 0.0))
                        components['alignment_mean'].append(comp.get('alignment_mean', 0.0))
                        components['likeness_score'].append(comp.get('likeness_score', 0.0))
                        # Use val_normalized instead of validation_reward (which is always 0)
                        components['validation_reward'].append(comp.get('val_normalized', 0.0))
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

def create_average_only_plot():
    """Create only the average across all folds plot with professional styling"""
    
    fold_data = parse_reward_components()
    
    if not fold_data:
        print("No reward component data found!")
        return None
    
    if len(fold_data) <= 1:
        print("Need at least 2 folds to create average plot!")
        return None
    
    # Create figure with professional styling
    fig, ax = plt.subplots(1, 1, figsize=(23, 15))
    fig.patch.set_facecolor('white')
    
    # Modern color palette with better contrast
    colors = {
        'fidelity_mean': '#2E86AB',         # Professional blue
        'alignment_mean': '#A23B72',         # Deep magenta
        'likeness_score': '#F18F01',    # Vibrant orange
        'validation_reward': '#C73E1D', # Strong red
    }
    
    # Calculate average components across all folds with normalization
    component_names = ['fidelity_mean', 'alignment_mean', 'likeness_score', 'validation_reward']
    
    # Collect all data for global normalization
    all_data = {name: [] for name in component_names}
    all_sel_ratios = []
    for fold, data in fold_data.items():
        for name in component_names:
            if name == 'validation_reward':
                # Flip validation values to show increasing trend (negate the negative values)
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
    all_episodes = []
    
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
                            raw_val = -raw_val  # Flip validation to show increasing trend
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
    
    # Plot average components with enhanced smoothing
    window_size = max(15, len(all_episodes) // 20)  # Adaptive smoothing
    
    fidelity_avg_smooth = smooth_data(avg_components['fidelity_mean'], window_size=window_size)
    alignment_avg_smooth = smooth_data(avg_components['alignment_mean'], window_size=window_size)
    likeness_avg_smooth = smooth_data(avg_components['likeness_score'], window_size=window_size)
    validation_avg_smooth = smooth_data(avg_components['validation_reward'], window_size=window_size)
    sel_ratio_avg_smooth = smooth_data(avg_sel_ratios, window_size=window_size)
    
    # Plot reward components with confidence intervals
    fidelity_std_smooth = smooth_data(std_components['fidelity_mean'], window_size=window_size)
    alignment_std_smooth = smooth_data(std_components['alignment_mean'], window_size=window_size)
    likeness_std_smooth = smooth_data(std_components['likeness_score'], window_size=window_size)
    validation_std_smooth = smooth_data(std_components['validation_reward'], window_size=window_size)
    
    # Plot main lines with enhanced styling
    ax.plot(all_episodes, fidelity_avg_smooth, color=colors['fidelity_mean'], 
            linewidth=5.0, label='$r_F$', alpha=0.9, zorder=3)
    ax.fill_between(all_episodes, 
                   np.array(fidelity_avg_smooth) - np.array(fidelity_std_smooth),
                   np.array(fidelity_avg_smooth) + np.array(fidelity_std_smooth),
                   color=colors['fidelity_mean'], alpha=0.0, zorder=1)
    
    ax.plot(all_episodes, alignment_avg_smooth, color=colors['alignment_mean'], 
            linewidth=7.0, label='$r_A$', alpha=0.9, zorder=3)
    ax.fill_between(all_episodes,
                   np.array(alignment_avg_smooth) - np.array(alignment_std_smooth),
                   np.array(alignment_avg_smooth) + np.array(alignment_std_smooth),
                   color=colors['alignment_mean'], alpha=0.0, zorder=1)
    
    ax.plot(all_episodes, likeness_avg_smooth, color=colors['likeness_score'], 
            linewidth=7.0, label='$r_D$', alpha=0.9, zorder=3)
    ax.fill_between(all_episodes,
                   np.array(likeness_avg_smooth) - np.array(likeness_std_smooth),
                   np.array(likeness_avg_smooth) + np.array(likeness_std_smooth),
                   color=colors['likeness_score'], alpha=0.0, zorder=1)
    
    ax.plot(all_episodes, validation_avg_smooth, color=colors['validation_reward'], 
            linewidth=7.0, label='$r_U$', alpha=0.9, zorder=3)
    ax.fill_between(all_episodes,
                   np.array(validation_avg_smooth) - np.array(validation_std_smooth),
                   np.array(validation_avg_smooth) + np.array(validation_std_smooth),
                   color=colors['validation_reward'], alpha=0.0, zorder=1)
    
    # Create second y-axis for selection ratio with enhanced styling
    ax2 = ax.twinx()
    ax2.plot(all_episodes, sel_ratio_avg_smooth, color='#2C3E50', 
            linewidth=10.0, label='$s_R$', alpha=0.8, linestyle='--', zorder=4)
    
    # Add confidence interval for selection ratio
    sel_std_smooth = smooth_data(std_sel_ratios, window_size=window_size)
    ax2.fill_between(all_episodes,
                    np.array(sel_ratio_avg_smooth) - np.array(sel_std_smooth),
                    np.array(sel_ratio_avg_smooth) + np.array(sel_std_smooth),
                    color='#2C3E50', alpha=0.0, zorder=1)
    
    # Enhanced formatting
    ax.set_xlabel('Training Episode', fontsize=38, fontweight='bold', color='#2C3E50')
    ax.set_ylabel('Reward ($r$)', fontsize=38, fontweight='bold', color='#2C3E50')
    ax2.set_ylabel('Selection Ratio ($s_R$)', fontsize=38, fontweight='bold', color='#2C3E50')
    
    # Improved axis styling
    ax.set_ylim(-0.05, 1.05)
    ax2.set_ylim(0, 1)
    
    # Enhanced grid
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.8, color='gray')
    ax.set_axisbelow(True)
    
    # Professional tick styling
    ax.tick_params(axis='both', labelsize=38, colors='#2C3E50', width=1.2)
    ax2.tick_params(axis='y', labelsize=38, colors='#2C3E50', width=1.2)
    
    # Add subtle reference lines
    ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.3, linewidth=1, zorder=0)
    ax2.axhline(y=0.5, color='gray', linestyle=':', alpha=0.3, linewidth=1, zorder=0)
    
    # Enhanced legend with better positioning
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    
    legend = ax.legend(lines1 + lines2, labels1 + labels2, 
                      loc='lower right', fontsize=38, frameon=True, 
                      fancybox=True, shadow=True, framealpha=0.9,
                      edgecolor='#2C3E50', facecolor='white')
    legend.get_frame().set_linewidth(1.2)
    
    # Add subtle title area with metrics summary
    if len(fold_data) > 1:
        final_metrics = {
            'fidelity': fidelity_avg_smooth[-1] if len(fidelity_avg_smooth) > 0 else 0,
            'alignment': alignment_avg_smooth[-1] if len(alignment_avg_smooth) > 0 else 0,
            'likeness': likeness_avg_smooth[-1] if len(likeness_avg_smooth) > 0 else 0,
            'validation': validation_avg_smooth[-1] if len(validation_avg_smooth) > 0 else 0,
            'selection': sel_ratio_avg_smooth[-1] if len(sel_ratio_avg_smooth) > 0 else 0
        }
        
        # Add text box with final metrics
        textstr = f'Final Metrics (Ep {max_episodes}):\n'
        textstr += f'$r_F$: {final_metrics["fidelity"]:.3f} | $r_A$: {final_metrics["alignment"]:.3f}\n'
        textstr += f'$r_D$: {final_metrics["likeness"]:.3f} | $r_U$: {final_metrics["validation"]:.3f}\n'
        textstr += f'$s_R$: {final_metrics["selection"]:.3f}'
        
        props = dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.6, edgecolor='#2C3E50')
        ax.text(0.65, 0.03, textstr, transform=ax.transAxes, fontsize=38,
                verticalalignment='bottom', horizontalalignment='right', bbox=props)
    
    # Enhance spine styling
    for spine in ax.spines.values():
        spine.set_color('#2C3E50')
        spine.set_linewidth(1.2)
    for spine in ax2.spines.values():
        spine.set_color('#2C3E50')
        spine.set_linewidth(1.2)
    
    plt.tight_layout()
    
    # Save with high quality
    output_path = 'logs/average_across_folds_only.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', 
                edgecolor='none', pad_inches=0.2)
    print(f"Enhanced professional plot saved to: {output_path}")
    
    plt.show()
    
    return fig

if __name__ == "__main__":
    # Create output directory
    os.makedirs("logs", exist_ok=True)
    
    # Generate the average plot
    print("Creating average across folds plot...")
    create_average_only_plot()
