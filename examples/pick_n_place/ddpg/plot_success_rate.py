#!/usr/bin/env python3
"""
Standalone script to plot success rate vs timesteps from training logs.
This script can be used to analyze existing training runs without re-running training.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import argparse

def plot_success_rate_vs_timesteps(log_dir, save_path=None, figsize=(12, 8), env_id="FrankaPushSparse-v0"):
    """
    Plot and save time steps vs success rate from training logs.
    
    Args:
        log_dir (str): Path to the log directory containing tensorboard events
        save_path (str, optional): Path to save the plot. If None, saves to log_dir
        figsize (tuple): Figure size for the plot
        env_id (str): Environment ID for the plot title
    """
    # Find tensorboard event files
    event_files = []
    for file in os.listdir(log_dir):
        if file.startswith('events.out.tfevents'):
            event_files.append(os.path.join(log_dir, file))
    
    if not event_files:
        print("No tensorboard event files found in log directory")
        return
    
    # Use the most recent event file
    event_file = sorted(event_files)[-1]
    print(f"Using event file: {event_file}")
    
    try:
        # Load tensorboard data
        ea = EventAccumulator(event_file)
        ea.Reload()
        
        # Get all available tags
        tags = ea.Tags()
        print(f"Available tags: {tags}")
        
        # Look for success rate related metrics
        success_metrics = []
        timesteps = []
        
        # Common success rate metric names
        possible_success_tags = [
            'eval/success_rate',
            'eval/ep_success_rate', 
            'eval/mean_success_rate',
            'success_rate',
            'ep_success_rate',
            'eval/episode_reward_mean',  # Sometimes success is tracked as reward
            'train/episode_reward_mean'
        ]
        
        # Try to find success rate data
        success_tag = None
        for tag in possible_success_tags:
            if tag in tags['scalars']:
                success_tag = tag
                print(f"Found success rate metric: {tag}")
                break
        
        if success_tag is None:
            print("No success rate metrics found. Available scalar tags:")
            for tag in tags['scalars']:
                print(f"  - {tag}")
            return
        
        # Extract success rate data
        success_events = ea.Scalars(success_tag)
        for event in success_events:
            timesteps.append(event.step)
            success_metrics.append(event.value)
        
        if not success_metrics:
            print("No success rate data found in the logs")
            return
        
        print(f"Found {len(success_metrics)} data points")
        print(f"Timesteps range: {min(timesteps)} to {max(timesteps)}")
        print(f"Success rate range: {min(success_metrics):.3f} to {max(success_metrics):.3f}")
        
        # Create the plot
        plt.figure(figsize=figsize)
        plt.plot(timesteps, success_metrics, 'b-', linewidth=2, marker='o', markersize=4)
        plt.xlabel('Training Timesteps', fontsize=12)
        plt.ylabel('Success Rate', fontsize=12)
        plt.title(f'Success Rate vs Training Timesteps\n{env_id}', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1.0)
        
        # Add some statistics
        if len(success_metrics) > 1:
            final_success = success_metrics[-1]
            max_success = max(success_metrics)
            avg_success = np.mean(success_metrics)
            
            plt.axhline(y=final_success, color='r', linestyle='--', alpha=0.7, 
                       label=f'Final: {final_success:.3f}')
            plt.axhline(y=max_success, color='g', linestyle='--', alpha=0.7, 
                       label=f'Max: {max_success:.3f}')
            plt.axhline(y=avg_success, color='orange', linestyle='--', alpha=0.7, 
                       label=f'Avg: {avg_success:.3f}')
            plt.legend()
        
        # Save the plot
        if save_path is None:
            save_path = os.path.join(log_dir, 'success_rate_vs_timesteps.png')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Success rate plot saved to: {save_path}")
        
        # Also save as CSV for further analysis
        csv_path = save_path.replace('.png', '.csv')
        df = pd.DataFrame({
            'timesteps': timesteps,
            'success_rate': success_metrics
        })
        df.to_csv(csv_path, index=False)
        print(f"Success rate data saved to: {csv_path}")
        
        # Print summary statistics
        print("\n" + "="*50)
        print("SUMMARY STATISTICS")
        print("="*50)
        print(f"Total training timesteps: {max(timesteps):,}")
        print(f"Number of evaluations: {len(success_metrics)}")
        print(f"Final success rate: {final_success:.3f}")
        print(f"Maximum success rate: {max_success:.3f}")
        print(f"Average success rate: {avg_success:.3f}")
        print(f"Success rate improvement: {final_success - success_metrics[0]:.3f}")
        
        plt.show()
        
    except Exception as e:
        print(f"Error reading tensorboard logs: {e}")
        print("Trying alternative method...")
        
        # Alternative: try to read from monitor logs
        try:
            monitor_files = []
            for file in os.listdir(log_dir):
                if file.endswith('.monitor.csv'):
                    monitor_files.append(os.path.join(log_dir, file))
            
            if monitor_files:
                print(f"Found monitor files: {monitor_files}")
                # You can add monitor file parsing here if needed
            else:
                print("No monitor files found either")
                
        except Exception as e2:
            print(f"Alternative method also failed: {e2}")

def main():
    parser = argparse.ArgumentParser(description='Plot success rate vs timesteps from training logs')
    parser.add_argument('--log_dir', type=str, default='./logs/franka_slide_ddpg',
                       help='Path to the log directory containing tensorboard events')
    parser.add_argument('--save_path', type=str, default=None,
                       help='Path to save the plot (optional)')
    parser.add_argument('--env_id', type=str, default='FrankaPushSparse-v0',
                       help='Environment ID for the plot title')
    parser.add_argument('--figsize', type=str, default='12,8',
                       help='Figure size as width,height (e.g., "12,8")')
    
    args = parser.parse_args()
    
    # Parse figsize
    try:
        figsize = tuple(map(int, args.figsize.split(',')))
    except:
        figsize = (12, 8)
    
    # Check if log directory exists
    if not os.path.exists(args.log_dir):
        print(f"Log directory does not exist: {args.log_dir}")
        return
    
    print(f"Analyzing logs in: {args.log_dir}")
    plot_success_rate_vs_timesteps(args.log_dir, args.save_path, figsize, args.env_id)

if __name__ == "__main__":
    main() 