"""This filed defines some commonly used plotting functions used in the notebooks"""



def plot_delta_shield_performance(deltas_list, goal_list, lava_list, truncated_list, 
                                  baseline_goal=None, baseline_lava=None, baseline_truncated=None):
    """
    Plot delta shield performance comparing different threshold values.
    
    Parameters:
    - deltas_list: List of delta threshold values
    - goal_list: List of goal achievement counts for each delta
    - lava_list: List of lava encounter counts for each delta  
    - truncated_list: List of truncated episode counts for each delta
    - baseline_goal: Baseline goal achievement count (no shield)
    - baseline_lava: Baseline lava encounter count (no shield)
    - baseline_truncated: Baseline truncated episode count (no shield)
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Plot lines for delta shield results
    plt.plot(deltas_list, goal_list, 'o-', color='green', linewidth=2.5, markersize=8, label='Reached Goal (Delta Shield)')
    plt.plot(deltas_list, lava_list, 's-', color='red', linewidth=2.5, markersize=8, label='Reached Lava (Delta Shield)')
    plt.plot(deltas_list, truncated_list, '^-', color='orange', linewidth=2.5, markersize=8, label='Truncated (Delta Shield)')
    
    # Add baseline reference lines if provided
    if baseline_goal is not None:
        plt.axhline(y=baseline_goal, color='darkgreen', linestyle='--', linewidth=2, alpha=0.8, label='Baseline Goal (No Shield)')
    if baseline_lava is not None:
        plt.axhline(y=baseline_lava, color='darkred', linestyle='--', linewidth=2, alpha=0.8, label='Baseline Lava (No Shield)')
    if baseline_truncated is not None:
        plt.axhline(y=baseline_truncated, color='darkorange', linestyle='--', linewidth=2, alpha=0.8, label='Baseline Truncated (No Shield)')
    
    # Customize the plot
    plt.xlabel('Delta (Shield Threshold)', fontsize=14, fontweight='bold')
    plt.ylabel('Frequency (out of episodes)', fontsize=14, fontweight='bold')
    plt.title('Delta Shield Performance: Episode Outcomes vs Shield Threshold', fontsize=16, fontweight='bold')
    
    # Add grid for better readability
    plt.grid(True, alpha=0.3)
    
    # Customize legend
    plt.legend(fontsize=12, loc='center right')
    
    # Set axis limits for better visualization
    plt.xlim(-0.05, max(deltas_list) + 0.05)
    max_y = max(max(goal_list), max(lava_list), max(truncated_list))
    if baseline_goal:
        max_y = max(max_y, baseline_goal)
    plt.ylim(0, max_y + 50)
    
    # Add some styling
    plt.tick_params(axis='both', which='major', labelsize=11)
    
    # Add annotations for key insights
    if len(goal_list) > 0:
        min_goal_idx = goal_list.index(min(goal_list))
        max_lava_idx = lava_list.index(max(lava_list))
        
        plt.annotate('Shield can reduce\ngoal achievement', 
                     xy=(deltas_list[min_goal_idx], goal_list[min_goal_idx]), 
                     xytext=(deltas_list[min_goal_idx] - 0.2, goal_list[min_goal_idx] - 100),
                     arrowprops=dict(arrowstyle='->', color='green', alpha=0.7),
                     fontsize=10, ha='center')
        
        plt.annotate('Shield can increase\nlava encounters', 
                     xy=(deltas_list[max_lava_idx], lava_list[max_lava_idx]), 
                     xytext=(deltas_list[max_lava_idx] + 0.2, lava_list[max_lava_idx] + 30),
                     arrowprops=dict(arrowstyle='->', color='red', alpha=0.7),
                     fontsize=10, ha='center')
    
    # Add baseline annotation if provided
    if baseline_goal and baseline_lava:
        mid_delta = deltas_list[len(deltas_list)//2]
        plt.annotate(f'Baseline: {baseline_goal} goals, {baseline_lava} lava', 
                     xy=(mid_delta, baseline_goal), xytext=(mid_delta, baseline_goal + 50),
                     arrowprops=dict(arrowstyle='->', color='darkgreen', alpha=0.7),
                     fontsize=10, ha='center', style='italic')
    
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    print("Delta Shield Performance Summary:")
    print("=" * 50)
    
    if baseline_goal is not None:
        print(f"BASELINE (No Shield): {baseline_goal} goals, {baseline_lava} lava, {baseline_truncated} truncated")
        print()
    
    print("DELTA SHIELD RESULTS:")
    if len(goal_list) > 0:
        best_goal_idx = goal_list.index(max(goal_list))
        worst_lava_idx = lava_list.index(max(lava_list))
        print(f"Best goal achievement: {max(goal_list)} episodes at delta = {deltas_list[best_goal_idx]}")
        print(f"Worst lava encounters: {max(lava_list)} episodes at delta = {deltas_list[worst_lava_idx]}")
        print(f"Average goal achievement: {np.mean(goal_list):.1f} episodes")
        print(f"Average lava encounters: {np.mean(lava_list):.1f} episodes")
        
        if baseline_goal is not None:
            print()
            print("COMPARISON TO BASELINE:")
            print(f"Goal achievement: Baseline = {baseline_goal}, Shield avg = {np.mean(goal_list):.1f} ({np.mean(goal_list) - baseline_goal:+.1f})")
            print(f"Lava encounters: Baseline = {baseline_lava}, Shield avg = {np.mean(lava_list):.1f} ({np.mean(lava_list) - baseline_lava:+.1f})")



