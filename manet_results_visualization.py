"""
MANET Opportunistic Routing - Results Visualization
Using collected experimental data for Nodes = [100, 200, 300, 400, 500], Speeds = [20, 25, 30, 35, 40]
"""

import numpy as np
import matplotlib.pyplot as plt

# ============================================================================
# EXPERIMENTAL RESULTS (Collected from runs)
# ============================================================================

node_counts = [100, 200, 300, 400, 500]
speeds = [20, 25, 30, 35, 40]

# PDR Results (Nodes x Speeds)
pdr_results = np.array([
    [0.988, 1.000, 1.000, 1.000, 1.000],  # 100 nodes
    [1.000, 0.994, 0.994, 0.994, 0.988],  # 200 nodes
    [0.996, 0.996, 1.000, 1.000, 1.000],  # 300 nodes
    [1.000, 0.994, 0.997, 1.000, 0.997],  # 400 nodes
    [1.000, 1.000, 0.998, 0.999, 0.998],  # 500 nodes
])

# Delay Results (seconds)
delay_results = np.array([
    [0.500, 0.412, 0.405, 0.565, 0.341],  # 100 nodes
    [0.418, 0.408, 0.527, 0.521, 0.429],  # 200 nodes
    [0.370, 0.319, 0.380, 0.384, 0.384],  # 300 nodes
    [0.274, 0.405, 0.354, 0.318, 0.327],  # 400 nodes
    [0.348, 0.325, 0.310, 0.295, 0.280],  # 500 nodes
])

# Throughput Results (Mbps)
throughput_results = np.array([
    [0.1101, 0.1114, 0.1101, 0.1114, 0.1114],  # 100 nodes
    [0.2228, 0.2215, 0.2215, 0.2215, 0.2202],  # 200 nodes
    [0.3329, 0.3329, 0.3342, 0.3342, 0.3342],  # 300 nodes
    [0.4456, 0.4430, 0.4443, 0.4456, 0.4443],  # 400 nodes
    [0.5571, 0.5571, 0.5550, 0.5560, 0.5545],  # 500 nodes
])

# Hop Count Results
hops_results = np.array([
    [1.99, 1.84, 1.81, 1.94, 1.78],  # 100 nodes
    [1.78, 1.82, 2.04, 1.99, 1.88],  # 200 nodes
    [1.71, 1.67, 1.71, 1.75, 1.76],  # 300 nodes
    [1.61, 1.78, 1.77, 1.65, 1.72],  # 400 nodes
    [1.67, 1.66, 1.62, 1.58, 1.55],  # 500 nodes
])

def create_line_plots():
    """Create line plots for all metrics vs number of nodes."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    plt.subplots_adjust(hspace=0.35, wspace=0.3, top=0.88)
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(speeds)))
    markers = ['o', 's', '^', 'D', 'v']
    
    # PDR vs Nodes
    ax1 = axes[0, 0]
    for j, speed in enumerate(speeds):
        ax1.plot(node_counts, pdr_results[:, j], 
                marker=markers[j], color=colors[j], linewidth=2, markersize=8,
                label=f'{speed} m/s')
    ax1.set_xlabel('Number of Nodes', fontsize=11)
    ax1.set_ylabel('Packet Delivery Ratio', fontsize=11)
    ax1.set_title('PDR vs Number of Nodes', fontsize=12, fontweight='bold', pad=10)
    ax1.legend(title='Speed', fontsize=8, loc='lower right')
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.set_ylim(0.95, 1.01)
    
    # Delay vs Nodes
    ax2 = axes[0, 1]
    for j, speed in enumerate(speeds):
        ax2.plot(node_counts, delay_results[:, j], 
                marker=markers[j], color=colors[j], linewidth=2, markersize=8,
                label=f'{speed} m/s')
    ax2.set_xlabel('Number of Nodes', fontsize=11)
    ax2.set_ylabel('Average Delay (seconds)', fontsize=11)
    ax2.set_title('Delay vs Number of Nodes', fontsize=12, fontweight='bold', pad=10)
    ax2.legend(title='Speed', fontsize=8)
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    # Throughput vs Nodes
    ax3 = axes[1, 0]
    for j, speed in enumerate(speeds):
        ax3.plot(node_counts, throughput_results[:, j], 
                marker=markers[j], color=colors[j], linewidth=2, markersize=8,
                label=f'{speed} m/s')
    ax3.set_xlabel('Number of Nodes', fontsize=11)
    ax3.set_ylabel('Throughput (Mbps)', fontsize=11)
    ax3.set_title('Throughput vs Number of Nodes', fontsize=12, fontweight='bold', pad=10)
    ax3.legend(title='Speed', fontsize=8)
    ax3.grid(True, linestyle='--', alpha=0.7)
    
    # Hops vs Nodes
    ax4 = axes[1, 1]
    for j, speed in enumerate(speeds):
        ax4.plot(node_counts, hops_results[:, j], 
                marker=markers[j], color=colors[j], linewidth=2, markersize=8,
                label=f'{speed} m/s')
    ax4.set_xlabel('Number of Nodes', fontsize=11)
    ax4.set_ylabel('Average Hop Count', fontsize=11)
    ax4.set_title('Hop Count vs Number of Nodes', fontsize=12, fontweight='bold', pad=10)
    ax4.legend(title='Speed', fontsize=8)
    ax4.grid(True, linestyle='--', alpha=0.7)
    
    fig.suptitle('MANET Opportunistic Routing Performance (TGNN + Multi-Agent)', 
                fontsize=14, fontweight='bold', y=0.98)
    plt.savefig('manet_line_plots.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Line plots saved to 'manet_line_plots.png'")


def create_heatmaps():
    """Create heatmaps for PDR and Throughput."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    plt.subplots_adjust(hspace=0.4, wspace=0.4, top=0.90, bottom=0.08, left=0.08, right=0.95)
    
    # PDR Heatmap
    ax1 = axes[0, 0]
    im1 = ax1.imshow(pdr_results, cmap='RdYlGn', aspect='auto', vmin=0.95, vmax=1.0)
    ax1.set_xticks(range(len(speeds)))
    ax1.set_xticklabels(speeds)
    ax1.set_yticks(range(len(node_counts)))
    ax1.set_yticklabels(node_counts)
    ax1.set_xlabel('Speed (m/s)', fontsize=10)
    ax1.set_ylabel('Number of Nodes', fontsize=10)
    ax1.set_title('PDR Heatmap', fontsize=11, fontweight='bold', pad=8)
    cbar1 = plt.colorbar(im1, ax=ax1, shrink=0.8)
    cbar1.set_label('PDR', fontsize=9)
    for i in range(len(node_counts)):
        for j in range(len(speeds)):
            ax1.text(j, i, f'{pdr_results[i, j]:.3f}',
                    ha='center', va='center', color='black', fontsize=9)
    
    # Delay Heatmap
    ax2 = axes[0, 1]
    im2 = ax2.imshow(delay_results, cmap='RdYlGn_r', aspect='auto')
    ax2.set_xticks(range(len(speeds)))
    ax2.set_xticklabels(speeds)
    ax2.set_yticks(range(len(node_counts)))
    ax2.set_yticklabels(node_counts)
    ax2.set_xlabel('Speed (m/s)', fontsize=10)
    ax2.set_ylabel('Number of Nodes', fontsize=10)
    ax2.set_title('Delay Heatmap (seconds)', fontsize=11, fontweight='bold', pad=8)
    cbar2 = plt.colorbar(im2, ax=ax2, shrink=0.8)
    cbar2.set_label('Delay (s)', fontsize=9)
    for i in range(len(node_counts)):
        for j in range(len(speeds)):
            ax2.text(j, i, f'{delay_results[i, j]:.3f}',
                    ha='center', va='center', color='black', fontsize=9)
    
    # Throughput Heatmap
    ax3 = axes[1, 0]
    im3 = ax3.imshow(throughput_results, cmap='YlOrRd', aspect='auto')
    ax3.set_xticks(range(len(speeds)))
    ax3.set_xticklabels(speeds)
    ax3.set_yticks(range(len(node_counts)))
    ax3.set_yticklabels(node_counts)
    ax3.set_xlabel('Speed (m/s)', fontsize=10)
    ax3.set_ylabel('Number of Nodes', fontsize=10)
    ax3.set_title('Throughput Heatmap (Mbps)', fontsize=11, fontweight='bold', pad=8)
    cbar3 = plt.colorbar(im3, ax=ax3, shrink=0.8)
    cbar3.set_label('Throughput', fontsize=9)
    for i in range(len(node_counts)):
        for j in range(len(speeds)):
            ax3.text(j, i, f'{throughput_results[i, j]:.3f}',
                    ha='center', va='center', color='black', fontsize=9)
    
    # Hops Heatmap
    ax4 = axes[1, 1]
    im4 = ax4.imshow(hops_results, cmap='Blues', aspect='auto')
    ax4.set_xticks(range(len(speeds)))
    ax4.set_xticklabels(speeds)
    ax4.set_yticks(range(len(node_counts)))
    ax4.set_yticklabels(node_counts)
    ax4.set_xlabel('Speed (m/s)', fontsize=10)
    ax4.set_ylabel('Number of Nodes', fontsize=10)
    ax4.set_title('Hop Count Heatmap', fontsize=11, fontweight='bold', pad=8)
    cbar4 = plt.colorbar(im4, ax=ax4, shrink=0.8)
    cbar4.set_label('Hops', fontsize=9)
    for i in range(len(node_counts)):
        for j in range(len(speeds)):
            ax4.text(j, i, f'{hops_results[i, j]:.2f}',
                    ha='center', va='center', color='black', fontsize=9)
    
    fig.suptitle('MANET Performance Heatmaps', fontsize=13, fontweight='bold', y=0.98)
    plt.savefig('manet_heatmaps.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Heatmaps saved to 'manet_heatmaps.png'")


def create_bar_charts():
    """Create grouped bar charts."""
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    plt.subplots_adjust(hspace=0.35, wspace=0.3, top=0.90, bottom=0.08)
    x = np.arange(len(node_counts))
    width = 0.15
    colors = ['#2ecc71', '#3498db', '#9b59b6', '#e74c3c', '#f39c12']
    
    metrics = [
        (pdr_results, 'Packet Delivery Ratio', axes[0, 0]),
        (delay_results, 'Average Delay (s)', axes[0, 1]),
        (throughput_results, 'Throughput (Mbps)', axes[1, 0]),
        (hops_results, 'Average Hop Count', axes[1, 1])
    ]
    
    for data, metric_name, ax in metrics:
        for j, speed in enumerate(speeds):
            offset = (j - len(speeds)/2 + 0.5) * width
            bars = ax.bar(x + offset, data[:, j], width, 
                         label=f'{speed} m/s', color=colors[j], 
                         edgecolor='black', linewidth=0.5)
        
        ax.set_xlabel('Number of Nodes', fontsize=10)
        ax.set_ylabel(metric_name, fontsize=10)
        ax.set_title(f'{metric_name} by Node Count and Speed', fontsize=11, fontweight='bold', pad=10)
        ax.set_xticks(x)
        ax.set_xticklabels(node_counts)
        ax.legend(title='Speed', fontsize=8, loc='best')
        ax.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    fig.suptitle('MANET Performance: Grouped Bar Charts', fontsize=13, fontweight='bold', y=0.98)
    plt.savefig('manet_bar_charts.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Bar charts saved to 'manet_bar_charts.png'")


def create_speed_comparison():
    """Create plots comparing metrics across different speeds."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    plt.subplots_adjust(hspace=0.35, wspace=0.3, top=0.88)
    
    colors = plt.cm.plasma(np.linspace(0, 1, len(node_counts)))
    markers = ['o', 's', '^', 'D', 'v']
    
    # PDR vs Speed
    ax1 = axes[0, 0]
    for i, n_nodes in enumerate(node_counts):
        ax1.plot(speeds, pdr_results[i, :], 
                marker=markers[i], color=colors[i], linewidth=2, markersize=8,
                label=f'{n_nodes} nodes')
    ax1.set_xlabel('Node Speed (m/s)', fontsize=11)
    ax1.set_ylabel('Packet Delivery Ratio', fontsize=11)
    ax1.set_title('PDR vs Node Speed', fontsize=12, fontweight='bold', pad=10)
    ax1.legend(title='Nodes', fontsize=8, loc='lower right')
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.set_ylim(0.95, 1.01)
    
    # Delay vs Speed
    ax2 = axes[0, 1]
    for i, n_nodes in enumerate(node_counts):
        ax2.plot(speeds, delay_results[i, :], 
                marker=markers[i], color=colors[i], linewidth=2, markersize=8,
                label=f'{n_nodes} nodes')
    ax2.set_xlabel('Node Speed (m/s)', fontsize=11)
    ax2.set_ylabel('Average Delay (seconds)', fontsize=11)
    ax2.set_title('Delay vs Node Speed', fontsize=12, fontweight='bold', pad=10)
    ax2.legend(title='Nodes', fontsize=8)
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    # Throughput vs Speed
    ax3 = axes[1, 0]
    for i, n_nodes in enumerate(node_counts):
        ax3.plot(speeds, throughput_results[i, :], 
                marker=markers[i], color=colors[i], linewidth=2, markersize=8,
                label=f'{n_nodes} nodes')
    ax3.set_xlabel('Node Speed (m/s)', fontsize=11)
    ax3.set_ylabel('Throughput (Mbps)', fontsize=11)
    ax3.set_title('Throughput vs Node Speed', fontsize=12, fontweight='bold', pad=10)
    ax3.legend(title='Nodes', fontsize=8)
    ax3.grid(True, linestyle='--', alpha=0.7)
    
    # Hops vs Speed
    ax4 = axes[1, 1]
    for i, n_nodes in enumerate(node_counts):
        ax4.plot(speeds, hops_results[i, :], 
                marker=markers[i], color=colors[i], linewidth=2, markersize=8,
                label=f'{n_nodes} nodes')
    ax4.set_xlabel('Node Speed (m/s)', fontsize=11)
    ax4.set_ylabel('Average Hop Count', fontsize=11)
    ax4.set_title('Hop Count vs Node Speed', fontsize=12, fontweight='bold', pad=10)
    ax4.legend(title='Nodes', fontsize=8)
    ax4.grid(True, linestyle='--', alpha=0.7)
    
    fig.suptitle('MANET Performance vs Node Speed (TGNN + Multi-Agent Opportunistic Routing)', 
                fontsize=13, fontweight='bold', y=0.98)
    plt.savefig('manet_speed_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Speed comparison plots saved to 'manet_speed_comparison.png'")


def print_summary_table():
    """Print formatted summary tables."""
    print("\n" + "=" * 90)
    print("EXPERIMENT RESULTS SUMMARY")
    print("=" * 90)
    
    print("\n--- Packet Delivery Ratio (PDR) ---")
    print(f"{'Nodes':<10}", end="")
    for speed in speeds:
        print(f"{speed} m/s".center(12), end="")
    print()
    print("-" * 70)
    for i, n_nodes in enumerate(node_counts):
        print(f"{n_nodes:<10}", end="")
        for j in range(len(speeds)):
            print(f"{pdr_results[i, j]:.3f}".center(12), end="")
        print()
    
    print("\n--- Average Delay (seconds) ---")
    print(f"{'Nodes':<10}", end="")
    for speed in speeds:
        print(f"{speed} m/s".center(12), end="")
    print()
    print("-" * 70)
    for i, n_nodes in enumerate(node_counts):
        print(f"{n_nodes:<10}", end="")
        for j in range(len(speeds)):
            print(f"{delay_results[i, j]:.3f}".center(12), end="")
        print()
    
    print("\n--- Throughput (Mbps) ---")
    print(f"{'Nodes':<10}", end="")
    for speed in speeds:
        print(f"{speed} m/s".center(12), end="")
    print()
    print("-" * 70)
    for i, n_nodes in enumerate(node_counts):
        print(f"{n_nodes:<10}", end="")
        for j in range(len(speeds)):
            print(f"{throughput_results[i, j]:.4f}".center(12), end="")
        print()
    
    print("\n--- Average Hop Count ---")
    print(f"{'Nodes':<10}", end="")
    for speed in speeds:
        print(f"{speed} m/s".center(12), end="")
    print()
    print("-" * 70)
    for i, n_nodes in enumerate(node_counts):
        print(f"{n_nodes:<10}", end="")
        for j in range(len(speeds)):
            print(f"{hops_results[i, j]:.2f}".center(12), end="")
        print()
    
    print("\n" + "=" * 90)


if __name__ == "__main__":
    print("MANET Opportunistic Routing - Experimental Results Visualization")
    print("=" * 60)
    print(f"Configurations: Nodes = {node_counts}, Speeds = {speeds}")
    print("-" * 60)
    
    # Print summary table
    print_summary_table()
    
    # Create all visualizations
    create_line_plots()
    create_heatmaps()
    create_bar_charts()
    create_speed_comparison()
    
    print("\n" + "=" * 60)
    print("All visualizations generated successfully!")
    print("Files saved:")
    print("  - manet_line_plots.png")
    print("  - manet_heatmaps.png")
    print("  - manet_bar_charts.png")
    print("  - manet_speed_comparison.png")
    print("=" * 60)
