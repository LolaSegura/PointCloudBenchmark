import argparse
import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

def parse_benchmark_output(file_path):
    """Parse the output from the point cloud benchmark program."""
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Extract input filename
    input_match = re.search(r'Input file: (.+)', content)
    input_file = input_match.group(1) if input_match else "Unknown"
    
    # Extract voxel size
    voxel_match = re.search(r'PCL voxelization \(leaf size = (.+?)\)', content)
    voxel_size = float(voxel_match.group(1)) if voxel_match else 0.0
    
    # Extract PCL voxelization results
    pcl_voxel = {
        'time': float(re.search(r'Running PCL voxelization.*?\n\s+Time: ([\d.]+) ms', content, re.DOTALL).group(1)),
        'memory': float(re.search(r'Running PCL voxelization.*?\n\s+Memory: ([\d.]+) KB', content, re.DOTALL).group(1)),
        'points': int(re.search(r'Running PCL voxelization.*?\n\s+Output points: (\d+)', content, re.DOTALL).group(1))
    }
    
    # Extract Open3D voxelization results
    o3d_voxel = {
        'time': float(re.search(r'Running Open3D voxelization.*?\n\s+Time: ([\d.]+) ms', content, re.DOTALL).group(1)),
        'memory': float(re.search(r'Running Open3D voxelization.*?\n\s+Memory: ([\d.]+) KB', content, re.DOTALL).group(1)),
        'points': int(re.search(r'Running Open3D voxelization.*?\n\s+Output points: (\d+)', content, re.DOTALL).group(1))
    }
    
    # Extract clustering parameters
    pcl_tol_match = re.search(r'PCL clustering \(tolerance = (.+?)\)', content)
    pcl_tolerance = float(pcl_tol_match.group(1)) if pcl_tol_match else 0.0
    
    o3d_eps_match = re.search(r'Open3D clustering \(eps = (.+?)\)', content)
    o3d_eps = float(o3d_eps_match.group(1)) if o3d_eps_match else 0.0
    
    # Extract PCL clustering results
    pcl_cluster = {
        'time': float(re.search(r'Running PCL clustering.*?\n\s+Time: ([\d.]+) ms', content, re.DOTALL).group(1)),
        'memory': float(re.search(r'Running PCL clustering.*?\n\s+Memory: ([\d.]+) KB', content, re.DOTALL).group(1)),
        'clusters': int(re.search(r'Running PCL clustering.*?\n\s+Clusters found: (\d+)', content, re.DOTALL).group(1))
    }
    
    # Extract Open3D clustering results
    o3d_cluster = {
        'time': float(re.search(r'Running Open3D clustering.*?\n\s+Time: ([\d.]+) ms', content, re.DOTALL).group(1)),
        'memory': float(re.search(r'Running Open3D clustering.*?\n\s+Memory: ([\d.]+) KB', content, re.DOTALL).group(1)),
        'clusters': int(re.search(r'Running Open3D clustering.*?\n\s+Clusters found: (\d+)', content, re.DOTALL).group(1))
    }
    
    # Extract ratios from summary if available
    voxel_ratio_match = re.search(r'Voxelization speed ratio \(PCL/Open3D\): ([\d.]+)', content)
    voxel_ratio = float(voxel_ratio_match.group(1)) if voxel_ratio_match else (pcl_voxel['time'] / o3d_voxel['time'] if o3d_voxel['time'] != 0 else 0)
    
    cluster_ratio_match = re.search(r'Clustering speed ratio \(PCL/Open3D\): ([\d.]+)', content)
    cluster_ratio = float(cluster_ratio_match.group(1)) if cluster_ratio_match else (pcl_cluster['time'] / o3d_cluster['time'] if o3d_cluster['time'] != 0 else 0)
    
    return {
        'input_file': input_file,
        'voxel_size': voxel_size,
        'pcl_tolerance': pcl_tolerance,
        'o3d_eps': o3d_eps,
        'pcl_voxel': pcl_voxel,
        'o3d_voxel': o3d_voxel,
        'pcl_cluster': pcl_cluster,
        'o3d_cluster': o3d_cluster,
        'voxel_ratio': voxel_ratio,
        'cluster_ratio': cluster_ratio
    }

def plot_benchmark_results(data, output_prefix=None):
    """Create visualizations of the benchmark results."""
    sns.set_style("whitegrid")
    plt.rcParams.update({'font.size': 12})
    
    # Create a DataFrame for easier plotting
    voxel_df = pd.DataFrame({
        'Library': ['PCL', 'Open3D'],
        'Execution Time (ms)': [data['pcl_voxel']['time'], data['o3d_voxel']['time']],
        'Memory Usage (KB)': [data['pcl_voxel']['memory'], data['o3d_voxel']['memory']],
        'Output Points': [data['pcl_voxel']['points'], data['o3d_voxel']['points']]
    })
    
    cluster_df = pd.DataFrame({
        'Library': ['PCL', 'Open3D'],
        'Execution Time (ms)': [data['pcl_cluster']['time'], data['o3d_cluster']['time']],
        'Memory Usage (KB)': [data['pcl_cluster']['memory'], data['o3d_cluster']['memory']],
        'Clusters Found': [data['pcl_cluster']['clusters'], data['o3d_cluster']['clusters']]
    })
    
    # Plot 1: Voxelization Comparison - Time and Memory
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Time comparison
    sns.barplot(x='Library', y='Execution Time (ms)', data=voxel_df, ax=ax1, palette=['#3498db', '#e74c3c'], legend=True)
    for i, v in enumerate(voxel_df['Execution Time (ms)']):
        ax1.text(i, v + 5, f"{v:.1f}", ha='center')
    ax1.set_title(f'Voxelization Time Comparison (Voxel Size: {data["voxel_size"]})')
    
    # Memory comparison
    sns.barplot(x='Library', y='Memory Usage (KB)', data=voxel_df, ax=ax2, palette=['#3498db', '#e74c3c'], legend=True)
    for i, v in enumerate(voxel_df['Memory Usage (KB)']):
        ax2.text(i, v + 5, f"{v:.1f}", ha='center')
    ax2.set_title('Voxelization Memory Usage Comparison')
    
    plt.suptitle(f'Point Cloud Voxelization Benchmark: PCL vs. Open3D\nInput: {data["input_file"]}', fontsize=16)
    plt.tight_layout()
    
    if output_prefix:
        plt.savefig(f'{output_prefix}_voxelization.png', dpi=300, bbox_inches='tight')
    
    # Plot 2: Clustering Comparison - Time and Memory
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Time comparison
    sns.barplot(x='Library', y='Execution Time (ms)', data=cluster_df, ax=ax1, palette=['#3498db', '#e74c3c'], legend=True)
    for i, v in enumerate(cluster_df['Execution Time (ms)']):
        ax1.text(i, v + 5, f"{v:.1f}", ha='center')
    ax1.set_title(f'Clustering Time Comparison\nPCL tolerance: {data["pcl_tolerance"]}, Open3D eps: {data["o3d_eps"]}')
    
    # Memory comparison
    sns.barplot(x='Library', y='Memory Usage (KB)', data=cluster_df, ax=ax2, palette=['#3498db', '#e74c3c'], legend=True)
    for i, v in enumerate(cluster_df['Memory Usage (KB)']):
        ax2.text(i, v + 5, f"{v:.1f}", ha='center')
    ax2.set_title('Clustering Memory Usage Comparison')
    
    plt.suptitle(f'Point Cloud Clustering Benchmark: PCL vs. Open3D\nInput: {data["input_file"]}', fontsize=16)
    plt.tight_layout()
    
    if output_prefix:
        plt.savefig(f'{output_prefix}_clustering.png', dpi=300, bbox_inches='tight')
    
    # Plot 3: Results Comparison - Output Points and Clusters
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Output points comparison
    sns.barplot(x='Library', y='Output Points', data=voxel_df, ax=ax1, palette=['#3498db', '#e74c3c'], legend=True)
    for i, v in enumerate(voxel_df['Output Points']):
        ax1.text(i, v + 5, f"{v}", ha='center')
    ax1.set_title(f'Voxelization Output Points (Voxel Size: {data["voxel_size"]})')
    
    # Clusters comparison
    sns.barplot(x='Library', y='Clusters Found', data=cluster_df, ax=ax2, palette=['#3498db', '#e74c3c'], legend=True)
    for i, v in enumerate(cluster_df['Clusters Found']):
        ax2.text(i, v + 0.5, f"{v}", ha='center')
    ax2.set_title('Clustering Results Comparison')
    
    plt.suptitle(f'Point Cloud Processing Results: PCL vs. Open3D\nInput: {data["input_file"]}', fontsize=16)
    plt.tight_layout()
    
    if output_prefix:
        plt.savefig(f'{output_prefix}_results.png', dpi=300, bbox_inches='tight')
    
    # Plot 4: Speed Ratio Summary
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ratio_df = pd.DataFrame({
        'Operation': ['Voxelization', 'Clustering'],
        'Speed Ratio (PCL/Open3D)': [data['voxel_ratio'], data['cluster_ratio']]
    })
    
    bars = sns.barplot(x='Operation', y='Speed Ratio (PCL/Open3D)', data=ratio_df, ax=ax, palette=['#2ecc71', '#9b59b6'], legend=True)
    
    # Add horizontal line at y=1 to indicate equal performance
    ax.axhline(y=1, color='red', linestyle='--', alpha=0.7)
    
    # Annotate bars
    for i, v in enumerate(ratio_df['Speed Ratio (PCL/Open3D)']):
        ax.text(i, v + 0.05, f"{v:.2f}x", ha='center')
        # Add interpretation
        if v > 1:
            ax.text(i, v/2, "PCL slower", ha='center', color='white', fontweight='bold')
        elif v < 1:
            ax.text(i, v/2, "Open3D slower", ha='center', color='white', fontweight='bold')
    
    ax.set_title('Performance Ratio: PCL vs. Open3D\nValues > 1 mean PCL is slower, < 1 mean Open3D is slower')
    
    plt.suptitle(f'Point Cloud Library Performance Comparison\nInput: {data["input_file"]}', fontsize=16)
    plt.tight_layout()
    
    if output_prefix:
        plt.savefig(f'{output_prefix}_ratio.png', dpi=300, bbox_inches='tight')
    
    return plt

def main():
    parser = argparse.ArgumentParser(description='Visualize point cloud benchmark results')
    parser.add_argument('input', help='Path to the benchmark output file')
    parser.add_argument('--output', '-o', help='Prefix for output image files', default=None)
    parser.add_argument('--show', '-s', action='store_true', help='Show plots instead of saving to files')
    
    args = parser.parse_args()
    
    try:
        data = parse_benchmark_output(args.input)
        plt = plot_benchmark_results(data, None if args.show else args.output)
        
        if args.show:
            plt.show()
            
        print(f"Benchmark visualization completed successfully!")
        if args.output:
            print(f"Output images saved with prefix: {args.output}")
            
    except Exception as e:
        print(f"Error processing benchmark results: {e}")
        return 1
        
    return 0

if __name__ == "__main__":
    main()
