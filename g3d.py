import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse
import os

def load_and_process_tsv(filepath, steps_per_epoch=100):
    """Load TSV log and group continuous steps into epochs for analysis"""
    df = pd.read_csv(filepath, sep='\t')
    
    # Create epoch numbers from step numbers
    df['epoch'] = df['step'] // steps_per_epoch
    
    # Classify problem types
    def classify_problem(row):
        if row['operator'] == '->':
            return 'counting'
        elif row['operator'] == '+':
            return 'addition'
        else:
            return 'unknown'
    
    df['problem_type'] = df.apply(classify_problem, axis=1)
    
    # Calculate correctness
    df['correct'] = (df['predicted'] == df['target']).astype(int)
    
    return df

def calculate_accuracy_by_epoch(df):
    """Calculate accuracy percentage by epoch and problem type"""
    # Group by epoch and problem type, calculate mean accuracy
    accuracy_data = df.groupby(['epoch', 'problem_type'])['correct'].agg(['mean', 'count']).reset_index()
    accuracy_data['accuracy_percent'] = accuracy_data['mean'] * 100
    
    # Filter out epochs with too few samples (less than 5) for reliability
    accuracy_data = accuracy_data[accuracy_data['count'] >= 5]
    
    return accuracy_data

def create_3d_histogram(accuracy_data, output_path=None):
    """Create 3D histogram plot"""
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Prepare data
    epochs = sorted(accuracy_data['epoch'].unique())
    problem_types = ['counting', 'addition']
    
    # Create coordinate arrays
    xpos, ypos = np.meshgrid(epochs, range(len(problem_types)), indexing='ij')
    xpos = xpos.flatten()
    ypos = ypos.flatten()
    zpos = np.zeros_like(xpos)
    
    # Dimensions of bars
    dx = np.ones_like(xpos) * 0.8  # Width in epoch direction
    dy = np.ones_like(ypos) * 0.4  # Width in problem type direction
    
    # Heights (accuracy percentages)
    dz = []
    colors = []
    
    for i, epoch in enumerate(epochs):
        for j, ptype in enumerate(problem_types):
            # Find accuracy for this epoch and problem type
            mask = (accuracy_data['epoch'] == epoch) & (accuracy_data['problem_type'] == ptype)
            if mask.any():
                accuracy = accuracy_data[mask]['accuracy_percent'].iloc[0]
                dz.append(accuracy)
                # Color coding: blue for counting, red for addition
                colors.append('skyblue' if ptype == 'counting' else 'salmon')
            else:
                dz.append(0)  # No data point
                colors.append('lightgray')
    
    dz = np.array(dz)
    
    # Create the 3D bar plot
    ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color=colors, alpha=0.8, edgecolor='black')
    
    # Customize the plot
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Problem Type', fontsize=12)
    ax.set_zlabel('Accuracy (%)', fontsize=12)
    ax.set_title('Learning Progress: Accuracy by Epoch and Problem Type', fontsize=14, pad=20)
    
    # Set y-axis labels
    ax.set_yticks(range(len(problem_types)))
    ax.set_yticklabels(problem_types)
    
    # Set z-axis to 0-100% range
    ax.set_zlim(0, 100)
    
    # Add a legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='skyblue', edgecolor='black', label='Counting'),
        Patch(facecolor='salmon', edgecolor='black', label='Addition')
    ]
    ax.legend(handles=legend_elements, loc='upper left')
    
    # Adjust viewing angle for better visibility
    ax.view_init(elev=25, azim=225)    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {output_path}")
    
    plt.show()

def print_summary_stats(accuracy_data):
    """Print summary statistics"""
    print("\n=== LEARNING PROGRESS SUMMARY ===")
    
    for ptype in ['counting', 'addition']:
        type_data = accuracy_data[accuracy_data['problem_type'] == ptype]
        if len(type_data) > 0:
            print(f"\n{ptype.upper()}:")
            print(f"  Epochs analyzed: {len(type_data)}")
            print(f"  Initial accuracy: {type_data['accuracy_percent'].iloc[0]:.1f}%")
            print(f"  Final accuracy: {type_data['accuracy_percent'].iloc[-1]:.1f}%")
            print(f"  Best accuracy: {type_data['accuracy_percent'].max():.1f}%")
            print(f"  Average accuracy: {type_data['accuracy_percent'].mean():.1f}%")

def main():
    parser = argparse.ArgumentParser(description='Analyze SMM learning progress from TSV logs')
    parser.add_argument('tsv_file', help='Path to the TSV log file')
    parser.add_argument('--steps-per-epoch', type=int, default=100, 
                       help='Number of continuous learning steps to group into one epoch (default: 100)')
    parser.add_argument('--output', help='Output file path for the plot (optional)')
    
    args = parser.parse_args()
    
    # Check if file exists
    if not os.path.exists(args.tsv_file):
        print(f"Error: File '{args.tsv_file}' not found!")
        return
    
    print(f"Loading data from: {args.tsv_file}")
    print(f"Using {args.steps_per_epoch} steps per epoch")
    
    # Process the data
    df = load_and_process_tsv(args.tsv_file, args.steps_per_epoch)
    print(f"Loaded {len(df)} training steps")
    print(f"Problem types found: {df['problem_type'].value_counts().to_dict()}")
    
    # Calculate accuracies
    accuracy_data = calculate_accuracy_by_epoch(df)
    print(f"Calculated accuracy for {len(accuracy_data)} epoch-problem_type combinations")
    
    # Print summary
    print_summary_stats(accuracy_data)
    
    # Create the plot
    print("\nGenerating 3D histogram plot...")
    create_3d_histogram(accuracy_data, args.output)

if __name__ == "__main__":
    main()
    
