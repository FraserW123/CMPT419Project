import ast
import matplotlib.pyplot as plt
import matplotlib as mpl

def plot_landmarks_from_row(row):
    """Visualize landmarks from a CSV row"""
    # Configure plot
    mpl.rcParams['figure.figsize'] = (8, 8)
    mpl.rcParams['axes.xmargin'] = 0
    mpl.rcParams['axes.ymargin'] = 0
    
    # Parse data
    landmarks = ast.literal_eval(row['landmarks'])
    gesture = row['gesture']
    timestamp = row['timestamp']
    
    # Create figure
    fig, ax = plt.subplots()
    ax.invert_yaxis()  # MediaPipe coordinates have Y increasing downward
    
    # Plot points
    x = [lm[0] for lm in landmarks]
    y = [lm[1] for lm in landmarks]
    ax.scatter(x, y, s=10, c='blue')
    
    # Add connections (simplified pose connections)
    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),    # Face
        (1, 5), (5, 6), (6, 7), (7, 8),    # Left arm
        (1, 9), (9, 10), (10, 11), (11, 12), # Right arm
        (11, 13), (13, 15),                 # Right leg
        (12, 14), (14, 16)                  # Left leg
    ]
    
    for connection in connections:
        ax.plot(
            [x[connection[0]], x[connection[1]]],  # Fixed parenthesis
            [y[connection[0]], y[connection[1]]],  # Fixed parenthesis
            'r-'
        )
    
    # Add labels
    plt.title(f"Gesture: {gesture}\n{timestamp}")
    plt.xlabel("X (normalized)")
    plt.ylabel("Y (normalized)")
    plt.show()

# Example usage:
if __name__ == "__main__":
    import pandas as pd
    
    # Load sample data
    df = pd.read_csv('gesture_dataset.csv')
    
    # Plot first entries
    for _, row in df.head(1).iterrows():
        plot_landmarks_from_row(row)