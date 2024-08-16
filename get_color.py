import numpy as np

def get_colors():
    colors = np.zeros((31, 3))  # Preallocate an array for 31 colors with 3 RGB components
    colors[0, :] = [0, 0, 1]  # Start with the color blue
    
    i = 0
    while i < 10:
        colors[i + 1, :] = [colors[i, 0] + 0.1, 0, colors[i, 2] - 0.1]
        i += 1
    
    while i < 20:
        colors[i + 1, :] = [colors[i, 0] - 0.1, colors[i, 1] + 0.1, 0]
        i += 1
    
    while i < 30:
        colors[i + 1, :] = [0, colors[i, 1] - 0.1, colors[i, 2] + 0.1]
        i += 1

    return colors
