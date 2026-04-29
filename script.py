import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# 1. Setup Parameters
S0 = 100          # Initial Price
K = 105           # Strike Price
T = 1.0           # Time to Expiry (1 year)
r = 0.05          # Risk-free rate
sigma = 0.25      # Volatility
n_paths = 50      # Number of paths to animate (keep low for performance)
n_steps = 100     # Time steps in the chart
dt = T / n_steps  # Time increment

# 2. Generate Brownian Motion Paths
# We create a 2D array: each row is a path, each column is a time step
paths = np.zeros((n_paths, n_steps + 1))
paths[:, 0] = S0

for t in range(1, n_steps + 1):
    z = np.random.standard_normal(n_paths)
    paths[:, t] = paths[:, t-1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z)

# 3. Setup the Figure for Animation
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={'width_ratios': [3, 1]})
plt.subplots_adjust(wspace=0.3)

# Chart 1: The Price Paths
ax1.set_xlim(0, n_steps)
ax1.set_ylim(np.min(paths) * 0.9, np.max(paths) * 1.1)
ax1.axhline(K, color='red', linestyle='--', label=f'Strike Price ({K})')
ax1.set_title("Monte Carlo Price Path Evolution")
ax1.set_xlabel("Time Steps")
ax1.set_ylabel("Price")
lines = [ax1.plot([], [], lw=1, alpha=0.6)[0] for _ in range(n_paths)]

# Chart 2: The Distribution (Histogram)
ax2.set_ylim(ax1.get_ylim())
ax2.set_title("Price Distribution")
ax2.set_xticks([])

# 4. The Animation Function
def update(frame):
    # 1. Update the line paths
    for i, line in enumerate(lines):
        line.set_data(np.arange(frame), paths[i, :frame])
    
    # 2. Update the distribution in real-time
    if frame > 1:
        ax2.cla() # Clear the histogram axis
        ax2.set_ylim(ax1.get_ylim()) # Keep the scales synced
        ax2.set_title("Current Distribution")
        ax2.set_xticks([])
        
        # Get the prices of all paths at the current 'frame'
        current_prices = paths[:, frame-1]
        
        # Draw the histogram
        ax2.hist(current_prices, orientation='horizontal', 
                 bins=15, color='skyblue', edgecolor='black', alpha=0.7)
        
        # Keep the strike price visible
        ax2.axhline(K, color='red', linestyle='--')

    return lines

# Note: Set blit=False when clearing axes in an animation
ani = FuncAnimation(fig, update, frames=n_steps + 1, interval=30, blit=False, repeat=False)

plt.show()
