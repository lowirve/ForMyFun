"""

"""

import numpy as np
import matplotlib.pyplot as plt

def image(E):
    fig, ax = plt.subplots()
    ax0 = ax.pcolormesh(np.abs(E**2))
    fig.colorbar(ax0, ax=ax)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    
    plt.show()