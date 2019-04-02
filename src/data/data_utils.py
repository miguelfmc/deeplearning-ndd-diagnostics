""" Useful functions to interact with the gaitndd data
"""

import os
import random
import numpy as np
import scipy.signal as sgn
import matplotlib.pyplot as plt
import wfdb

def plot_segment(name, fs=300):
    in_dir = os.path.join('data', 'interim', 'normalized-signals')

    signal = np.load(os.path.join(in_dir, name))

    # plot signal segment
    fig = plt.figure(figsize=(16,10))
    ax = plt.axes()
    ax.set_xlabel('time (s)')
    ax.set_ylabel('norm. sensor voltage')
    ax.plot(np.linspace(0, len(signal) / fs, len(signal)), signal)
    
    # save image to image directory
    plt.savefig(os.path.join('img', name + '.png'), bbox_inches='tight')

