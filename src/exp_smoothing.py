from sklearn.neighbors.kde import KernelDensity
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
from scipy import interpolate

import pandas as pd

def smooth_plot_strokes_list(strokes_list, name, plot_indicator):
    """ Calls both the smoothing and the plotting of the smoothing (if plot_indicator=True)"""
    strokes_list = strokes_list.values
    smoothed_strokes = resolution_smoothing(strokes_list)
    if plot_indicator:
        plot_strokes_list(strokes_list, smoothed_strokes, name)
    return smoothed_strokes

def resolution_smoothing(strokes_list):
    """
    Loops through the measurements and if it finds an resolution irregularity (value
    before and after current value are the same) it takes the average of the current value
    and this reoccurring value and replaces the current value by it.
    :param strokes_list: list of stroke rate measurements
    :return: smoothed list of stroke measurements
    """
    smoothed_strokes_list = []
    for index, value in enumerate(strokes_list):
        if index > 0 and index < (len(strokes_list)-1):
            if strokes_list[index-1] == strokes_list[index+1]:
                value = np.mean([value,strokes_list[index+1]])
        smoothed_strokes_list.append(value)
    return smoothed_strokes_list

def plot_strokes_list(strokes_list, smoothed_strokes, name):
    """
    Creates and saves a plot containing both the original measurements and the smoothed measurements, to be able to
    compare them
    :param strokes_list: the list containing the original stroke measurements
    :param smoothed_strokes: the list containing the smoothed stroke measurements
    :param name: consists of 'year', 'countries', 'contest', 'round' and 'boattype'
    """
    distances = [50 * x for x in range(1, 41)]
    colors = mpl.cm.rainbow(np.linspace(0, 1, 2))
    fig, ax = plt.subplots()
    ax.plot(distances, strokes_list, label='original', color=colors[0])
    ax.plot(distances, smoothed_strokes, label='smoothed', color=colors[1])
    plt.ylabel('gradient (strokes/minute)')
    plt.xlabel('distance (meters)')
    # Create a legend
    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    # Put a legend to the right of the current axis
    legend = ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), shadow=True)
    # legend = ax.legend(loc='upper right', shadow=True)
    frame = legend.get_frame()
    frame.set_facecolor('0.90')
    for label in legend.get_texts():
        label.set_fontsize('large')
    for label in legend.get_lines():
        label.set_linewidth(1.5)  # the legend line width
    # Saves the figure in the smoothed_strokes directory in the figures folder
    plt.savefig('../figures/smoothed_strokes/' + name + '.png')





