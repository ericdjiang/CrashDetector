# sampling a sine wave programmatically
import matplotlib; matplotlib.use("TkAgg")
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.io import wavfile
import matplotlib.gridspec as gridspec
import time
#!/usr/bin/env python
# Implementation of algorithm from https://stackoverflow.com/a/22640362/6029703
import pylab

def plot_energies(file_name):
    fs, data = wavfile.read(file_name)  # fs = sample rate (Hz), data is the amplitudes
    try:
        data = (np.delete(data, (1), axis=1)).transpose()[0]  # delete second channel
    except:
        print("Error processing multiple audio channels")
    
    t = 0.1  # seconds of sampling
    N = int(fs*t)  # total points in chunk
    start_t = 0    # starting time in s
    
    """
    convert start time (s) to x scale time 
    e.g. if the frequency rate is 44100Hz: x_time = 4410 would correspond to the data point at data[4410]
    x_time = 88200 also represents time = 2 seconds in real time
    """
    x_time = int(fs*start_t)
    
    total_time = len(data)/fs  # total time in seconds
    curr_time = 0.0  # temporary value for for loop (s)
    energies = []   # normal array to store squared sum of each 1 sec chunk
    
    # loop through entire data set (not in real time)
    # every .1 seconds, square and sum amplitudes in clip chunk
    while curr_time < total_time:
        clip = data[x_time:x_time + N]
        energies.append(np.sum(np.square(clip, dtype='int64')))
        x_time += N
        curr_time += t
    
    energy_times = np.arange(len(energies))*t  # prepare x values for energies array
    
    gs = gridspec.GridSpec(1, 1)  # prepare graph layout
    fig = plt.figure()
    
    ax1 = fig.add_subplot(gs[0,0])  # row 1, span all columns
    ax1.plot(energy_times, energies, color='k', linewidth="0.5")
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Energy')
    
    plt.show()

def peak_detection_smoothed_zscore_v2(x, lag, threshold, influence):
    '''
    iterative smoothed z-score algorithm
    Implementation of algorithm from https://stackoverflow.com/a/22640362/6029703
    '''
    import numpy as np
    labels = np.zeros(len(x))
    filtered_y = np.array(x)
    avg_filter = np.zeros(len(x))
    std_filter = np.zeros(len(x))
    var_filter = np.zeros(len(x))

    avg_filter[lag - 1] = np.mean(x[0:lag])
    std_filter[lag - 1] = np.std(x[0:lag])
    var_filter[lag - 1] = np.var(x[0:lag])
    for i in range(lag, len(x)):
        if abs(x[i] - avg_filter[i - 1]) > threshold * std_filter[i - 1]:
            if x[i] > avg_filter[i - 1]:
                labels[i] = 1
            else:
                labels[i] = -1
            filtered_y[i] = influence * x[i] + (1 - influence) * filtered_y[i - 1]
        else:
            labels[i] = 0
            filtered_y[i] = x[i]
        # update avg, var, std
        avg_filter[i] = avg_filter[i - 1] + 1. / lag * (filtered_y[i] - filtered_y[i - lag])
        var_filter[i] = var_filter[i - 1] + 1. / lag * ((filtered_y[i] - avg_filter[i - 1]) ** 2 - (
            filtered_y[i - lag] - avg_filter[i - 1]) ** 2 - (filtered_y[i] - filtered_y[i - lag]) ** 2 / lag)
        std_filter[i] = np.sqrt(var_filter[i])

    return dict(signals=labels,
                avgFilter=avg_filter,
                stdFilter=std_filter)
"""    
def detect_peak(file_name):
    fs, data = wavfile.read(file_name)
    lag = 30
    threshold = 5
    influence = 0
    
    # Run algo with settings from above
    result = peak_detection_smoothed_zscore_v2(data[:,0], lag=lag, threshold=threshold, influence=influence)
    
    # Plot result
    pylab.subplot(211)
    pylab.plot(np.arange(1, len(data[:,0])+1), data[:,0])
    
    pylab.plot(np.arange(1, len(data[:,0])+1),
               result["avgFilter"], color="cyan", lw=2)
    
    pylab.plot(np.arange(1, len(data[:,0])+1),
               result["avgFilter"] + threshold * result["stdFilter"], color="green", lw=2)
    
    pylab.plot(np.arange(1, len(data[:,0])+1),
               result["avgFilter"] - threshold * result["stdFilter"], color="green", lw=2)
    
    pylab.subplot(212)
    pylab.step(np.arange(1, len(data[:,0])+1), result["signals"], color="red", lw=2)
    pylab.ylim(-1.5, 1.5)
    """
if __name__ == "__main__":
    num_crashes = 12 # define how many crashes to use
    for k in range(num_crashes):
        plot_energies('../Data/Crashes/crash_{:003d}.wav'.format(k+1))
        #detect_peak('../Data/Crashes/crash_{:003d}.wav'.format(k+1))
        # Settings: lag = 30, threshold = 5, influence = 0
        