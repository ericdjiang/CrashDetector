# sampling a sine wave programmatically
import matplotlib; matplotlib.use("TkAgg")
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.io import wavfile
import matplotlib.gridspec as gridspec
import time

fs, data = wavfile.read('./Data/Crashes/crash_001.wav')  # fs = sample rate (Hz), data is the amplitudes
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