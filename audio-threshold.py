
# sampling a sine wave programmatically
import matplotlib; matplotlib.use("TkAgg")
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.io import wavfile
import matplotlib.gridspec as gridspec
import time

fs, data = wavfile.read('./bridge-2.wav')  # fs = sample rate
data = (np.delete(data, (1), axis=1)).transpose()[0]  # delete second channel

t = 0.1  # seconds of sampling
N = int(fs*t)  # total points in chunk
start_t = 0
start = int(fs*start_t)

clip = data[start:start+int(N)]
times = np.arange(len(clip))/float(fs)

full_clip = data[:]
full_times = np.arange(len(full_clip))/float(fs)

"""
print("Chunk", N)
print("Clip:", clip.ndim)
print("Times:", len(times))
"""

total_time = len(full_times)/fs
curr_time = 0.0
energies = []

while curr_time < total_time:
    clip = data[start:start + N]
    energies.append(np.sum(np.square(clip, dtype='int64')))
    start += N
    curr_time += t

energy_times = np.arange(len(energies))*t

gs = gridspec.GridSpec(1, 1)
fig = plt.figure()

ax1 = fig.add_subplot(gs[0,0])  # row 1, span all columns
ax1.plot(energy_times, energies, color='k', linewidth="0.5")
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Amplitude')

plt.show()