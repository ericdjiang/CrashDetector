
# sampling a sine wave programmatically
import matplotlib; matplotlib.use("TkAgg")
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.io import wavfile
import matplotlib.gridspec as gridspec
import time


plt.style.use('ggplot')

# signal information
# sampling information

"""freq = 100 # in hertz, the desired natural frequency
omega = 2*np.pi*freq # angular frequency for sine waves

t_vec = np.arange(N)*T # time vector for plotting

print(t_vec[len(t_vec)-1])
y = np.sin(omega*t_vec)

plt.plot(t_vec,y)
plt.show()

# fourier transform and frequency domain
#
Y_k = np.fft.fft(y)[0:int(N/2)]/N # FFT function from numpy
Y_k[1:] = 2*Y_k[1:] # need to take the single-sided spectrum only
Pxx = np.abs(Y_k) # be sure to get rid of imaginary part

f = Fs*np.arange((N/2))/N # frequency vector

# plotting
fig,ax = plt.subplots()
plt.plot(f,Pxx,linewidth=1)
ax.set_xscale('log')
ax.set_yscale('log')
plt.ylabel('Amplitude')
plt.xlabel('Frequency [Hz]')
plt.show()"""

fs, data = wavfile.read('./bridge-2.wav') #fs = samplerate

#https://gist.github.com/leouieda/9043213
t = 0.1 # seconds of sampling
N = fs*t # total points in signal
start_t = 12
start = int(fs*start_t)

clip = data[start:start+int(N)]
times = np.arange(len(clip))/float(fs)

print("Chunk", N)
print("Clip:", clip.ndim)
print("Times:", len(times))

gs = gridspec.GridSpec(2, 2)
fig = plt.figure()

#need to try clip[:,0] since clip might have only 1 channel (like 1000Hz tone)
ax1 = fig.add_subplot(gs[0,0]) # row 0, column 0
line1, = ax1.plot(times, clip[:,0], color='k', linewidth="0.5")
ax1.set_xlim(times[0], times[-1])
ax1.set_ylim(-10**4, 10**4)
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Amplitude')


Y_k = np.fft.fft(clip)[0:int(N/2)]/N # FFT function from numpy
Y_k[1:] = 2*Y_k[1:] # need to take the single-sided spectrum only
Pxx = np.abs(Y_k) # be sure to get rid of imaginary part

f = fs*np.arange((N/2))/N # frequency vector

# plotting
ax2 = fig.add_subplot(gs[0,1]) # row 0, column 0
line2, = ax2.plot(f,Pxx[:,0],color='k', linewidth="0.5")
ax2.set_ylabel('Amplitude')
ax2.set_xlabel('Frequency [Hz]')

full_clip = data[:]
full_times = np.arange(len(full_clip))/float(fs)

#need to try clip[:,0] since clip might have only 1 channel (like 1000Hz tone)
ax3 = fig.add_subplot(gs[1, :]) # row 1, span all columns
ax3.plot(full_times, full_clip[:,0], color='k', linewidth="0.5")
ax3.set_xlim(full_times[0], full_times[-1])
ax3.set_xlabel('Time (s)')
ax3.set_ylabel('Amplitude')

line3, = ax3.plot([start_t, start_t], [-10**4.5, 10**4.5])

prev_energies = 10000.
def animate(i):
    global start, clip, times, data, Pxx, f, prev_energies
    clip = data[start:start + int(N)]
    times = np.arange(start, start+int(N))/float(fs)
    line1.set_data(times,clip[:,0])
    ax1.set_xlim(times[0], times[-1])
    start += int(N)

    Y_k = np.fft.fft(clip)[0:int(N / 2)] / N  # FFT function from numpy
    Y_k[1:] = 2 * Y_k[1:]  # need to take the single-sided spectrum only
    Pxx = np.abs(Y_k)  # be sure to get rid of imaginary part

    print(np.max(Pxx))

    f = fs * np.arange((N / 2)) / N  # frequency vector
    line2.set_data(f, Pxx[:,0])

    line3.set_xdata([times[0], times[0]])
    #print(start/fs, times[0], times[-1])

    energies = (np.delete(clip, (1), axis = 1)).transpose()
    energies = np.square(energies, dtype = 'int64')
    energies = np.sum(energies)

    if (energies/prev_energies>10):
        print("crash")
    prev_energies = energies
    return line1, line2, line3,

ani = animation.FuncAnimation(
    fig, animate,interval=t*1000, blit=True) #to change time frame
#https://stackoverflow.com/questions/44594887/how-to-update-plot-title-with-matplotlib-using-animation for blitting crap

plt.show()


end_time = time.time() + 350
start_time = time.time()


"""
while time.time() < end_time:
  print(1.0 - ((time.time() - start_time) % 1.0))
  time.sleep(1.0 - ((time.time() - start_time) % 1.0))


while time.time() < t_end:
    time.sleep(1)

    diff = time.time() - start_time
    print(diff)
    start += int(diff/t)
    clip = data[start:start + int(N)]
    line1.set_ydata(clip[:, 0])
    ax1.set_xlim(times[0], times[-1])

    print(clip[1100])
    #line1.set_ydata(np.sin(x + phase))
    fig.canvas.draw()
    fig.canvas.flush_events()
"""
