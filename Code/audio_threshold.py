# sampling a sine wave programmatically
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
#!/usr/bin/env python
# Implementation of algorithm from https://stackoverflow.com/a/22640362/6029703
import os
from matplotlib.backends.backend_pdf import PdfPages
import datetime

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
    crashes = []

    avg_filter[lag - 1] = np.mean(x[0:lag])
    std_filter[lag - 1] = np.std(x[0:lag])
    var_filter[lag - 1] = np.var(x[0:lag])
    for i in range(lag, len(x)):
        if abs(x[i] - avg_filter[i - 1]) > threshold * std_filter[i - 1]:
            if x[i] > avg_filter[i - 1]:
                labels[i] = 1
                crashes += [i]
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
                stdFilter=std_filter,
                crashes=crashes)


def detect_peak(file_name, t=0.1, start_t=0, lag=30, threshold=6,
                influence=0.5, print_pdf='', tick_dist=60.0):
    fs, data = wavfile.read(file_name)
    try:
        data = (np.delete(data, (1), axis=1)).transpose()[0]  # delete second channel
    except:
        pass
        #print("Error processing multiple audio channels")
        
    x_time = int(fs*start_t)
    """
    convert start time (s) to x scale time 
    e.g. if the frequency rate is 44100Hz: x_time = 4410 would correspond to the data point at data[4410]
    x_time = 88200 also represents time = 2 seconds in real time
    """
    N = int(fs*t)  # total points in chunk
    
    total_time = len(data)/fs  # total time in seconds
    curr_time = 0.0  # temporary value for loop (s)
    energies = []   # normal array to store squared sum of each 1 sec chunk
    
    # loop through entire data set (not in real time)
    # every .1 seconds, square and sum amplitudes in clip chunk
    while curr_time < total_time:
        clip = data[x_time:x_time + N]
        energies.append(np.sum(np.square(clip, dtype='int64')))
        x_time += N
        curr_time += t
    
    energy_times = np.arange(len(energies))*t  # prepare x values for energies array
    
    crash_id = file_name.split('/')[-1].replace('.wav','')
    crash_id = crash_id.split('\\')[-1]
    
    fig, [ax1, ax2, ax3, ax4] = plt.subplots(4, 1, sharex=True)
    
    fig.suptitle(crash_id)
    ax1.plot(energy_times, energies, color='k', linewidth="0.5")
    ax4.set_xlabel('Time (s)')
    ax4.set_xticks(np.arange(min(energy_times),max(energy_times)+1, tick_dist))
    ax3.set_xlim(0, len(energies)*t)
    
    ax1.set_ylabel('Energy')
    ax1.set_yticks([])
    ax2.set_ylabel('Threshold')
    ax2.set_yticks([])
    ax3.set_ylabel('Detection')
    ax3.set_yticks([])
    ax4.set_ylabel('Z-score')
    ax4.set_yticks(np.arange(0,13,2))
    ax4.set_ylim(0, 12)
    
    # Run algo with settings from above
    result = peak_detection_smoothed_zscore_v2(energies,
                                               lag=lag,
                                               threshold=threshold,
                                               influence=influence)
    
    ax2.plot(energy_times, energies)
    ax2.plot(np.arange(lag, len(energies))*t,
             result["avgFilter"][lag:], color="cyan", lw=2)
    ax2.plot(np.arange(lag, len(energies))*t,
             result["avgFilter"][lag:] + threshold * result["stdFilter"][lag:],
             color="green", lw=2)
    ax3.step(np.arange(lag, len(energies))*t,
             result["signals"][lag:],
             color="red", 
             lw=2)
    ax4.plot(np.arange(lag, len(energies))*t,
             (energies[lag:]-result["avgFilter"][lag:]) / result["stdFilter"][lag:],
             color='cyan',
             lw=3)
    
      
    print_pdf.savefig() #save to pdf
    plt.close(fig) # do not display figures
    pp.close() #close pdf
    
    print('Potential crashes at times:')
    crashes = result['crashes']
    crashes_secs = map(lambda x: np.round(x*t), crashes)
    crash_times_mins = map(lambda x: str(datetime.timedelta(seconds=x)), crashes_secs)
    print(result['crashes'])
    print(list(crash_times_mins))
    
if __name__ == "__main__":
    t = 0.1
    lag = 1500
    threshold = 6
    influence = 0.75
    
    
    pdf_name='../Data/compilation_threshold_graphs.pdf'
    pp = PdfPages(pdf_name)
    detect_peak("C://Users/elind/Box/11Foot8/Data/Compilations/full_crash_compilation.wav",
                t=t,
                lag=lag,
                threshold=threshold,
                influence=influence,
                print_pdf=pp,
                tick_dist=2400.0)
    
    """
    pdf_name='../Data/threshold_graphs.pdf'
    pp = PdfPages(pdf_name)
    
    directories = ["C://Users/elind/Box/11Foot8/Data/Full_Crashes/Audio",
                   "C://Users/elind/Box/11Foot8/Data/Trains/Audio"]
    for dir_string in directories:
        directory = os.fsencode(dir_string)
        for file in os.listdir(directory):
            filename = os.fsdecode(file)
            if filename.endswith(".wav"):
                detect_peak(os.path.join(dir_string, filename),
                        t=t,
                        lag=lag,
                        threshold=threshold,
                        influence=influence,
                        print_pdf=pp)
                continue
            else:
                continue
    """ # Close PDF
    print('Done!')