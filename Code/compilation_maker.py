# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 15:19:40 2018

@author: elind
"""
import os
import wave
import contextlib
import numpy as np

def make_compilation(directories, outfile, ext='.wav'):
    """
    directories: list of directories that contain all of the files you want to join
    outfile: 
    """
    
    event_dict = {0:'nothing',
                  2:'car_crash',
                  1:'crash',
                  3:'train',
                  4:'car passing',
                  5:'plane flying',
                  9:'miscellaneous'}
    
    infiles = []
    event_times = []
    event_ids = []
    files = []
    durations = []   
    
    for dir_string in directories:
        directory = os.fsencode(dir_string)
        for file in os.listdir(directory):
            filename = os.fsdecode(file)
            
            if filename.endswith(ext):
                infile = os.path.join(dir_string, filename)
                infiles += infile 
                event_info = filename.replace(ext,'').replace('b', '').strip()
                event_time = event_info.split(' at ')[-1]
                event_time = event_time.split('-')
                event_times += [float(event_time[0])*60+float(event_time[1])]
                files += [filename]
                
                # get duration of files
                with contextlib.closing(wave.open(infile,'r')) as f:
                    frames = f.getnframes()
                    rate = f.getframerate()
                    duration = frames / float(rate)
                    durations += [duration]
                
                # classify each .wav by event
                for keycode in event_dict:
                    if event_dict[keycode] in event_info:
                        event_ids += [keycode]
                        break
                    else:
                        pass
            
            else:
                continue
    
    # create array for event times with ending value of 0
    event_times = np.array(event_times + [0])
    # create cumulative sum of durations to add to event_times with first value 0
    durations_cum_sum = np.cumsum([0] + durations)
    # create cumulative sum of event times to identify time in compilation
    # without last value
    event_times_cum_sum = (event_times + durations_cum_sum)[:-1]
    
    print(durations_cum_sum[-1])
    
    return event_ids, event_times_cum_sum, files

if __name__ == '__main__':
    directories = ["C://Users/elind/Box/11Foot8/Data/Full_Crashes/Audio",
                   "C://Users/elind/Box/11Foot8/Data/Trains/Audio"]
    output = "C:/Users/elind/Box/11Foot8/Data/Compilations/full_crash_compilation.wav"
    ext = '.wav'
    event_ids, event_times_cum_sum, files = make_compilation(directories, output, ext)
    