# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 15:19:40 2018

@author: elind
"""
import os
import wave


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
    
    for dir_string in directories:
        directory = os.fsencode(dir_string)
        for file in os.listdir(directory):
            filename = os.fsdecode(file)
            
            if filename.endswith(ext):
                infiles += [os.path.join(dir_string, filename)]
                event_info = filename.replace(ext,'').replace('b', '').strip()
                event_time = event_info.split(' at ')[-1]
                event_time = event_time.split('-')
                event_times += [float(event_time[0])*60+float(event_time[1])]
                files += [filename]
                
                for keycode in event_dict:
                    if event_dict[keycode] in event_info:
                        event_ids += [keycode]
                        break
                    else:
                        pass
            
            else:
                continue
    
    return event_times, event_ids, files
'''
    data= []
    for infile in infiles:
        w = wave.open(infile, 'rb')
        data.append( [w.getparams(), w.readframes(w.getnframes())] )
        w.close()
    
    output = wave.open(outfile, 'wb')
    output.setparams(data[0][0])
    for params,frames in data:
        output.writeframes(frames)
    output.close()
'''   
    
if __name__ == '__main__':
    directories = ["C://Users/elind/Box/11Foot8/Data/Full_Crashes/Audio",
                   "C://Users/elind/Box/11Foot8/Data/Trains/Audio"]
    output = "C:/Users/elind/Box/11Foot8/Data/Compilations/full_crash_compilation.wav"
    ext = '.wav'
    print(make_compilation(directories, output, ext))
