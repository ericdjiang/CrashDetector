# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 15:19:40 2018

@author: elind
"""
import os
import wave

infiles = []
outfile = "C:/Users/elind/Box/11Foot8/Data/Compilations/full_crash_compilation.wav"

directories = ["C://Users/elind/Box/11Foot8/Data/Full_Crashes/Audio",
                   "C://Users/elind/Box/11Foot8/Data/Trains/Audio"]
for dir_string in directories:
    directory = os.fsencode(dir_string)
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(".wav"):
            infiles += [os.path.join(dir_string, filename)]
            continue
        else:
            continue
        
print(infiles)
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