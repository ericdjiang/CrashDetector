# bridge-prototyping
Glorified loud noise detector. Utilizes moving average and custom FFT analysis to detect and notify user if a truck has crashed into the 11ft 8in bridge in Durham, NC.

## Background
The 11'8" bridge in Durham, North Carolina is notoriously known as the "can opener" bridge; tall trucks often crash into this deceptively low bridge, resulting in spectacular destruction and noise. Crashes occur multiple times each month, and Durham native Jurgen Henn runs a YouTube channel entirely dedicated to documenting 11foot8 crashes.

The purpose of this application is to detect and notify the client, Mr.Henn, of when a crash has occurred. Once a crash is detected, a notification of the crash will be immediately sent to the client.

## Usage
From the command prompt, simply run
```
python /Code/audio_threshold.py
```
This will kick off a python program which drawing upon a microphone input feed, searching for irregularities in the input sound. Due to the combined use of an amplitude-based moving average and an FFT algorithm to check if the frequency content of a noise matches that of a typical crash, false positives are quite rare. In testing, the system achieved a 89% detection rate on sample crashes.

