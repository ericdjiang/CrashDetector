# -*- coding: utf-8 -*-
"""
Created on Mon Dec  3 13:46:47 2018

@author: elind
"""

import numpy as np
import matplotlib.pyplot as plt


bridge_readings = np.loadtxt("../Data/Bridge_Vibration_Readings/Bridge_readings_Nov_30_2018_8_mins.txt")

x_data = bridge_readings[:,0]
y_data = bridge_readings[:,1]
z_data = bridge_readings[:,2]

none_readings = np.loadtxt("../Data/Bridge_Vibration_Readings/Blank_test_2mins40secs.txt")
none_x = none_readings[:,0]
none_y = none_readings[:,1]
none_z = none_readings[:,2]

fig, [ax1, ax2, ax3] = plt.subplots(3, 1, sharex=True)

ax1.plot(x_data)
ax2.plot(y_data)
ax3.plot(z_data)

fig.show()

fig_none, [ax1_none, ax2_none, ax3_none] = plt.subplots(3, 1, sharex=True)

ax1_none.plot(none_x)
ax2_none.plot(none_y)
ax3_none.plot(none_z)

fig_none.show()