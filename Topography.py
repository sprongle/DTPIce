import numpy as np
import matplotlib.pyplot as plt
# data from Bedmap2, read with the Matlab toolkit.
# profile through the Lambert Glacier, Antarctica
# XData = length in km
# YData = elevation in m

bedXData = np.loadtxt('bedXData.txt')
bedYData = np.loadtxt('bedYData.txt')
iceXData = np.loadtxt('iceXData.txt')
iceYData = np.loadtxt('iceYData.txt')
waterXData = np.loadtxt('waterXData.txt')
waterYData = np.loadtxt('waterYData.txt')

plt.plot(bedXData, bedYData)
plt.plot(iceXData, iceYData)
plt.plot(waterXData, waterYData)
plt.show()

