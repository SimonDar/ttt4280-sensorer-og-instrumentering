import numpy as np
import matplotlib
matplotlib.use('TkAgg')
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy.signal as signal


def raspi_import(path, channels=5):
    """
    Import data produced using adc_sampler.c.
    Returns sample period and ndarray with one column per channel.
    Sampled data for each channel, in dimensions NUM_SAMPLES x NUM_CHANNELS.
    """

    with open(path, 'r') as fid:
        sample_period = np.fromfile(fid, count=1, dtype=float)[0]
        data = np.fromfile(fid, dtype=np.uint16)
        data = data.reshape((-1, channels))
    return sample_period, data

print("Nothing has happened yet.")

# Import data from bin file
sample_period, data = raspi_import('adcData.bin', 3)

data = signal.detrend(data, axis=0)  # removes DC component for each channel
sample_period *= 1e-6  # change unit to micro seconds

# Decide interval of samples to look at
smp_start = 5000
smp_end = 5250
N = smp_end - smp_start


# Generate time axis
num_of_samples = data.shape[0]  # returns shape of matrix
t = np.linspace(start=0, stop=num_of_samples*sample_period, num=num_of_samples)

# Generate frequency axis and take FFT
freq = np.fft.fftfreq(n=num_of_samples, d=sample_period)
spectrum = np.fft.fft(data, axis=0)  # takes FFT of all channels

# Plot the results in two subplots
# NOTICE: This lazily plots the entire matrixes. All the channels will be put into the same plots.
# If you want a single channel, use data[:,n] to get channel n

plt.subplot(2, 1, 1)
plt.title("Time domain signal")
plt.xlabel("Time [us]")
plt.ylabel("Voltage")
plt.plot(t[smp_start:smp_end], data[smp_start:smp_end])

plt.subplot(2, 1, 2)
plt.title("Power spectrum of signal")
plt.xlabel("Frequency [Hz]")
plt.ylabel("Power [dB]")
plt.plot(freq, 20*np.log(np.abs(spectrum))) # get the power spectrum

plt.show()
print("Plot shown.")

''' Tied to lab 1 & 2
# Find time delays between microphone signals
f_s = 31250

def channelSamplesArray(data, channels, sample_start, sample_end): # Returns a 2D array with channel as the fist list and the respective 
    #channels = np.array(channels)-1
    samples = [0]*(np.max(channels)+1)
    for ch in channels:
        samples[ch] = data[sample_start:sample_end, ch]
    return np.array(samples)
    

def lagAtMaxCorr(arr1, arr2, length): # This function assumes that arr1 and arr2 has the same length
    corr = np.correlate(arr1, arr2, "full")
    return np.argmax(np.abs(corr)) - (length - 1)

def lagArray(smps, length): # This function assumes that the sample list for every channel in smps is the same length
    lags = []
    channel_amt = len(smps)
    for ch1 in range(channel_amt):
        ch1_lags = []
        for ch2 in range(channel_amt):
            ch1_lags.append(lagAtMaxCorr(smps[ch1], smps[ch2], length))
        lags.append(ch1_lags)
    return np.array(lags)

smps = channelSamplesArray(data, [0,1,2], smp_start, smp_end)
lags = lagArray(smps, N)

print("Lags at max correlation between the three channels:")
print(lags)
print("Verify that the elements in the [0][0] - [1][1] - [2][2] diagonal (the lags at max autocorrelation) of the matrix are 0.")
print()

tan_y = np.sqrt(3)*(lags[1][0]+lags[2][0])
tan_x = (lags[1][0]-lags[2][0]-2*lags[2][1])

angle = np.arctan2(tan_y, tan_x)

print("Vinkel [Rad]:")
print(angle)
print()
'''