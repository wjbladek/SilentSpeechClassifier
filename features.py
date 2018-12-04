# -*- coding: utf-8 -*-
"""A simple set of functions for extracting features from wave-like data,
especially EEG. Originally implemented by Stanisław Karkosz for Br41nhack
2018 in Poznań.
"""

import numpy as np
from scipy import stats
from scipy import signal
from scipy.spatial import distance

# positive area
def par(sequence):
    positive = [n for n in sequence if n >= 0]
    par = np.sum(positive)
    return par

# negative area
def nar(sequence):
    negative = [n for n in sequence if n <= 0]
    nar = np.sum(negative)
    return nar

# total area
def tar(sequence):
    tar = par(sequence) + nar(sequence)
    return tar

# total absolute area 
def taar(sequence):
    taar = par(sequence) + np.abs(nar(sequence))
    return taar

# amplitude, the maximal signal value.
def amp(sequence):
    mins = np.abs(np.min(sequence))
    maxs = np.max(sequence)
    if ( mins <= maxs ):
        amp = maxs
    elif ( maxs <= mins ):
        amp = np.min(sequence)
    return amp

# latency
def lat(sequence):
    latency = np.where(sequence == amp(sequence))[0]
    return(latency[0])

# latency/amplitude ratio:
def lar(sequence):
    lar = lat(sequence)/amp(sequence)
    return lar

# absolute amplitude
def aamp(sequence):
    aamp = np.abs(amp(sequence))
    return aamp

# absolute latency/amplitude ratio
def alar(sequence):
    alar = np.abs(lar(sequence))
    return alar

# peak-to-peak
def pp(sequence):
    return np.ptp(sequence)

# peak-to-peak time window
def ppt(sequence):
    ppt = np.where(sequence == np.max(sequence))[0] - np.where(sequence == np.min(sequence))
    return ppt[0][0]

# zero crossings
def zc(sequence):
    zc = np.count_nonzero(np.where(np.diff(np.sign(sequence)))[0])
    # solution taken from stack overflow
    return zc

# peak-to-peak slope
def pps(sequence):
    pps = pp(sequence)/ppt(sequence) 
    return pps[0][0]

# zero crossings density
def zcd(sequence):
    zcd = zc(sequence)/ppt(sequence)
    return zcd[0][0]

# standard deviation 
def std(sequence):
    return np.std(sequence)
# variance
def variance(sequence):
    return np.var(sequence)

# mean signal value
def mean_value(sequence):
    return np.mean(sequence)

# median signal value
def median_value(sequence):
    return np.median(sequence)

# mode signal value, rounded to a one decimal
def mode_value1(sequence):
    mode = stats.mode(np.round(sequence, decimals=1), axis=None)
    return mode[0][0]

# mode signal value, rounded to two decimals
def mode_value2(sequence):
    mode = stats.mode(np.round(sequence, decimals=2), axis=None)
    return mode[0][0]

# mode signal value, rounded to a one decimal
def mode_value3(sequence):
    mode = stats.mode(np.round(sequence, decimals=3), axis=None)
    return mode[0][0]
    
# frequency-domain features
class freq_dom_feat:

    def __init__(self, sequence):
        self.t = len(sequence)
        self.fft = abs(np.fft.fft(sequence*signal.windows.hamming(self.t, sym=False)))*2/self.t
        self.fft = self.fft[:self.t//2]
        self.freqs = np.fft.fftfreq(self.t)[:self.t//2]

    # mean freq, a centroid of the spectrum
    def mean_freq(self):
        sum = np.sum(self.fft)
        # sorted = np.sort(self.fft)
        ind = 1
        additive = 0
        while additive <= sum/2:
            additive+=self.fft[ind]
            ind+=1
        return self.freqs[ind-1]

    # mode freq
    def mode_freq(self):
        return self.freqs[np.argmax(self.fft)]

    # median freq
    def median_freq(self):
        index = np.argsort(self.fft)
        freq_left = self.freqs[index[self.t//4+1]]
        if self.t % 2 != 0:
            return freq_left
        else:
            freq_right = self.freqs[index[self.t//4]]
            return (freq_right+freq_left)/2
    

# Gathering function. 'Pretty' solution, using the functions above.
def pretty_feat_array(sequence, channel):
    """Extract features for a given sequence and return an structured array.
    
    Notes
    -----
    This functions takes a sequence and, using a set of functions, extracts
    various characteristics of the sequence. It returns them in a structured
    ndarray with a given channel name, features's names and features's
    values. Somewhat slow due to many calls to functions.

    Parameters
    ----------
    sequence : array_like
        Signal to extract features from. Should be an 1d array/list of
        ints, floats or doubles.
    channel : string (U4)
        ID of a source of a sequence, for instance a name of an EEG channel.

    Returns
    -------
    ndarray
        Structured numpy array (shape 24,) with 3 fields per feature:
        a channel name (4 unicode chars), a feature's name (4 unicode 
        chars) and a feature's value (float).
    """
    fdf = freq_dom_feat(sequence)
    feat_array = np.array([
        (channel, 'par', par(sequence)),
        (channel, 'tar', tar(sequence)),
        (channel, 'nar', nar(sequence)),
        (channel, 'taar', taar(sequence)),
        (channel, 'amp', amp(sequence)),
        (channel, 'lat', lat(sequence)),
        (channel, 'lar', lar(sequence)),
        (channel, 'aamp', aamp(sequence)),
        (channel, 'alar', alar(sequence)),
        (channel, 'pp', pp(sequence)),
        (channel, 'ppt', ppt(sequence)),
        (channel, 'zc', zc(sequence)),
        (channel, 'pps', pps(sequence)),
        (channel, 'zcd', zcd(sequence)),
        (channel, 'std', std(sequence)),
        (channel, 'var', variance(sequence)),
        (channel, 'mns', mean_value(sequence)),
        (channel, 'mes', median_value(sequence)),
        (channel, 'md1s', mode_value1(sequence)),
        (channel, 'md2s', mode_value2(sequence)),
        (channel, 'md3s', mode_value3(sequence)),
        (channel, 'mnf', fdf.mean_freq()),
        (channel, 'mef', fdf.median_freq()),
        (channel, 'mdf', fdf.mode_freq())],
        dtype = [('channel', 'U4'),('feature_name','U4'),('feature_value','f8')])
    return feat_array


# Around 2x faster implementation of the function above. 
# Only unwrapped from all the function calls.
def fast_feat_array(sequence, channel):
    """Extract features for a given sequence and return an structured array.
    
    Notes
    -----
    This functions takes a sequence and, using a set of functions, extracts
    various characteristics of the sequence. It returns them in a structured
    ndarray with a given channel name, features's names and features's
    values. Optimised with speed in mind (reduction in function calls).

    Parameters
    ----------
    sequence : array_like
        Signal to extract features from. Should be an 1d array/list of
        ints, floats or doubles.
    channel : string (U4)
        ID of a source of a sequence, for instance a name of an EEG channel.

    Returns
    -------
    ndarray
        Structured numpy array (shape 24,) with 3 fields per feature:
        a channel name (4 unicode chars), a feature's name (4 unicode 
        chars) and a feature's value (float).
    """
    positive = [n for n in sequence if n >= 0]
    par = np.sum(positive)
    negative = [n for n in sequence if n <= 0]
    nar = np.sum(negative)
    tar = par + nar
    taar = par + np.abs(nar)

    mins = np.abs(np.min(sequence))
    maxs = np.max(sequence)
    if ( mins <= maxs ):
        amp = maxs
    elif ( maxs <= mins ):
        amp = np.min(sequence)

    latency = np.where(sequence == amp)[0]
    lar = latency[0]/amp
    aamp = np.abs(amp)
    alar = np.abs(lar)
    pp = np.max(sequence) - np.min(sequence)
    ppt = np.where(sequence == np.max(sequence))[0] - np.where(sequence == np.min(sequence))
    zc = np.count_nonzero(np.where(np.diff(np.sign(sequence)))[0])
    pps = pp/ppt
    zcd = zc/ppt
    std = np.std(sequence)
    variance = np.var(sequence)
    mean_value = np.mean(sequence)
    median_value = np.median(sequence)
    mode_value1 = stats.mode(np.round(sequence, decimals=1), axis=None)
    mode_value2 = stats.mode(np.round(sequence, decimals=2), axis=None)
    mode_value3 = stats.mode(np.round(sequence, decimals=3), axis=None)
    fdf = freq_dom_feat(sequence)

    feat_array = np.array([
        (channel, 'par', par),
        (channel, 'tar', tar),
        (channel, 'nar', nar),
        (channel, 'taar', taar),
        (channel, 'amp', amp),
        (channel, 'lat', latency[0]),
        (channel, 'lar', lar),
        (channel, 'aamp', aamp),
        (channel, 'alar', alar),
        (channel, 'pp', pp),
        (channel, 'ppt', ppt[0][0]),
        (channel, 'zc', zc),
        (channel, 'pps', pps[0][0]),
        (channel, 'zcd', zcd[0][0]),
        (channel, 'std', std),
        (channel, 'var', variance),
        (channel, 'mns', mean_value),
        (channel, 'mes', median_value),
        (channel, 'md1s', mode_value1[0][0]),
        (channel, 'md2s', mode_value2[0][0]),
        (channel, 'md3s', mode_value3[0][0]),
        (channel, 'mnf', fdf.mode_freq()),
        (channel, 'mef', fdf.median_freq()),
        (channel, 'mdf', fdf.mode_freq())],
        dtype = [('channel', 'U4'),('feature_name','U4'),('feature_value','f8')])

    return feat_array