# -*- coding: utf-8 -*-
"""A simple set of functions for extracting features from wave-like data,
especially EEG. Originally implemented by Stanisław Karkosz for Br41nhack
2018 in Poznań.
"""

import numpy as np

# positive area
def par(sygnal):
    positive = [n for n in sygnal if n >= 0]
    par = np.sum(positive)
    return par

# negative area
def nar(sygnal):
    negative = [n for n in sygnal if n <= 0]
    nar = np.sum(negative)
    return nar

# total area
def tar(sygnal):
    tar = par(sygnal) + nar(sygnal)
    return tar

# total absolute area 
def taar(sygnal):
    taar = par(sygnal) + np.abs(nar(sygnal))
    return taar

# amplitude, the maximal signal value.
def amp(sygnal):
    mins = np.abs(np.min(sygnal))
    maxs = np.max(sygnal)
    if ( mins <= maxs ):
        amp = maxs
    elif ( maxs <= mins ):
        amp = np.min(sygnal)
    return amp

# latency
def lat(sygnal):
    latency = np.where(sygnal == amp(sygnal))[0]
    return(latency[0])

# latency/amplitude ratio:
def lar(sygnal):
    lar = lat(sygnal)/amp(sygnal)
    return lar

# absolute amplitude
def aamp(sygnal):
    aamp = np.abs(amp(sygnal))
    return aamp

# absolute latency/amplitude ratio
def alar(sygnal):
    alar = np.abs(lar(sygnal))
    return alar

# peak-to-peak
def pp(sygnal):
    pp = np.max(sygnal) - np.min(sygnal)
    return pp

# peak-to-peak time window
def ppt(sygnal):
    ppt = np.where(sygnal == np.max(sygnal))[0] - np.where(sygnal == np.min(sygnal))
    return ppt[0][0]

# zero crossings
def zc(sygnal):
    zc = np.count_nonzero(np.where(np.diff(np.sign(sygnal)))[0])
    # solution taken from stack overflow
    return zc


# Gathering function. 'Pretty' solution, using the functions above.
def pretty_feat_array(signal, channel):
    """Extract features for a given signal and return an structured array.
    
    Notes
    -----
    This functions takes a signal and, using a set of functions, extracts
    various characteristics of the signal. It returns them in a structured
    ndarray with a given channel name, features's names and features's
    values. Somewhat slow due to many calls to functions.

    Parameters
    ----------
    signal : array_like
        Signal to extract features from. Should be an 1d array/list of
        ints, floats or doubles.
    channel : string (U4)
        ID of a source of a signal, for instance a name of an EEG channel.

    Returns
    -------
    ndarray
        Structured numpy array (shape 12,) with 3 fields per feature:
        a channel name (4 unicode chars), a feature's name (4 unicode 
        chars) and a feature's value (float).
    """
    feat_array = np.array([
        (channel, 'par', par(signal)),
        (channel, 'tar', tar(signal)),
        (channel, 'nar', nar(signal)),
        (channel, 'taar', taar(signal)),
        (channel, 'amp', amp(signal)),
        (channel, 'lat', lat(signal)),
        (channel, 'lar', lar(signal)),
        (channel, 'aamp', aamp(signal)),
        (channel, 'alar', alar(signal)),
        (channel, 'pp', pp(signal)),
        (channel, 'ppt', ppt(signal)),
        (channel, 'zc', zc(signal))],
        dtype = [('channel', 'U4'),('feature_name','U4'),('feature_value','f8')])
    return feat_array


# Around 2x faster implementation of the function above. 
# Only unwrapped from all the function calls.
def fast_feat_array(signal, channel):
    """Extract features for a given signal and return an structured array.
    
    Notes
    -----
    This functions takes a signal and, using a set of functions, extracts
    various characteristics of the signal. It returns them in a structured
    ndarray with a given channel name, features's names and features's
    values. Optimised with speed in mind (reduction in function calls).

    Parameters
    ----------
    signal : array_like
        Signal to extract features from. Should be an 1d array/list of
        ints, floats or doubles.
    channel : string (U4)
        ID of a source of a signal, for instance a name of an EEG channel.

    Returns
    -------
    ndarray
        Structured numpy array (shape 12,) with 3 fields per feature:
        a channel name (4 unicode chars), a feature's name (4 unicode 
        chars) and a feature's value (float).
    """
    positive = [n for n in signal if n >= 0]
    par = np.sum(positive)
    negative = [n for n in signal if n <= 0]
    nar = np.sum(negative)
    tar = par + nar
    taar = par + np.abs(nar)

    mins = np.abs(np.min(signal))
    maxs = np.max(signal)
    if ( mins <= maxs ):
        amp = maxs
    elif ( maxs <= mins ):
        amp = np.min(signal)

    latency = np.where(signal == amp)[0]
    lar = latency[0]/amp
    aamp = np.abs(amp)
    alar = np.abs(lar)
    pp = np.max(signal) - np.min(signal)
    ppt = np.where(signal == np.max(signal))[0] - np.where(signal == np.min(signal))
    zc = np.count_nonzero(np.where(np.diff(np.sign(signal)))[0])

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
        (channel, 'zc', zc)],
        dtype = [('channel', 'U4'),('feature_name','U4'),('feature_value','f8')])

    return feat_array