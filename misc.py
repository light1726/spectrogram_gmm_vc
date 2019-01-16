# -*- coding: utf-8 -*-

import os
import numpy as np
from scipy.signal import firwin, lfilter
from extfrm import extfrm
from delta import static_delta

def low_cut_filter(x, fs, cutoff=70):
    """Low cut filter

    Parameters
    ---------
    x : array, shape(`samples`)
        Waveform sequence
    fs: array, int
        Sampling frequency
    cutoff : float, optional
        Cutoff frequency of low cut filter
        Default set to 70 [Hz]

    Returns
    ---------
    lcf_x : array, shape(`samples`)
        Low cut filtered waveform sequence
    """

    nyquist = fs // 2
    norm_cutoff = cutoff / nyquist

    # low cut filter
    fil = firwin(255, norm_cutoff, pass_zero=False)
    lcf_x = lfilter(fil, 1, x)

    return lcf_x


def extsddata(data, npow, power_threshold=-20):
    """Get power extract static and delta feature vector

    Paramters
    ---------
    data : array, shape (`T`, `dim`)
        Acoustic feature vector
    npow : array, shape (`T`)
        Normalized power vector
    power_threshold : float, optional,
        Power threshold
        Default set to -20

    Returns
    -------
    extsddata : array, shape (`T_new` `dim * 2`)
        Silence remove static and delta feature vector

    """

    extsddata = extfrm(static_delta(data), npow,
                       power_threshold=power_threshold)
    return extsddata


def transform_jnt(array_list):
    num_files = len(array_list)
    for i in range(num_files):
        if i == 0:
            jnt = array_list[i]
        else:
            jnt = np.r_[jnt, array_list[i]]
    return jnt
