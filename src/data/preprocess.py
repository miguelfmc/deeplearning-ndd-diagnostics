""" Preprocessing of the Physionet gaitndd data
        1. Split signals
        2. NaN handling
        3. Normalization to zero mean and unit variance
        4. Creation of spectrograms
"""

import sys
import os
import numpy as np
import scipy.signal as sgn
import matplotlib.pyplot as plt
import wfdb

def nan_helper(y):
    """Helper to handle indices and logical indices of NaNs.

    Input:
        - y, 1d numpy array with possible NaNs
    Output:
        - nans, logical indices of NaNs
        - index, a function, with signature indices= index(logical_indices),
          to convert logical indices of NaNs to 'equivalent' indices
    """

    return np.isnan(y), lambda z: z.nonzero()[0]


def remove_from_list(blacklist_filename, target_dir):
    """Removes .npy files from target_dir if they appear in text file blacklist_filename
    """
    # read signal segment names from blacklist.txt and add them to list
    blacklist = []
    with open(blacklist_filename, 'r') as handle:
        for line in handle.readlines():
            blacklist.append(line.strip() + '.npy')

    # remove files in blacklist from data directory
    for file in os.listdir(target_dir):
        if file in blacklist:
            os.remove(os.path.join(out_dir, file))


def split_normalize_signals(in_dir, out_dir, n_fragments=10, nan_threshold=50):
    """Reads WFDB records from in_dir, splits left and right signal in n_fragments, handles NaN values,
    normalizes segments and saves them as .npy files in out_dir
    """
    for in_filename in sorted(os.listdir(in_dir)):
        if in_filename.endswith('hea'):
            record_name = in_filename[:-4]
            record = wfdb.rdrecord(os.path.join(in_dir, record_name), sampfrom=0, sampto=90000)
            
            signal = record.p_signal.T
            signal_right = signal[1, :]
            signal_left = signal[0, :]
            
            for i, fragment in enumerate(np.split(signal_right, n_fragments)[1:]):
                nans, t = nan_helper(fragment)
                num_of_nans = np.sum(nans) 
                
                if num_of_nans > nan_threshold:
                    continue
                elif num_of_nans <= nan_threshold and num_of_nans > 0:
                    fragment[nans] = np.interp(t(nans), t(~nans), fragment[~nans])
                
                normalized = (fragment - fragment.mean()) / fragment.std()
                out_filename = os.path.join(out_dir, record_name + '_rit_' + str(i+2))
                np.save(out_filename, normalized)
            
            for i, fragment in enumerate(np.split(signal_left, n_fragments)[1:]):
                nans, t = nan_helper(fragment)
                num_of_nans = np.sum(nans) 
                
                if num_of_nans > nan_threshold:
                    continue
                elif num_of_nans <= nan_threshold and num_of_nans > 0:
                    fragment[nans] = np.interp(t(nans), t(~nans), fragment[~nans])
                
                normalized = (fragment - fragment.mean()) / fragment.std()
                out_filename = os.path.join(out_dir, record_name + '_let_' + str(i+2))
                np.save(out_filename, normalized)


def create_spectrograms(in_dir, out_dir, L=512, fs=300, freq_max=7):
    """Creates spectrograms from signal segments in in_dir and saves them in out_dir
    """
    f_limit = freq_max * L // fs
    
    for in_filename in os.listdir(in_dir):
        signal = np.load(os.path.join(in_dir, in_filename))
        f, t, Sxx = sgn.spectrogram(signal, nperseg=L, window='hamming', noverlap=L//2)
        f = f[0:f_limit] * fs
        t = t / fs
        Sxx = Sxx[0:f_limit]
        out_filename = in_filename[:-4]
        np.savez(os.path.join(out_dir, out_filename), f=f, t=t, Sxx=Sxx)


def make_data_directory():
    try:
        os.makedirs(os.path.join('data', 'interim'))
    except OSError:
        pass
    
    try:
        os.makedirs(os.path.join('data', 'interim', 'normalized-signals'))
    except OSError:
        pass

    try:
        os.makedirs(os.path.join('data', 'interim', 'spectrograms'))
    except OSError:
        pass

def main():
    raw_dir = os.path.join('data', 'raw')
    interim_dir = os.path.join('data', 'interim')
    norm_signals_dir = os.path.join(interim_dir, 'normalized-signals')
    spectrograms_dir = os.path.join(interim_dir, 'spectrograms')

    make_data_directory()

    split_normalize_signals(raw_dir, norm_signals_dir)
    create_spectrograms(norm_signals_dir, spectograms_dir)

if __name__ == "__main__":
    main()
