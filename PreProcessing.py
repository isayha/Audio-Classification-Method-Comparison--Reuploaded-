# CPSC 473 - Introduction to Data Mining
# Team Project - Audio Analysis and Classification

# External Packages:
import librosa
import numpy

# Returns a list of tuples containing:
    # The floating point value time series equivalents of all .wav files found within the specified directory
        # Relevant .wav metadata, including file name, duration, and sample rate
# Note: All .wav files found within the specified directory retain their native sample rate, and are:
    # Trimmed of leading and trailing white noise (this is accounted for in the calculation of duration)
    # Converted into mono (where natively stereo)
def get_wav_file(dir, filename):
    file = dict()
    fp_values, sample_rate = librosa.load(path = (dir + '\\' + filename), sr = None, mono = True)
    fp_values_trimmed, trim_index = librosa.effects.trim(fp_values)
    dur_trimmed = librosa.get_duration(fp_values_trimmed, sample_rate)
    file['name'] = filename
    file['fp_values'] = fp_values_trimmed
    file['sample_rate'] = sample_rate
    file['dur'] = dur_trimmed
    return file

# Returns the (estimated) maximum frequency across the entire specified .wav by:
    # Taking the spectral rolloff of all samples in the .wav with a high roll percent
        # Taking the maximum of all of the calculated spectral rolloffs
def get_est_max_freq(fp_values, sample_rate):
    est_max_freq = float('-inf')
    spectral_rolloffs = librosa.feature.spectral_rolloff(fp_values, sample_rate, roll_percent = 0.99)[0]
    for rolloff in spectral_rolloffs:
        if rolloff > est_max_freq:
            est_max_freq = rolloff
    return est_max_freq

# Returns the (estimated) minimum frequency across the entire specified .wav by:
    # Taking the spectral rolloff of all samples in the .wav with a low roll percent
        # Taking the minimum of all of the calculated spectral rolloffs
def get_est_min_freq(fp_values, sample_rate):
    est_min_freq = float('inf')
    spectral_rolloffs = librosa.feature.spectral_rolloff(fp_values, sample_rate, roll_percent = 0.01)[0]
    for rolloff in spectral_rolloffs:
        if rolloff < est_min_freq:
            est_min_freq = rolloff
    return est_min_freq

# Returns (the average) spectral flatness (tonality) across the entire specified .wav by:
    # Taking the spectral flatness of all samples in the .wav
        # Calculating the average of all of the calculated spectral flatness values
def get_mean_spectral_flatness(fp_values):
    spectral_flatness_frames = librosa.feature.spectral_flatness(fp_values)[0]
    mean_spectral_flatness = sum(spectral_flatness_frames)/len(spectral_flatness_frames)
    return mean_spectral_flatness

# Returns the maximum (absolute) floating point value across the entire specified .wav
def get_max_fp_value(fp_values):
    max_fp_value = float('-inf')
    for fp_value in fp_values:
        abs_sample = abs(fp_value)
        if abs_sample > max_fp_value:
            max_fp_value = abs_sample
    return max_fp_value

# Returns the minimum (absolute) floating point value across the entire specified .wav
def get_min_fp_value(fp_values):
    min_fp_value = float('inf')
    for fp_value in fp_values:
        abs_sample = abs(fp_value)
        if abs_sample < min_fp_value:
            min_fp_value = abs_sample
    return min_fp_value

# Returns (the average) spectral centroid (center of frequencies) across the entire specified .wav by:
    # Taking the spectral centroid of all samples in the .wav
        # Calculating the average of all of the calculated spectral centroid values
def get_mean_spectral_centroid(fp_values, sample_rate):
    spectral_centroid_frames = librosa.feature.spectral_centroid(fp_values, sample_rate)[0]
    mean_spectral_centroid = sum(spectral_centroid_frames)/len(spectral_centroid_frames)
    return mean_spectral_centroid

# Returns the average and the maximum amplitudes across the entire specified .wav divided into three (3) frequency bands (low, mid, and high)
    # The definitions of the frequency bands themselves can be changed by adjusting the values of low_cutoff_freq and mid_cutoff_freq
def get_mean_and_max_amp_per_band(fp_values, sample_rate):
    bands = dict()

    fft_window_len = 2048
    low_cutoff_freq = 200
    mid_cutoff_freq = 2000

    # Gets the frequencies associated with each frequency bin found within the (later generated) Short Time Fourier Transformation
    # Gets the indexes of the frequency bins at which cutoff frequencies are (approximately) found

    low_window_count = 0
    mid_first_window_index = None
    mid_window_count = 0
    high_first_window_index = None
    high_window_count = 0

    freqs = librosa.fft_frequencies(sample_rate, fft_window_len)

    for index in range (0, len(freqs)):
        if freqs[index] < low_cutoff_freq:
            low_window_count += 1
        elif freqs[index] < mid_cutoff_freq:
            if mid_first_window_index is None:
                mid_first_window_index = index
            mid_window_count += 1
        else:
            if high_first_window_index is None:
                high_first_window_index = index
            high_window_count += 1

    short_time_fourier_trans = librosa.stft(fp_values, fft_window_len)
    short_time_fourier_trans_abs = numpy.abs(short_time_fourier_trans)

    # Get the average and maximum amplitudes within the low band

    low_max_amp = float('-inf')
    low_amps = []
    
    for index in range (0, mid_first_window_index):
        freq_bin = short_time_fourier_trans_abs[index]
        for amp in freq_bin:
            low_amps.append(amp)
            if amp > low_max_amp:
                low_max_amp = amp

    low_mean_amp = sum(low_amps) / len(low_amps)

    low = dict()
    low['max_amp'] = low_max_amp
    low['mean_amp'] = low_mean_amp

    # Get the average and maximum amplitudes within the mid band

    mid_max_amp = float('-inf')
    mid_amps = []
    
    for index in range (mid_first_window_index, high_first_window_index):
        freq_bin = short_time_fourier_trans_abs[index]
        for amp in freq_bin:
            mid_amps.append(amp)
            if amp > mid_max_amp:
                mid_max_amp = amp

    mid_mean_amp = sum(mid_amps) / len(mid_amps)

    mid = dict()
    mid['max_amp'] = mid_max_amp
    mid['mean_amp'] = mid_mean_amp

    # Get the average and maximum amplitudes within the high band

    high_max_amp = float('-inf')
    high_amps = []
    
    for index in range (high_first_window_index, len(short_time_fourier_trans_abs)):
        freq_bin = short_time_fourier_trans_abs[index]
        for amp in freq_bin:
            high_amps.append(amp)
            if amp > high_max_amp:
                high_max_amp = amp

    high_mean_amp = sum(high_amps) / len(high_amps)

    high = dict()
    high['max_amp'] = high_max_amp
    high['mean_amp'] = high_mean_amp

    # Return all average and maximum amplitudes
    
    bands['low'] = low
    bands['mid'] = mid
    bands['high'] = high

    return bands

def get_mean_mfccs(fp_value, sample_rate, mfcc_count = 20):
    """Calculates and returns the mean value of each Mel-frequency cepstral coefficient (MFCC) for a given .wav file (in the form of a time series and the sample rate)

        Parameters
        ----------
        fp_value : list[int]
            The floating point value time series equivalent of a .wav file
        sample_rate : int
            The native sample rate of a .wav file
        mfcc_count : int
            (Optional) The number of MFCCs to calculate; the default value is 20
    """

    mfccs = librosa.feature.mfcc(fp_value, sample_rate, n_mfcc = mfcc_count)
    mean_mfccs = numpy.mean(mfccs.T, axis = 0)
    return mean_mfccs

# Returns a fully pre-processed version of the output provided by get_wav_file
# Default attribute set (value of preprocess_method) is 1 (newest preprocessing type; based on mean Mel-frequency cepstral coefficients)
def get_file_w_attr(file, preprocess_method = 1):
    file_w_attr = dict()
    file_w_attr['name'] = file['name']
    file_w_attr['fp_values'] = file['fp_values']
    file_w_attr['sample_rate'] = file['sample_rate']
    file_w_attr['dur'] = file['dur']

    if preprocess_method in [0,2]:
        file_w_attr['mean_spectral_flatness'] = get_mean_spectral_flatness(file_w_attr['fp_values'])
        file_w_attr['mean_spectral_centroid'] = get_mean_spectral_centroid(file_w_attr['fp_values'], file_w_attr['sample_rate'])

        mean_and_max_amp_per_band = get_mean_and_max_amp_per_band(file_w_attr['fp_values'], file_w_attr['sample_rate'])

        mean_and_max_amp_low = mean_and_max_amp_per_band['low']
        mean_and_max_amp_mid = mean_and_max_amp_per_band['mid']
        mean_and_max_amp_high = mean_and_max_amp_per_band['high']

        file_w_attr['low_mean_amp'] = mean_and_max_amp_low['mean_amp']
        file_w_attr['low_max_amp'] = mean_and_max_amp_low['max_amp']
        file_w_attr['mid_mean_amp'] = mean_and_max_amp_mid['mean_amp']
        file_w_attr['mid_max_amp'] = mean_and_max_amp_mid['max_amp']
        file_w_attr['high_mean_amp'] = mean_and_max_amp_high['mean_amp']
        file_w_attr['high_max_amp'] = mean_and_max_amp_high['max_amp']

    if preprocess_method in [1,2]:
        # Gets mean MFCC values and adds them to file_w_attr under keys 'mean_mfcc_0' through 'mean_mfcc_n'
        mean_mfccs = get_mean_mfccs(file_w_attr['fp_values'],  file_w_attr['sample_rate'])
        mfcc_number = 0
        for mean_mfcc in mean_mfccs:
            file_w_attr['mean_mfcc_' + str(mfcc_number)] = mean_mfcc
            mfcc_number += 1

    # Deemed useless (do not extract useful information); left commented out for demonstrative purposes:
    # file_w_attr['min_fp_value'] = get_min_fp_value(file_w_attr['fp_values'])
    # file_w_attr['max_fp_value'] = get_max_fp_value(file_w_attr['fp_values'])
    # file_w_attr['est_max_freq'] = get_est_max_freq(file_w_attr['fp_values'], file_w_attr['sample_rate'])
    # file_w_attr['est_min_freq'] = get_est_min_freq(file_w_attr['fp_values'], file_w_attr['sample_rate'])

    return(file_w_attr)