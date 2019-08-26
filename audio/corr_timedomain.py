from __future__ import print_function
import scipy.io.wavfile as wavfile
import scipy
import scipy.fftpack
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import scipy.stats as stats

def crosscorr(datax, datay, lag=0, wrap=False):
    """ Lag-N cross correlation.
    Shifted data filled with NaNs

    Parameters
    ----------
    lag : int, default 0
    datax, datay : pandas.Series objects of equal length

    Returns
    ----------
    crosscorr : float
    """
    if wrap:
        shiftedy = datay.shift(lag)
        shiftedy.iloc[:lag] = datay.iloc[-lag:].values
        return datax.corr(shiftedy)
    else:
        return datax.corr(datay.shift(lag))


def corr(signal, signal2):
    signal = signal.sum(axis=1) / 2
    signal2 = signal2.sum(axis=1) / 2
    df = pd.DataFrame({'sig': signal, 'sig2': signal2})
    print(df)

    overall_pearson_r = df.corr().iloc[0, 1]
    print(f"Pandas computed Pearson r: {overall_pearson_r}")

    r, p = stats.pearsonr(df.dropna()['sig'], df.dropna()['sig2'])
    print(f"Scipy computed Pearson r: {r} and p-value: {p}")

    f, ax = plt.subplots(figsize=(14, 3))
    df.rolling(window=30, center=True).median().plot(ax=ax)
    ax.set(xlabel='Frame', ylabel='Smiling evidence', title=f"Overall Pearson r = {np.round(overall_pearson_r, 2)}");

    plt.show()

    d1 = df['sig']
    d2 = df['sig2']

    seconds = 5
    fps = 30

    rs = [crosscorr(d1, d2, lag) for lag in range(-int(seconds * fps - 1), int(seconds * fps))]
    offset = np.ceil(len(rs) / 2) - np.argmax(rs)
    f, ax = plt.subplots(figsize=(14, 3))
    ax.plot(rs)
    # print(rs)
    ax.axvline(np.ceil(len(rs) / 2), color='k', linestyle='--', label='Center')
    ax.axvline(np.argmax(rs), color='r', linestyle='--', label='Peak synchrony')
    ax.set(title=f'Offset = {offset} frames\nS1 leads <> S2 leads', ylim=[-.31, .31], xlim=[0, 300], xlabel='Offset',
           ylabel='Pearson r')
    ax.set_xticklabels([int(item - 150) for item in ax.get_xticks()]);
    plt.legend()
    plt.show()

if __name__ == '__main__':
    fs_rate, signal = wavfile.read("sound_0_0.wav")

    print("Frequency sampling", fs_rate)

    l_audio = len(signal.shape)

    print("Channels", l_audio)

    N = signal.shape[0]

    print("Complete Samplings N", N)
    secs = N / float(fs_rate)
    print("secs", secs)
    Ts = 1.0 / fs_rate  # sampling interval in time
    print("Timestep between samples Ts", Ts)
    t = scipy.arange(0, secs, Ts)  # time vector as scipy arange field / numpy.ndarray

    for i in range(1,9):
        fs_rate2, signal2 = wavfile.read(f"sound_0_{i}.wav")
        corr(signal,signal2)