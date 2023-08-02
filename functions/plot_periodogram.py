from scipy.signal import periodogram
import pandas as pd
import matplotlib.pyplot as plt

def plot_periodogram(series):

    fs = pd.Timedelta("1Y") / pd.Timedelta("1W")
    frequencies, spectrum = periodogram(series, fs=fs,
                                       detrend='linear', window="boxcar", scaling='spectrum')
    fig, ax = plt.subplots()
    ax.step(frequencies, spectrum, color="purple")
    ax.set_xscale("log")
    ax.set_xticks([1, 2, 4, 6, 12, 26])
    ax.set_xticklabels(
            [
                "Annual (1)",
                "Semiannual (2)",
                "Quarterly (4)",
                "Bimonthly (6)",
                "Monthly (12)",
                "Biweekly (26)"
            ],
            rotation=90,
        )
    ax.set_ylabel("Variance")
    ax.set_title("Periodogram")

    return ax