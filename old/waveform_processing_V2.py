"""
CREATED BY MOHIT GOLA
"""
import uproot as ur
import numpy as np
from scipy import stats
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

def gaussian(x, amp, mean, stddev):
    return amp * np.exp(-((x - mean) ** 2) / (2 * stddev ** 2))

def get_peak_timebins(waveform, threshold, brb):
    values, counts = np.unique(waveform, return_counts=True)
    baseline = values[np.argmax(counts)]
    below = (baseline - waveform[0]) <= threshold
    peak_timebins = []
    max_val = 0
    max_timebin = -1
    for i in range(len(waveform)):
        if below and (baseline - waveform[i]) > threshold:
            below = False
            max_val = 0
            max_timebin = -1
        if not below:
            if (baseline - waveform[i]) > max_val:
                max_val = baseline - waveform[i]
                max_timebin = i
            if (baseline - waveform[i]) <= threshold:
                below = True
                if brb != 3 or len(peak_timebins) == 0 or max_timebin > peak_timebins[-1] + 4:
                    peak_timebins.append(max_timebin)
    return peak_timebins

def get_cfd(times, adcs):
    c = -2.
    d = 2
    baseline = (adcs[0] + adcs[1] + adcs[2]) / 3.
    n_largest_vals = sorted(baseline - np.array(adcs), reverse=True)[:3]
    amp = sum(n_largest_vals)

    data = [(baseline - adcs[i]) + c * (baseline - adcs[i - d]) for i in range(d, len(adcs))]
    max_diff = 0
    i_md = -1
    for iv in range(1, len(data)):
        if data[iv - 1] > 0. and data[iv] < 0.:
            if data[iv - 1] - data[iv] > max_diff:
                max_diff = data[iv - 1] - data[iv]
                i_md = iv

    if i_md > -1:
        x0 = i_md - 1
        y0 = data[i_md - 1]
        x1 = i_md
        y1 = data[i_md]
        x = x0 - (x1 - x0) / (y1 - y0) * y0
        t = x - 0.5703  # Adjust if necessary
    else:
        t = -999
        amp = -999

    return t, amp, baseline

def find_multiple_peaks(hist, prominence=0.1, height=10):
    smoothed_hist = gaussian_filter1d(hist, sigma=1)
    peaks, properties = find_peaks(smoothed_hist, prominence=prominence, height=height)
    return peaks, properties


def get_processed_data(run_number):
    threshold = 20  # Minimum above baseline to be considered a signal
    run_file = ur.open(f"/Users/mgola/Downloads/TimeCal-main/rootFiles/output_000{run_number:03d}.root")
    midas_data = run_file['midas_data']
    brb = 0  # Change to BRB 1 if required
    data = midas_data.arrays(library="np")
    stats_list = []
    fig, axs = plt.subplots(4, 5, figsize=(20, 15))
    fig.suptitle(f'Fitted Peak Times for All Channels (BRB 1) in Run {run_number}', fontsize=16)

    for channel in range(20):
        ch_adc = data[f'BRB_waveform_ch{channel}']
        brbno = data['brbno']
        ch_wf = [ch_adc[i] for i in range(len(brbno)) if brbno[i] == brb]  # Filter for BRB 1

        ch_led = ch_wf[:5310]  # Adjusted to use up to 5310 samples
        peak_timebins_all = [get_peak_timebins(ch_led[i], threshold, brb) for i in range(len(ch_led))]
        
        fitted_peak_times = []
        unfitted_means = []
        unfitted_stds = []

        for i in range(len(ch_led)):
            for j in range(len(peak_timebins_all[i])):
                fit_start = peak_timebins_all[i][j] - 7
                ts = np.array(range(fit_start, peak_timebins_all[i][j] + 2))
                adcs = np.array([ch_led[i][its] for its in ts])
                unfitted_means.append(np.mean(adcs))
                unfitted_stds.append(np.std(adcs))

                if peak_timebins_all[i][j] < len(ch_led[i]) - 4:
                    ts_cfd = np.array(range(fit_start, peak_timebins_all[i][j] + 4))
                    adcs_cfd = np.array([ch_led[i][its] for its in ts_cfd])
                    t0, amp, baseline = get_cfd(ts_cfd, adcs_cfd)
                    if t0 > 0:
                        fitted_peak_times.append(t0 + fit_start % 100)

        non_zero_fitted_peak_times = [t for t in fitted_peak_times if t > 0]
        
        # Skip fitting if there are no valid fitted peak times
        if len(non_zero_fitted_peak_times) == 0:
            print(f"Channel {channel} has no valid fitted peak times. Skipping fitting.")
            continue

        # Create histogram for fitted peak times
        hist, bin_edges = np.histogram(non_zero_fitted_peak_times, bins=2000, density=True)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        
        # Check for NaNs or infinite values in the histogram
        if np.any(np.isnan(hist)) or np.any(np.isinf(hist)):
            print(f"Channel {channel} has NaN or inf values in histogram. Skipping fitting.")
            continue

        # Find peaks in the histogram
        peaks, properties = find_multiple_peaks(hist, prominence=0.1, height=0.01)
        num_peaks = len(peaks)
        print(f"Channel {channel} - Number of Peaks: {num_peaks}")

        # Plot histogram and detected peaks for visual verification
        axs[channel // 5, channel % 5].hist(non_zero_fitted_peak_times, bins=2000, color='blue', alpha=0.7, density=True)
        axs[channel // 5, channel % 5].plot(bin_centers, hist, color='green', alpha=0.5, label='Histogram')
        axs[channel // 5, channel % 5].plot(bin_centers[peaks], hist[peaks], 'ro', label='Detected Peaks')

        if num_peaks > 0:
            axs[channel // 5, channel % 5].set_xlim([bin_centers[peaks].min() - 2, bin_centers[peaks].max() + 2])
        else:
            axs[channel // 5, channel % 5].set_xlim([0, 100])
        axs[channel // 5, channel % 5].set_title(f'Channel {channel}')
        axs[channel // 5, channel % 5].grid(True)
        axs[channel // 5, channel % 5].legend()

        # Proceed if only one peak is present for fitting

        if num_peaks == 1:
            try:
                # Adjust the rising edge mask to include more data points
                rising_edge_mask = (bin_centers >= bin_centers[peaks[0]] - 5) & (bin_centers <= bin_centers[peaks[0]])
                rising_edge_hist = hist[rising_edge_mask]
                rising_edge_bins = bin_centers[rising_edge_mask]

                if len(rising_edge_hist) < 5:  # Ensure enough data points for fitting
                    print(f"Not enough data points for fitting in Channel {channel}.")
                    continue
                
                # Adjusting the initial guess based on histogram data
                initial_guess = [max(rising_edge_hist), bin_centers[peaks[0]], np.std(rising_edge_hist)]
                bounds = ([0, bin_centers[peaks[0]] - 5, 0], [np.inf, bin_centers[peaks[0]], np.inf])

                # Fit the Gaussian to the rising edge
                popt, pcov = curve_fit(gaussian, rising_edge_bins, rising_edge_hist, p0=initial_guess, bounds=bounds)
                fitted_mean = popt[1]
                fitted_std = popt[2]

                # Plotting the fitted Gaussian on the rising edge
                axs[channel // 5, channel % 5].plot(rising_edge_bins, gaussian(rising_edge_bins, *popt), 'r-', label='Fitted Gaussian', linewidth=2)
                axs[channel // 5, channel % 5].axvline(fitted_mean, color='orange', linestyle='--', label='Fitted Mean', linewidth=1)

                # Display fitted mean and standard deviation in the title
                axs[channel // 5, channel % 5].set_title(f'Channel {channel} (Mean: {fitted_mean:.2f}, Std: {fitted_std:.2f})')
        
            except RuntimeError:
                fitted_mean = None
                fitted_std = None
                print(f"Fitting failed for Channel {channel}.")
        
            stats_list.append((channel, np.mean(unfitted_means), np.std(unfitted_stds), fitted_mean, fitted_std))

    stats_df = pd.DataFrame(stats_list, columns=["Channel", "Unfitted Mean", "Unfitted Std", "Fitted Mean", "Fitted Std"])
    stats_df.to_csv(f'Statistics_Run_{run_number}.csv', index=False)
    print(stats_df)


    plt.tight_layout()
    plt.savefig(f'Fitted_Peak_Times_Run_{run_number}.png')
    plt.show()

# Example usage:
get_processed_data(213)
'''
def get_processed_data(run_number):
    threshold = 20  # Minimum above baseline to be considered a signal
    run_file = ur.open(f"/Users/mgola/Downloads/TimeCal-main/rootFiles/output_000{run_number:03d}.root")
    midas_data = run_file['midas_data']
    brb = 0  # Change to BRB 1
    data = midas_data.arrays(library="np")
    stats_list = []
    fig, axs = plt.subplots(4, 5, figsize=(20, 15))
    fig.suptitle(f'Fitted Peak Times for All Channels (BRB 1) in Run {run_number}', fontsize=16)

    for channel in range(20):
        ch_adc = data[f'BRB_waveform_ch{channel}']
        brbno = data['brbno']
        ch_wf = []
        for i in range(len(brbno)):
            if brbno[i] == brb:  # Filter for BRB 1
                ch_wf.append(ch_adc[i])

        ch_led = ch_wf[:5310]  # Adjusted to use up to 6420 samples
        peak_timebins_all = []
        for i in range(len(ch_led)):
            times = get_peak_timebins(ch_led[i], threshold, brb)
            peak_timebins_all.append(times)

        fitted_peak_times = []
        data_length = 1024
        unfitted_means = []
        unfitted_stds = []

        for i in range(len(ch_led)):
            for j in range(len(peak_timebins_all[i])):
                fit_start = peak_timebins_all[i][j] - 7
                ts = np.array([i for i in range(fit_start, peak_timebins_all[i][j] + 2)])
                adcs = np.array([ch_led[i][its] for its in ts])
                unfitted_means.append(np.mean(adcs))
                unfitted_stds.append(np.std(adcs))

                if peak_timebins_all[i][j] < data_length - 4:
                    ts_cfd = np.array([i for i in range(fit_start, peak_timebins_all[i][j] + 4)])
                    adcs_cfd = np.array([ch_led[i][its] for its in ts_cfd])
                    t0, amp, baseline = get_cfd(ts_cfd, adcs_cfd)
                    if t0 > 0:
                        fitted_peak_times.append(t0 + fit_start % 100)

        non_zero_fitted_peak_times = [t for t in fitted_peak_times if t > 0]

        # Check if there are valid fitted peak times
        if len(non_zero_fitted_peak_times) == 0:
            print(f"Channel {channel} has no valid fitted peak times. Skipping fitting.")
            continue  # Skip fitting for this channel

        # Create histogram for fitted peak times for plotting
        hist, bin_edges = np.histogram(non_zero_fitted_peak_times, bins=2000, density=True)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        # Check for NaNs or infinite values in the histogram
        if np.any(np.isnan(hist)) or np.any(np.isinf(hist)):
            print(f"Channel {channel} has NaN or inf values in histogram. Skipping fitting.")
            continue  # Skip fitting and plotting for this channel

        # Find peaks in the histogram
        peaks, properties = find_multiple_peaks(hist, prominence=0.1, height=0.01)

        # Print the number of peaks for debugging
        num_peaks = len(peaks)
        print(f"Channel {channel} - Number of Peaks: {num_peaks}")

        # Skip fitting if the number of peaks is not exactly one
        if num_peaks != 1:
            print(f"Channel {channel} has {num_peaks} peaks. Skipping fitting.")
            continue  # Skip fitting for this channel

        # Plot histogram and detected peaks for visual verification
        axs[channel // 5, channel % 5].hist(non_zero_fitted_peak_times, bins=2000, color='blue', alpha=0.7, density=True)
        axs[channel // 5, channel % 5].plot(bin_centers, hist, color='green', alpha=0.5, label='Histogram')
        axs[channel // 5, channel % 5].plot(bin_centers[peaks], hist[peaks], 'ro', label='Detected Peaks')

        # Use the detected peaks to set x-limits dynamically
        if num_peaks > 0:
            axs[channel // 5, channel % 5].set_xlim([bin_centers[peaks].min() - 2, bin_centers[peaks].max() + 2])
        else:
            axs[channel // 5, channel % 5].set_xlim([0, 100])

        axs[channel // 5, channel % 5].set_title(f'Channel {channel}')
        axs[channel // 5, channel % 5].grid(True)
        axs[channel // 5, channel % 5].legend()

        # Proceed if only one peak is present for fitting
        try:
            # Update initial guess based on actual peak data
            initial_guess = [max(hist), np.mean(non_zero_fitted_peak_times), np.std(non_zero_fitted_peak_times)]
            bounds = ([0, 70, 0], [np.inf, 85, np.inf])
            initial_guess[1] = np.clip(initial_guess[1], bounds[0][1], bounds[1][1])
            initial_guess[2] = max(initial_guess[2], bounds[0][2])

            popt, pcov = curve_fit(gaussian, bin_centers, hist, p0=initial_guess, bounds=bounds)
            fitted_mean = popt[1]
            fitted_std = popt[2]

            # Plotting the fitted Gaussian
            axs[channel // 5, channel % 5].plot(bin_centers, gaussian(bin_centers, *popt), 'r-', label='Fitted Gaussian')

        except RuntimeError:
            fitted_mean = None
            fitted_std = None
            print(f"Fitting failed for Channel {channel}.")

        stats_list.append((channel, np.mean(unfitted_means), np.std(unfitted_stds), fitted_mean, fitted_std))

    stats_df = pd.DataFrame(stats_list, columns=["Channel", "Unfitted Mean", "Unfitted Std", "Fitted Mean", "Fitted Std"])
    stats_df.to_csv(f'Statistics_Run_{run_number}.csv', index=False)
    print(stats_df)

    # Save the histogram figure
    plt.tight_layout()
    plt.savefig(f'Fitted_Peak_Times_Run_{run_number}.png')
    plt.show()
'''

