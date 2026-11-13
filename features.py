#%%
from mne.preprocessing import ICA
from mne_icalabel import label_components
from mne_bids import BIDSPath, read_raw_bids
import pandas as pd
import numpy as np
from scipy.signal import butter, find_peaks, hilbert, sosfiltfilt
from matplotlib import pyplot as plt

DATASET_PATH = "ds007526"
ELECTRODES_OF_INTEREST = ["C1", "C3", "C5", "C2", "C4", "C6", "FC1", "FC3", "FC2", "FC4", "CP1", "CP2"]
PERCENTILES = np.arange(20, 100, 5)
HALF_BANDWIDTH = 3

participants_df = pd.read_csv(f"{DATASET_PATH}/participants.tsv", sep="\t", na_values="n/a")
datasets = []
for p in participants_df.itertuples():
    subject = p.participant_id.replace("sub-", "")
    rest_path = BIDSPath(subject=subject, task="rest", datatype="eeg", root=DATASET_PATH)
    walk_path = BIDSPath(subject=subject, task="walk", datatype="eeg", root=DATASET_PATH)
    if rest_path.fpath.exists():
        datasets.append((p.participant_id, "rest", rest_path))
    if walk_path.fpath.exists():
        datasets.append((p.participant_id, "walk", walk_path))

verbose = True
dataframe_rows = []
for subject, task, path in datasets:
    try:
        if verbose: print("--- printing & plotting intermediate results for first dataset ---")
        raw = read_raw_bids(path)
        raw.load_data()
        raw.set_channel_types({"EOG1": "eog", "EOG2": "eog", "EOG3": "eog", "EOG4": "eog"})
        raw.pick("eeg")
        if "VREF" in raw.ch_names: raw.drop_channels("VREF")
        raw.filter(1.0, 45.0)
        raw.set_eeg_reference("average")

        # independent component analysis to get rid of eye blinks, heart beats, line noise, etc
        ica = ICA(n_components=0.99, method="infomax", fit_params=dict(extended=True), random_state=0)
        ica.fit(raw)
        labels = label_components(raw, ica, method="iclabel")
        ica_exclusions = []
        for index, (label, probability) in enumerate(zip(labels["labels"], labels["y_pred_proba"])):
            if label not in ("brain", "other") and probability > 0.8:
                if verbose: print(f"removing component {index}: {label} (probability = {probability})")
                ica_exclusions.append(index)
        ica.exclude = ica_exclusions
        ica.apply(raw)

        sampling_frequency = raw.info["sfreq"]
        burst_min_samples = int(np.ceil(0.1 * sampling_frequency))
        trim = int(1.0 * sampling_frequency)
        electrode_rows = []
        for electrode in ELECTRODES_OF_INTEREST:
            spectrum = raw.compute_psd(picks=electrode, method="welch", fmin=13, fmax=35, n_fft=int(4 * sampling_frequency))
            psds, freqs = spectrum.get_data(return_freqs=True)
            peaks, _ = find_peaks(psds[0])
            if len(peaks) > 0:
                peak_index = peaks[np.argmax(psds[0][peaks])]
                peak_found = True
            else:
                peak_index = np.argmax(psds[0])
                peak_found = False
            peak_freq = freqs[peak_index]
            bandpass_low_freq = peak_freq - HALF_BANDWIDTH
            bandpass_high_freq = peak_freq + HALF_BANDWIDTH

            signal = raw.get_data(picks=electrode)[0]
            signal = sosfiltfilt(
                butter(
                    4,
                    [bandpass_low_freq, bandpass_high_freq],
                    btype="bandpass",
                    fs=sampling_frequency,
                    output="sos"
                ),
                signal
            )                                                 # band pass filter
            signal = (signal - signal.mean()) / signal.std()  # z-score normalisation
            envelope = np.abs(hilbert(signal))                # amplitude envelope
            signal, envelope = signal[trim:-trim], envelope[trim:-trim] # trim first and last second

            thresholds = np.percentile(envelope, PERCENTILES)
            duration_profile = []
            for threshold in thresholds:
                samples_above_threshold = np.concatenate(([0], (envelope > threshold).astype(int), [0]))
                threshold_crossings = np.diff(samples_above_threshold)
                burst_starts = np.flatnonzero(threshold_crossings == 1)
                burst_ends = np.flatnonzero(threshold_crossings == -1)
                long_enough = (burst_ends - burst_starts) >= burst_min_samples
                burst_starts, burst_ends = burst_starts[long_enough], burst_ends[long_enough]
                average_burst_duration = (
                    ((burst_ends - burst_starts) / sampling_frequency).mean() if len(burst_starts) > 0
                    else np.nan
                )
                duration_profile.append(average_burst_duration)

            row = {
                "subject": subject,
                "task": task,
                "electrode": electrode,
                "peak_found": peak_found,
                "peak_freq": peak_freq,
                "n_ica_removed": len(ica_exclusions),
            }
            for percentile, average_burst_duration in zip(PERCENTILES, duration_profile):
                row[str(percentile)] = average_burst_duration
            electrode_rows.append(row)

            if verbose:
                print("--- map of electrodes ---")
                raw.plot_sensors(kind="topomap", show_names=True)
                print("--- intermediate results ---")
                _, axs = plt.subplots(3, 1, figsize=(8, 8), layout="constrained")
                # spectrum.plot()
                # plot power spectral density
                axs[0].set_title("power spectral density")
                axs[0].plot(freqs, psds[0])
                axs[0].set_xlabel("frequency (Hz)"); axs[0].set_ylabel("power (dB/Hz re 1 µV^2)");# axs[0].legend()
                axs[0].axvspan(bandpass_low_freq, bandpass_high_freq, color="yellow", alpha=0.3, zorder=0)
                axs[0].axvline(bandpass_low_freq, color="r", ls="-", lw=0.5)
                axs[0].axvline(bandpass_high_freq, color="r", ls="-", lw=0.5)
                axs[0].axvline(peak_freq, color="k", ls=":", lw=1.5)
                # plot filtered signal and amplitude envelope
                axs[1].set_title("filtered signal & amplitude envelope")
                axs[1].set_xlabel("time (s)"); axs[1].set_ylabel("normalised z-score"); axs[1].legend()
                samples = 1000
                axs[1].plot(np.arange(samples) / sampling_frequency, signal[:samples], label="filtered signal")
                axs[1].plot(np.arange(samples) / sampling_frequency, envelope[:samples], color="black", lw=2, label="amplitude envelope")
                # plot burst duration profile
                axs[2].set_title("burst duration profile")
                axs[2].plot(PERCENTILES, duration_profile)
                axs[2].set_xlabel("percentile of max z-score"); axs[2].set_ylabel("mean burst duration (s)");# axs[0].legend()
                plt.show()
                verbose = False # only plot the first one
                break
            break
        # dataframe_rows.extend(electrode_rows)
        # pd.DataFrame(dataframe_rows).to_csv("duration_profiles.csv", index=False)
        break
    except Exception as e:
        print(f"FAILED {subject} {task}: {e}")
        continue

# %%


_, axs = plt.subplots(3, 1, figsize=(8, 8), layout="constrained")
# spectrum.plot()
# plot power spectral density
axs[0].set_title("power spectral density")
axs[0].plot(freqs, psds[0])
axs[0].set_xlabel("frequency (Hz)"); axs[0].set_ylabel("power (dB/Hz re 1 µV^2)");# axs[0].legend()
axs[0].axvspan(bandpass_low_freq, bandpass_high_freq, color="yellow", alpha=0.3, zorder=0)
axs[0].axvline(bandpass_low_freq, color="r", ls="-", lw=0.5, label="bandpass filter bounds (+/-3Hz)")
axs[0].axvline(bandpass_high_freq, color="r", ls="-", lw=0.5)
axs[0].axvline(peak_freq, color="k", ls=":", lw=1.5, label="power spectrum peak frequency")
axs[0].legend()
# plot filtered signal and amplitude envelope
axs[1].set_title("filtered signal & amplitude envelope")
axs[1].set_xlabel("time (s)"); axs[1].set_ylabel("normalised z-score"); 
samples = 1000
axs[1].plot(np.arange(samples) / sampling_frequency, signal[:samples], label="filtered signal")
axs[1].plot(np.arange(samples) / sampling_frequency, envelope[:samples], color="black", lw=2, label="amplitude envelope")
axs[1].legend()
# plot burst duration profile
axs[2].set_title("burst duration profile")
axs[2].plot(PERCENTILES, duration_profile)
axs[2].set_xlabel("percentile of max z-score"); axs[2].set_ylabel("mean burst duration (s)");# axs[0].legend()
plt.show()
# %%
