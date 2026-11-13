#%%
from mne.preprocessing import ICA
from mne_icalabel import label_components
from mne_bids import BIDSPath, read_raw_bids

DATASET_PATH = "ds007526"
ELECTRODES_OF_INTEREST = ["C1", "C3", "C5", "C2", "C4", "C6", "FC1", "FC3", "FC2", "FC4", "CP1", "CP2"]

path = BIDSPath(subject="019", task="walk", datatype="eeg", root=DATASET_PATH)
raw = read_raw_bids(path)
raw.load_data()
raw.set_channel_types({"EOG1": "eog", "EOG2": "eog", "EOG3": "eog", "EOG4": "eog"})
raw.pick("eeg")
if "VREF" in raw.ch_names: raw.drop_channels("VREF")
raw.filter(1.0, 45.0)
raw.set_eeg_reference("average")

#%%
ica = ICA(n_components=0.99, method="infomax", fit_params=dict(extended=True), random_state=0)
ica.fit(raw)
labels = label_components(raw, ica, method="iclabel")
ica_exclusions = []
for index, (label, probability) in enumerate(zip(labels["labels"], labels["y_pred_proba"])):
    if label not in ("brain", "other") and probability > 0.8:
        print(f"{index=}, {label=}, {probability=}")
        ica_exclusions.append(index)
ica.exclude = ica_exclusions
raw_before = raw.copy()
ica.apply(raw)

ica.plot_components(picks=ica.exclude)
ica.plot_properties(raw_before, picks=ica.exclude)      # topography, spectrum, activity
ica.plot_sources(raw_before, picks=ica.exclude, start=10, stop=40)   # time courses (s)

# %%
print(ica.exclude)
# %%
import matplotlib.pyplot as plt
import numpy as np

comp = ica.exclude[0]
sources = ica.get_sources(raw_before).get_data()        # (n_components, n_times)
peak = np.argmax(np.abs(sources[comp]))
fs = raw.info["sfreq"]
lo, hi = max(0, int(peak - 2 * fs)), int(peak + 2 * fs)  # 4 s window
t = np.arange(lo, hi) / fs
channel = "Fp1"                                           # frontal, shows blinks best

before = raw_before.get_data(picks=channel)[0, lo:hi] * 1e6   # volts -> µV
after = raw.get_data(picks=channel)[0, lo:hi] * 1e6

plt.figure(figsize=(10, 3))
plt.plot(t[:len(before)], before, color="0.6", label="before ICA", lw=2)
plt.plot(t[:len(after)], after, color="C0", label="after ICA", lw=2)
plt.title(f"Example peak of ICA component {comp} ({labels['labels'][comp]})")
plt.xlabel("time (s)"); plt.ylabel("µV"); plt.legend()
plt.show()

# %%
