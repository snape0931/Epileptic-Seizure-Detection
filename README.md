# Epileptic Seizure Detection

Detecting epileptic seizures from EEG signals. Each signal is decomposed into modes (EMD and VMD so far, MVMD planned), features are extracted from the modes, and a classifier separates the recordings into seizure, non-seizure epilepsy and normal classes (3-class), or into the five original Bonn sets (5-class).

## Dataset

[Bonn University EEG dataset](https://ukbonn.de/epileptologie/arbeitsgruppen/ag-lehnertz-neurophysik/downloads) (Andrzejak et al., 2001). It has 5 sets (A-E) with 100 single-channel recordings each. Every recording has 4096 samples (23.6 s, sampling rate about 173.61 Hz).

| Set | Folder | Description |
|---|---|---|
| A | Z | Healthy volunteers, eyes open |
| B | O | Healthy volunteers, eyes closed |
| C | N | Interictal (between seizures), hippocampus opposite the epileptogenic zone |
| D | F | Interictal (between seizures), epileptogenic zone |
| E | S | Seizure (ictal) activity |

### Class grouping

| Class | Sets | Meaning | Recordings |
|---|---|---|---|
| Seizure | E | Seizure activity | 100 |
| Non-seizure (epilepsy patient) | C + D | Interictal activity | 200 |
| Normal subject | A + B | Healthy controls | 200 |

## Pipeline

```
Load data -> Preprocess -> Decompose -> Extract features -> Feature matrix -> Classify -> Compare methods
```

### 1. Load data
All 500 recordings are read into an array of shape (500, 4096) with set labels. Files in folder N use the `.TXT` extension (uppercase), so the loader matches extensions without regard to case.

### 2. Preprocessing
4th-order Butterworth low-pass filter at 40 Hz, applied to every signal (zero-phase, `filtfilt`). The raw data covers 0.5 to 85 Hz, and the dataset authors treat the 40 Hz filter as the first analysis step.

### 3. Signal decomposition
Each signal is split into K = 6 modes. Modes are ordered from fast to slow (Mode 1 = highest frequency) for every method, so that mode k means the same thing across methods.

| Method | Full name | Status | Notes |
|---|---|---|---|
| EMD | Empirical Mode Decomposition | Done | Data-driven. IMFs run from fast to slow. First 6 IMFs kept (padded with zeros if fewer; the residue is dropped). |
| VMD | Variational Mode Decomposition | Done | Univariate, alpha = 2000, tau = 0, DC = 0, tol = 1e-6. Modes are sorted by center frequency, then reversed so Mode 1 is the fastest (checked by zero-crossing counts). |
| CEEMDAN | Complete Ensemble EMD with Adaptive Noise | Planned | Improved EMD variant. |
| MVMD | Multivariate VMD | Planned | Not yet implemented. |

Output of each method has shape (500 recordings, 6 modes, 4096 samples) and is cached as `modes_emd.npy` and `modes_vmd.npy`. Delete the files to recompute (VMD takes about 8 minutes, EMD about 1 minute).

An EMD vs VMD comparison is plotted for one recording per set, including the absolute difference per mode and a table of MSE, correlation and energy ratio per set. These plots are visual only and do not show which method is better for seizure detection.

### 4. Feature extraction
12 features are computed for each of the 6 modes, giving a feature matrix of shape (500, 72) per method.

| Group | Features |
|---|---|
| Statistical | mean, std, skewness, kurtosis |
| Amplitude / energy | energy, peak-to-peak |
| Time-domain | zero-crossing rate, Hjorth mobility, Hjorth complexity |
| Entropy | permutation entropy (order 3), spectral entropy (Welch), sample entropy (m = 2, r = 0.2 std) |

Column names follow the pattern `m{mode}_{feature}`, for example `m1_std`. Constant (all-zero) modes get 0 for the features that are undefined. Matrices are saved to `features/features_emd.csv` and `features/features_vmd.csv`, with the `set` and `label3` columns added.

### 5. Classification
Standardization (`StandardScaler`) followed by one of three classifiers, evaluated with stratified 10-fold cross-validation (shuffled, seed 0):

- SVM (RBF kernel, C = 10)
- Random Forest (300 trees)
- KNN (k = 5)

Two tasks are run: 5-class (sets A-E) and 3-class (Normal / Interictal / Seizure). Accuracy and macro-F1 are saved to `classification_results.csv`, and a classification report and confusion matrix are produced for the best combination per task.

## Results

5-class (A-E), accuracy in %:

| Decomposition | SVM | RF | KNN |
|---|---|---|---|
| EMD | 78.6 | 85.0 | 67.4 |
| VMD | 86.4 | 86.6 | 75.6 |

3-class (Normal / Interictal / Seizure): best is VMD + SVM at 98.0% (EMD + RF reaches 97.0%).

Most errors in the 5-class task are between sets C and D, the two interictal sets, which differ only in recording location. Seizure recordings (set E) are classified with about 97% precision and recall.

These numbers use the same cross-validation folds for model selection and reporting, so they are slightly optimistic. Bonn also does not identify subjects, so the results do not show generalization to new patients.

## Run

Open `sop.ipynb` in Google Colab and use Run All. The notebook installs its dependencies (`EMD-signal`, `vmdpy`, `scikit-learn`, `scipy`, `joblib` and others), clones this repository for the data, and runs the whole pipeline in about 12-15 minutes on the first run. Later runs are faster because the decompositions are cached.

## Next steps

- Add CEEMDAN and MVMD decompositions and compare them with EMD and VMD.
- Extend the feature library and check feature importance per mode.
- Validate with label-shuffle, repeated-CV and held-out tests.
