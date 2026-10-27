# Epileptic Seizure Detection

Detecting epileptic seizures from EEG signals. Each signal is decomposed into modes with EMD, VMD and MVMD, features are extracted from the modes, and a classifier separates seizure, non-seizure epilepsy and normal recordings. 

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
Each signal is split into K = 6 modes using three methods:

| Method | Full name | Notes |
|---|---|---|
| EMD | Empirical Mode Decomposition | Data-driven. IMFs run from fast to slow. First 6 IMFs kept (padded with zeros if fewer). |
| VMD | Variational Mode Decomposition | alpha = 2000, tau = 0, tol = 1e-6. Modes sorted from slow to fast by center frequency. |
| MVMD | Multivariate VMD | Decomposes several channels together with shared center frequencies. |

**MVMD channel construction.** Bonn recordings are single-channel, so MVMD input is built by bundling recording *i* from each set into one 5-channel group `[A_i, B_i, C_i, D_i, E_i]`, giving 100 bundles. Each bundle holds one recording per class, so the decomposition never sees the labels. Channels are scaled to unit variance before decomposition so the large seizure amplitudes do not dominate, and scaled back afterwards.

Output of each method has shape (500 recordings, 6 modes, 4096 samples).

### 4. Feature extraction (planned)
Feature library computed on every mode:
- Time-based: mean, standard deviation, skewness, kurtosis, energy, peak-to-peak, zero-crossing rate, Hjorth mobility and complexity
- Entropy-based: permutation entropy, spectral entropy, sample entropy

### 5. Feature matrix (planned)
One row per recording, one column per mode-feature pair (6 modes x 12 features = 72 columns), built separately for each decomposition method.

### 6. Classification (planned)
SVM, Random Forest and KNN with stratified 10-fold cross-validation on the 3 classes.

### 7. Method comparison (planned)
Accuracy and per-class recall compared across EMD, VMD and MVMD.

## Progress

- [x] Understand the Bonn dataset and class grouping
- [x] Load all 500 recordings
- [x] Preprocessing: 40 Hz low-pass filter
- [x] Decomposition with EMD, VMD and MVMD
- [ ] Feature library
- [ ] Feature matrix
- [ ] Classification
- [ ] Comparison of decomposition methods

## Setup

```bash
pip install numpy pandas scipy scikit-learn matplotlib joblib tqdm EMD-signal vmdpy pysdkit antropy
```

Open the notebook in Jupyter, VS Code or Google Colab. If you use Colab, clone this repo first so the notebook can read the `DATASET` folder:

```python
!git clone https://github.com/snape0931/Epileptic-Seizure-Detection.git
```

Decomposed `.npy` files are not stored in the repo because of their size (each is about 100 MB). Re-run the notebook to regenerate them.

## References

- Andrzejak R.G. et al. (2001). Indications of nonlinear deterministic and finite-dimensional structures in time series of brain electrical activity. *Phys. Rev. E* 64, 061907.
- Huang N.E. et al. (1998). The empirical mode decomposition and the Hilbert spectrum for nonlinear and non-stationary time series analysis. *Proc. R. Soc. Lond. A* 454, 903-995.
- Dragomiretskiy K., Zosso D. (2014). Variational Mode Decomposition. *IEEE Trans. Signal Process.* 62(3), 531-544.
- Rehman N., Aftab H. (2019). Multivariate Variational Mode Decomposition. *IEEE Trans. Signal Process.* 67(23), 6039-6052.
- Carvalho V.R. et al. (2020). Evaluating five different adaptive decomposition methods for EEG signal seizure detection and classification. *Biomed. Signal Process. Control* 62, 102073.
