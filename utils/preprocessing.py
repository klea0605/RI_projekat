import mne
from mne.decoding import CSP
import numpy as np

# Svaki trial je hronoloski konzistentan. Izmedju 1s i 3s se radi ono sto nas zanima; fs je sampling frequency default 256 call ugl 254
def Select_time_window(X, t_start, t_end, fs = 256):
    
    t_max=X.shape[2]
    start = max(round(t_start * fs), 0)
    end = min(round(t_end * fs), t_max)

    #Copy interval
    X = X[:, :, start:end]
    return X


# Vraca filtriran dataset
def bandpass_filter(bands, sample_freq, X, device):

    X_return = []

    n_jobs = 'cuda' if device == "cuda" else None

    for band_number, band in enumerate(bands):
                freq_low = band[0]
                freq_high = band[1]
                X_filtered = mne.filter.filter_data(X, sample_freq, freq_low, freq_high, n_jobs = n_jobs, verbose = False)


                # Stack features
                if band_number == 0:
                    X_return = X_filtered
                else:
                    X_return = np.hstack([X_return, X_filtered])

    return X_return

# Razbija na train i test
# Nakon ovoga su podaci tabelarni 
def csp_transform(data_train, y_train, data_test, csp):
  Data_train = []
  Data_test = []

  Data_train = csp.fit_transform(data_train, y_train)
  Data_test = csp.transform(data_test)

  return Data_train, Data_test