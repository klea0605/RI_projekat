import enum
import torch
import sys
import os
from mne.decoding import CSP

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Common
ROOT = '.'
SAVE_DIR = './Results/'
SAVE = True


# DATASET SPECIFIC
NUM_CLASSES = 4
DATA_TYPE = 'EGG'
SUBJECT_NUMBERS = [1, 2]
DATA_TYPE = 'EEG'
CONDITIONS = ['IN']  # samo inner speech klasifikujem ali moze i ostalo
CLASSES = [['All']] # all classes = left, right, up, down

# Time window
SAMPLE_FREQ = 254
START_T = 1.5
END_T = 3.5

# Filter signal band = [low_band , high_band, "band_name"]
bands = [
          (0.5, 4, 'Delta (0.5-4 Hz)')
         ,(4, 8, 'Theta (4-8 Hz)')
         ,(8, 12, 'Alpha (8-12 Hz)')
         ,(12, 20, 'Low Beta (12-20 Hz)')
         ,(20, 30, 'High Beta (20-30 Hz)')
         ,(30, 45, 'Low Gamma (30-45Hz)')
          ]

# Common Spatial Patterns
""" Preuzeto od Nieta """
NUM_COMPONENTS = 6
RANK = None
REG = 'empirical'
LOG = True
NORM_TRACE = False
COV_EST = 'concat'
TRANSFORM_INTO = 'average_power'
csp = CSP(n_components=NUM_COMPONENTS,rank=RANK, reg=REG, log=LOG, norm_trace = NORM_TRACE, cov_est = COV_EST, transform_into = TRANSFORM_INTO)
""" ======================= """

# Model constants
N_SPLITS = 5
BATCH_SIZE = 64
EPOCHS = 13
LEARNING_RATE = 1e-4
LR_STEP_SIZE = 5
