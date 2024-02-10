import enum
import torch
import sys
import os

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Common
NUM_CLASSES = 4
DATA_TYPE = 'EGG'

# Model constants
NUM_WORKERS = 0
BATCH_SIZE = 64
EPOCHS = 13
LEARNING_RATE = 1e-4
LR_STEP_SIZE = 5

# Fully Connected Network and CNN Feature
NUM_MFCC_FEATURES = 52 # Changed from 50 to 52

# DATASET SPECIFIC

# We take 2 seconds of each audio file
SAMPLE_FREQ = 254

# Time window
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

class SupportedModels(enum.Enum):
    FFNN = 0
    CNN = 1
    VGG = 2