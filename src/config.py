import os

PROJECT_DIR = os.path.split(os.path.dirname(os.path.abspath(__file__)))[0]
DATASETS_DIR = os.path.join(PROJECT_DIR, 'datasets')
MODELS_DIR = os.path.join(PROJECT_DIR, 'models')
NETWORK_DEMS = 256
TIME_STEPS = 32
MIN_TIME_STEPS = 12
CROP_CLASSES = 2
N_CHANNELS = 2
TRAINING_LOOPS = 1
MIN_TRAINING_SAMPLES = 1600
AUGMENTATION_PROBABILITY = 0.5
BATCH_SIZE = 16