import os

PROJECT_DIR = os.path.split(os.path.dirname(os.path.abspath(__file__)))[0]
DATASETS_DIR = os.path.join(PROJECT_DIR, 'datasets')
MODELS_DIR = os.path.join(PROJECT_DIR, 'models')
NETWORK_DEMS = 64
TIME_STEPS = 6
CROP_CLASSES = 2
N_CHANNELS = 2
TRAINING_LOOPS = 1
MIN_TRAINING_SAMPLES = 20000
AUGMENTATION_PROBABILITY = 0.5
BATCH_SIZE = 32