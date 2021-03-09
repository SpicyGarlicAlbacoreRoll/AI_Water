import os

PROJECT_DIR = os.path.split(os.path.dirname(os.path.abspath(__file__)))[0]
DATASETS_DIR = os.path.join(PROJECT_DIR, 'datasets')
MODELS_DIR = os.path.join(PROJECT_DIR, 'models')
NETWORK_DEMS = 1024
TIME_STEPS = 64
MIN_TIME_STEPS = 12
CROP_CLASSES = 2
N_CHANNELS = 2
TRAINING_LOOPS = 1
MIN_TRAINING_SAMPLES = 2000
AUGMENTATION_PROBABILITY = 0.25
BATCH_SIZE = 1
NUM_FILTERS = 32

#python3 main.py test V7.0.depth4_f32_b8_64s_jaccard WA_2018_1024
# V7.0.depth4_f32_b8_64s_jaccard_e25