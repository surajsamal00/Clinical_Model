# config.py

MODEL_NAME = "facebook/bart-base"

# Training
NUM_EPOCHS = 5
BATCH_SIZE = 4
LEARNING_RATE = 5e-5

# Tokenizer / Input
MAX_INPUT_LEN = 512
MAX_TARGET_LEN = 128

# Data
DATA_PATH = "data/raw/indiana_reports.csv"
TRAIN_SPLIT = 0.8

#Model
MODEL_PATH = "checkpoint_latest.pth"
