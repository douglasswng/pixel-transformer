from pathlib import Path
import torch

RAW_COORD_FOLDER = Path('data') / 'raw_coord'

IAM_FOLDER = Path('data') / 'iam'
RAW_COORD_FOLDER = Path('data') / 'iam_raw_coord'
BASE_CHECKPOINT_FOLDER = Path('checkpoints') / 'base'

PRETTY_COORD_FOLDER = Path('data') / 'pretty_coord'
PRETTY_CHECKPOINT_FOLDER = Path('checkpoints') / 'pretty'

IMG_H = 64
IMG_W = 256

START_TOKEN = '<start>'
PAUSE_TOKEN = '<pause>'
END_TOKEN = '<end>'
PAD_TOKEN = '<pad>'
SPECIAL_TOKENS = [START_TOKEN, PAUSE_TOKEN, END_TOKEN, PAD_TOKEN]
TOKEN_COUNT = IMG_H * IMG_W + len(SPECIAL_TOKENS)

MODEL_DIM = 512
MAX_SEQ_LEN = 2000
FFN_RATIO = 4
ATTENTION_HEADS = 8
LAYERS = 8
DROPOUT = 0.1

BATCH_SIZE = 128
EPOCHS = 15
LEARNING_RATE = 1e-3
MIN_LEARNING_RATE = 1e-5
WEIGHT_DECAY = 1e-5

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')