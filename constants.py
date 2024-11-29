from pathlib import Path
import torch

RAW_COORD_FOLDER = Path('data') / 'raw_coord'

IAM_FOLDER = Path('data') / 'iam'
RAW_COORD_FOLDER = Path('data') / 'iam_raw_coord'
BASE_CHECKPOINT_FOLDER = Path('checkpoints') / 'base'

PRETTY_COORD_FOLDER = Path('data') / 'pretty_coord'
PRETTY_CHECKPOINT_FOLDER = Path('checkpoints') / 'pretty'

IMG_H = 64
IMG_W = 128

PAD_TOKEN = '<pad>'
START_TOKEN = '<start>'
MOVE_TOKEN = '<move>'
DRAW_TOKEN = '<draw>'
END_TOKEN = '<end>'

PAD_ID = 0
START_ID = 1
MOVE_ID = 2
DRAW_ID = 3
END_ID = 4

SPECIAL_TOKENS = [PAD_TOKEN, START_TOKEN, MOVE_TOKEN, DRAW_TOKEN, END_TOKEN]
TOKEN_COUNT = IMG_H * IMG_W + len(SPECIAL_TOKENS)

MODEL_DIM = 512
MAX_SEQ_LEN = 400
FFN_RATIO = 2
ATTENTION_HEADS = 8
LAYERS = 4
DROPOUT = 0.1

BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 1e-4
MIN_LEARNING_RATE = 1e-6
WEIGHT_DECAY = 1e-5

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')