from pathlib import Path
from typing import List, Tuple, Union
import random
import copy
import io
import json
import PIL.Image as Image
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
from dataloader.word_dataclass import Word
from dataloader.augment import WordAugmenter
from dataloader.word_dataclass import Coordinate, Start, Pause, End
from constants import RAW_COORD_FOLDER, IMG_W, IMG_H, SPECIAL_TOKENS, TOKEN_COUNT, BATCH_SIZE

random.seed(42)
torch.manual_seed(42)

pad_id = 0
start_id = 1
pause_id = 2
end_id = 3

def to_id(word: Word, target_len: int) -> List[int]:
    token_ids = []
    for token in word.tokens:
        if isinstance(token, Coordinate):
            x, y = token.x, token.y
            token_id = x + IMG_W * y + len(SPECIAL_TOKENS)
            token_ids.append(token_id)
        elif isinstance(token, Start):
            token_ids.append(start_id)
        elif isinstance(token, Pause):
            token_ids.append(pause_id)
        elif isinstance(token, End):
            token_ids.append(end_id)

    for id in token_ids:
        if id > TOKEN_COUNT - 1:
            raise ValueError(f"Token ID {id} exceeds the maximum allowed value of {TOKEN_COUNT}")
        
    return token_ids + [pad_id] * (target_len - len(token_ids))

def to_tokens(ids: List[int]) -> List[Union[Coordinate, Start, Pause, End]]:
    tokens = []
    for id in ids:
        if id == pad_id:
            continue
        elif id == start_id:
            tokens.append(Start())
        elif id == pause_id:
            tokens.append(Pause())
        elif id == end_id:
            tokens.append(End())
        else:
            adjusted_id = id - (len(SPECIAL_TOKENS))
            x = adjusted_id % IMG_W
            y = adjusted_id // IMG_W
            tokens.append(Coordinate(x=x, y=y))
    return tokens

def draw_tokens(tokens: List[Union[Coordinate, Start, Pause, End]]) -> Image.Image:
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_xlim(0, IMG_W)
    ax.set_ylim(IMG_H, 0)
    ax.axis('off')

    fig.patch.set_facecolor('white')

    current_path = []
    for token in tokens:
        if isinstance(token, Coordinate):
            current_path.append((token.x, token.y))
        elif isinstance(token, Pause):
            if current_path:
                x, y = zip(*current_path)
                ax.plot(x, y, 'k-', linewidth=2)
                current_path = []
        elif isinstance(token, End):
            break

    if current_path:
        x, y = zip(*current_path)
        ax.plot(x, y, 'k-', linewidth=2)

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight', pad_inches=0, facecolor='white')
    buf.seek(0)
    img = Image.open(buf)
    
    border_size = 10
    bordered_img = Image.new('RGB', (img.width + 2*border_size, img.height + 2*border_size), color='lightgrey')
    bordered_img.paste(img, (border_size, border_size))

    plt.close(fig)
    return bordered_img
    
class WordAugDataset(Dataset):
    def __init__(self, raw_coord_folder: Path):
        self.words = []
        for raw_coord_file in list(raw_coord_folder.iterdir()):
            with open(raw_coord_file, 'r') as f:
                data = json.load(f)
            self.words.append(Word(data['word'], data['tokens']))

    def __len__(self):
        return len(self.words)

    def __getitem__(self, idx):
        word_dataclass = self.words[idx]
        word = word_dataclass.word
        augmenter1 = WordAugmenter(copy.deepcopy(word_dataclass))
        augmenter2 = WordAugmenter(copy.deepcopy(word_dataclass))
        augmenter1.random_augment()
        augmenter2.random_augment()
        augmented_word1 = augmenter1.word
        augmented_word2 = augmenter2.word
        return augmented_word1, augmented_word2, word
    
def collate_fn(batch: List[Tuple[Word, Word, str]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    augmented_words1, augmented_words2, words = zip(*batch)
    
    encoder_max_len = max(len(w1.tokens) for w1 in augmented_words1)
    decoder_max_len = max(len(w2.tokens) for w2 in augmented_words2)
    
    encoder_input_ids = [to_id(w1, encoder_max_len) for w1 in augmented_words1]
    decoder_input_ids = [to_id(w2, decoder_max_len) for w2 in augmented_words2]

    label_ids = []
    for seq in decoder_input_ids:
        shifted = seq[1:]
        label_ids.append(shifted)
    
    encoder_pad = [
        [1 if id != pad_id else 0 for id in seq]
        for seq in encoder_input_ids
    ]

    decoder_pad = [
        [1 if id != pad_id else 0 for id in seq]
        for seq in decoder_input_ids
    ]
    
    encoder_input_ids = torch.tensor(encoder_input_ids, dtype=torch.long)
    decoder_input_ids = torch.tensor(decoder_input_ids, dtype=torch.long)
    encoder_pad = torch.tensor(encoder_pad, dtype=torch.bool)
    decoder_pad = torch.tensor(decoder_pad, dtype=torch.bool)
    label_ids = torch.tensor(label_ids, dtype=torch.long)

    return encoder_input_ids, decoder_input_ids, label_ids, encoder_pad, decoder_pad

def create_dataloaders(batch_size: int = BATCH_SIZE, train_ratio: float = 0.8) -> Tuple[DataLoader, DataLoader]:
    dataset = WordAugDataset(RAW_COORD_FOLDER)
    dataset_size = len(dataset)
    train_size = int(dataset_size * train_ratio)
    val_size = dataset_size - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, 
        [train_size, val_size]
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    return train_loader, val_loader

if __name__ == '__main__':
    train_loader, val_loader = create_dataloaders()
    print(f"Number of training batches: {len(train_loader)}")
    print(f"Number of validation batches: {len(val_loader)}")
    for batch in val_loader:
        encoder_input_ids, decoder_input_ids, label_ids, encoder_pad, decoder_pad = batch
        print()
        print("Encoder input shape: ", encoder_input_ids.shape)
        print("Decoder input shape: ", decoder_input_ids.shape)
        print("Label shape: ", label_ids.shape)
        print("Encoder pad shape: ", encoder_pad.shape)
        print("Decoder pad shape: ", decoder_pad.shape)
        print()
        print("Encoder input ids:\n", encoder_input_ids)
        print("Decoder input ids:\n", decoder_input_ids)
        print("Label ids:\n", label_ids)
        print("Encoder pad:\n", encoder_pad)
        print("Decoder pad:\n", decoder_pad)
        break