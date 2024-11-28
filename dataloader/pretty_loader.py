from pathlib import Path
from typing import List, Tuple
import random
import json
import copy
import torch
from torch.utils.data import Dataset, DataLoader
from dataloader.word_dataclass import Word
from dataloader.augment import WordAugmenter
from constants import PRETTY_COORD_FOLDER, BATCH_SIZE

random.seed(42)
torch.manual_seed(42)

from dataloader.loader import pad_id, to_id

class PrettyWordDataset(Dataset):
    def __init__(self, pretty_coord_folder: Path):
        self.word_pairs = []
        for pretty_coord_file in pretty_coord_folder.iterdir():
            with open(pretty_coord_file, 'r') as f:
                data = json.load(f)
            original_word = Word(data['word'], data['original_tokens'])
            pretty_word = Word(data['word'], data['pretty_tokens'])
            self.word_pairs.append((original_word, pretty_word))

    def __len__(self):
        return len(self.word_pairs)

    def __getitem__(self, idx):
        original_word, pretty_word = self.word_pairs[idx]
        augmenter1 = WordAugmenter(copy.deepcopy(original_word))
        augmenter2 = WordAugmenter(copy.deepcopy(pretty_word))
        augmenter1.normalize()
        augmenter2.normalize()
        original_word = augmenter1.word
        pretty_word = augmenter2.word
        return original_word, pretty_word
    
def collate_fn(batch: List[Tuple[Word, Word]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    original_words, pretty_words = zip(*batch)
    
    encoder_max_len = max(len(w1.tokens) for w1 in original_words)
    decoder_max_len = max(len(w2.tokens) for w2 in pretty_words)
    
    encoder_input_ids = [to_id(w1, encoder_max_len) for w1 in original_words]
    decoder_input_ids = [to_id(w2, decoder_max_len) for w2 in pretty_words]

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
    dataset = PrettyWordDataset(PRETTY_COORD_FOLDER)
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