from pathlib import Path
from typing import List, Tuple
import random
import copy
import json
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
from dataloader.word_dataclass import Word
from dataloader.augment import WordAugmenter
from utils.tokens import to_id, to_tokens
from constants import RAW_COORD_FOLDER, MAX_SEQ_LEN, BATCH_SIZE, PAD_ID

random.seed(42)
torch.manual_seed(42)

class WordAugDataset(Dataset):
    def __init__(self, raw_coord_folder: Path):
        self.words = []
        for raw_coord_file in list(raw_coord_folder.iterdir()):
            with open(raw_coord_file, 'r') as f:
                data = json.load(f)

            word, tokens = data['word'], data['tokens']
            if len(tokens) > MAX_SEQ_LEN:
                continue
            
            self.words.append(Word(word, tokens))

    def __len__(self):
        return len(self.words)

    def __getitem__(self, idx):
        word_dataclass = self.words[idx]
        word = word_dataclass.word
        augmenter1 = WordAugmenter(copy.deepcopy(word_dataclass))
        augmenter2 = WordAugmenter(copy.deepcopy(word_dataclass))
        augmenter1.random_augment()
        #augmenter2.random_augment()
        augmenter2.normalize()
        augmented_word1 = augmenter1.word
        augmented_word2 = augmenter2.word
        return augmented_word1, augmented_word2, word
    
def collate_fn(batch: List[Tuple[Word, Word, str]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    augmented_words1, augmented_words2, words = zip(*batch)
    
    encoder_input_ids = [to_id(word) for word in augmented_words1]
    decoder_input_ids = [to_id(word) for word in augmented_words2]

    max_encoder_len = max(len(seq) for seq in encoder_input_ids)
    max_decoder_len = max(len(seq) for seq in decoder_input_ids)

    encoder_input_ids_padded = [seq + [PAD_ID] * (max_encoder_len - len(seq)) for seq in encoder_input_ids]
    decoder_input_ids_padded = [seq + [PAD_ID] * (max_decoder_len - len(seq)) for seq in decoder_input_ids]

    label_ids = [seq[1:] for seq in decoder_input_ids_padded]

    encoder_pad = [[1 if id != PAD_ID else 0 for id in seq] for seq in encoder_input_ids_padded]
    decoder_pad = [[1 if id != PAD_ID else 0 for id in seq] for seq in decoder_input_ids_padded]
    
    encoder_input_ids = torch.tensor(encoder_input_ids_padded, dtype=torch.long)
    decoder_input_ids = torch.tensor(decoder_input_ids_padded, dtype=torch.long)
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
    import matplotlib.pyplot as plt
    from collections import Counter
    from utils.tokens import draw_tokens

    train_loader, val_loader = create_dataloaders()
    print(f"Number of training batches: {len(train_loader)}")
    print(f"Number of validation batches: {len(val_loader)}")
    
    max_encoder_len = 0
    max_decoder_len = 0
    max_encoder_tokens = []
    max_decoder_tokens = []
    encoder_lengths = []
    decoder_lengths = []
    
    for batch in train_loader:
        encoder_input_ids, decoder_input_ids, _, encoder_pad, decoder_pad = batch
        
        # Calculate actual lengths by counting non-pad tokens
        batch_encoder_lengths = encoder_pad.sum(dim=1).tolist()
        batch_decoder_lengths = decoder_pad.sum(dim=1).tolist()
        
        # Update max lengths and tokens
        batch_max_encoder_len = max(batch_encoder_lengths)
        if batch_max_encoder_len > max_encoder_len:
            max_encoder_len = batch_max_encoder_len
            max_encoder_tokens = to_tokens(encoder_input_ids[batch_encoder_lengths.index(batch_max_encoder_len)].tolist())
        
        batch_max_decoder_len = max(batch_decoder_lengths)
        if batch_max_decoder_len > max_decoder_len:
            max_decoder_len = batch_max_decoder_len
            max_decoder_tokens = to_tokens(decoder_input_ids[batch_decoder_lengths.index(batch_max_decoder_len)].tolist())
        
        # Collect all lengths
        encoder_lengths.extend(batch_encoder_lengths)
        decoder_lengths.extend(batch_decoder_lengths)
    
    print(f"Largest encoder sequence length: {max_encoder_len}")
    print(f"Largest decoder sequence length: {max_decoder_len}")
    
    # Draw longest sequences
    if max_encoder_tokens:
        draw_tokens(max_encoder_tokens).show()
    if max_decoder_tokens:
        draw_tokens(max_decoder_tokens).show()
    
    # Plot distribution of lengths
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Encoder lengths distribution
    encoder_counter = Counter(encoder_lengths)
    ax1.bar(encoder_counter.keys(), encoder_counter.values())
    ax1.set_title('Distribution of Encoder Sequence Lengths')
    ax1.set_xlabel('Sequence Length')
    ax1.set_ylabel('Frequency')
    
    # Decoder lengths distribution
    decoder_counter = Counter(decoder_lengths)
    ax2.bar(decoder_counter.keys(), decoder_counter.values())
    ax2.set_title('Distribution of Decoder Sequence Lengths')
    ax2.set_xlabel('Sequence Length')
    ax2.set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.show()