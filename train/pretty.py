import os
import time
import random
from dotenv import load_dotenv
import wandb
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from tqdm import tqdm
from dataloader.pretty_loader import create_dataloaders
from dataloader.loader import to_tokens, draw_tokens, pad_id
from model.pixel_transformer import PixelTransformer
from constants import (
    BASE_CHECKPOINT_FOLDER,
    PRETTY_CHECKPOINT_FOLDER,
    EPOCHS,
    BATCH_SIZE,
    LEARNING_RATE,
    MIN_LEARNING_RATE,
    WEIGHT_DECAY,
    DEVICE,
)

random.seed(42)
torch.manual_seed(42)
np.random.seed(42)

def initialise_wandb():
    load_dotenv()
    wandb_api_key = os.getenv('WANDB_API_KEY')
    wandb.login(key=wandb_api_key)
    wandb.init(project='pretty-pixel-transformer', config={
        'epochs': EPOCHS,
        'learning_rate': LEARNING_RATE,
    })

def get_base_model_checkpoint():
    return sorted(list(BASE_CHECKPOINT_FOLDER.iterdir()))[-1]

def train(model, batch, optimizer, criterion):
    model.train()
    encoder_input_ids, decoder_input_ids, label_ids, encoder_pad, decoder_pad = batch
    encoder_input_ids, decoder_input_ids, label_ids = encoder_input_ids.to(DEVICE), decoder_input_ids.to(DEVICE), label_ids.to(DEVICE)
    optimizer.zero_grad()
    logits = model(encoder_input_ids, decoder_input_ids, encoder_pad, decoder_pad)
    loss = criterion(logits.view(-1, logits.size(-1)), label_ids.view(-1))
    loss.backward()
    optimizer.step()
    return loss.item()

def validate(model, val_loader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            encoder_input_ids, decoder_input_ids, label_ids, encoder_pad, decoder_pad = batch
            encoder_input_ids, decoder_input_ids, label_ids = encoder_input_ids.to(DEVICE), decoder_input_ids.to(DEVICE), label_ids.to(DEVICE)
            logits = model(encoder_input_ids, decoder_input_ids, encoder_pad, decoder_pad)
            loss = criterion(logits.view(-1, logits.size(-1)), label_ids.view(-1))
            total_loss += loss.item()

            if batch_idx == 0:
                original_token_ids = encoder_input_ids[0].cpu().tolist()
                original_tokens = to_tokens(original_token_ids)
                original_img = draw_tokens(original_tokens)

                label_token_ids = label_ids[0].cpu().tolist()
                label_tokens = to_tokens(label_token_ids)
                label_img = draw_tokens(label_tokens)

                pred_token_ids = logits[0].argmax(dim=-1).cpu().tolist()
                pred_tokens = to_tokens(pred_token_ids)
                pred_img = draw_tokens(pred_tokens)
                
                combined_img = np.concatenate([np.array(original_img), np.array(label_img), np.array(pred_img)], axis=1)
                wandb.log({
                    "validation_images": wandb.Image(combined_img, caption="Original vs Label vs Predicted")
                })

    average_loss = total_loss / len(val_loader)
    return average_loss

def save_checkpoint(model, optimizer, scheduler, epoch, train_loss, val_loss):
    if not PRETTY_CHECKPOINT_FOLDER.exists():
        PRETTY_CHECKPOINT_FOLDER.mkdir(parents=True, exist_ok=True)

    checkpoint_path = PRETTY_CHECKPOINT_FOLDER / f'epoch_{epoch}_{val_loss:.2f}'
    torch.save({
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss
    }, checkpoint_path)

    print(f"Checkpoint saved: {checkpoint_path}")

def reset():
    import shutil

    wandb_dir = wandb.run.dir if wandb.run else 'wandb'
    if os.path.exists(wandb_dir):
        shutil.rmtree(wandb_dir)
        print(f"Cleared wandb folder: {wandb_dir}")

    if PRETTY_CHECKPOINT_FOLDER.exists():
        shutil.rmtree(PRETTY_CHECKPOINT_FOLDER)
        print(f"Cleared pretty checkpoint folder: {PRETTY_CHECKPOINT_FOLDER}")

def main():
    initialise_wandb()

    print(f"Training on device: {DEVICE}")

    train_loader, val_loader = create_dataloaders(BATCH_SIZE)

    model = PixelTransformer()
    model_checkpoint = torch.load(get_base_model_checkpoint(), weights_only=True)['model']
    model.load_state_dict(model_checkpoint)
    model.to(DEVICE)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=MIN_LEARNING_RATE)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=pad_id)

    start_time = time.time()
    for epoch in range(EPOCHS):
        epoch_start_time = time.time()
        epoch_loss = 0

        with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{EPOCHS}") as pbar:
            for batch in train_loader:
                loss = train(model, batch, optimizer, criterion)
                epoch_loss += loss
                pbar.update(1)
                pbar.set_postfix({'loss': f'{loss:.4f}'})

        avg_train_loss = epoch_loss / len(train_loader)
        val_loss = validate(model, val_loader, criterion)

        epoch_time = time.time() - epoch_start_time
        total_time = time.time() - start_time
        estimated_time_remaining = (EPOCHS - epoch - 1) * epoch_time

        wandb.log({
            'epoch': epoch,
            'train_loss': avg_train_loss,
            'val_loss': val_loss,
            'learning_rate': scheduler.get_last_lr()[0],
            'epoch_time': epoch_time,
            'total_time': total_time,
            'estimated_time_remaining': estimated_time_remaining
        })

        print(f"Epoch {epoch+1}/{EPOCHS}, Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}")
        print(f"Epoch Time: {epoch_time:.2f}s, Total Time: {total_time:.2f}s, Est. Time Remaining: {estimated_time_remaining:.2f}s")

        scheduler.step()

        if (epoch + 1) % 1 == 0 or epoch == EPOCHS - 1:
            save_checkpoint(model, optimizer, scheduler, epoch, avg_train_loss, val_loss)

    total_training_time = time.time() - start_time
    print(f"Total training time: {total_training_time:.2f}s")
    wandb.log({"total_training_time": total_training_time})

    wandb.finish()

if __name__ == '__main__':
    main()