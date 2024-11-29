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
from dataloader.loader import create_dataloaders
from utils.tokens import to_tokens, draw_tokens
from model.pixel_transformer import PixelTransformer
from constants import (
    BASE_CHECKPOINT_FOLDER,
    EPOCHS,
    BATCH_SIZE,
    LEARNING_RATE,
    MIN_LEARNING_RATE,
    WEIGHT_DECAY,
    DEVICE,
    PAD_ID
)

random.seed(42)
torch.manual_seed(42)
np.random.seed(42)


def initialise_wandb():
    load_dotenv()
    wandb_api_key = os.getenv('WANDB_API_KEY')
    wandb.login(key=wandb_api_key)
    wandb.init(project='pixel-transformer', config={
        'epochs': EPOCHS,
        'learning_rate': LEARNING_RATE,
    })


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

            if batch_idx != 0:
                continue

            batch_size = encoder_input_ids.size(0)
            if batch_size <= 2:
                print("Batch size is less than 2, cannot select two random images.")
                continue

            random_indices = random.sample(range(batch_size), 2)
            
            combined_images = []
            for idx in random_indices:
                original_token_ids = encoder_input_ids[idx].cpu().tolist()
                original_tokens = to_tokens(original_token_ids)
                original_img = draw_tokens(original_tokens)

                label_token_ids = label_ids[idx].cpu().tolist()
                label_tokens = to_tokens(label_token_ids)
                label_img = draw_tokens(label_tokens)

                pred_token_ids = logits[idx].argmax(dim=-1).cpu().tolist()
                pred_tokens = to_tokens(pred_token_ids)
                pred_img = draw_tokens(pred_tokens)
                
                combined_img = np.concatenate([np.array(original_img), np.array(label_img), np.array(pred_img)], axis=1)
                combined_images.append(combined_img)
            
            final_combined_img = np.concatenate(combined_images, axis=0)
            
            wandb.log({
                "validation_images": wandb.Image(final_combined_img, caption="Original vs Label vs Predicted (2 random samples)")
            })

    avg_val_loss = total_loss / len(val_loader)
    return avg_val_loss


def save_checkpoint(model, optimizer, scheduler, epoch, train_loss, val_loss):
    if not BASE_CHECKPOINT_FOLDER.exists():
        BASE_CHECKPOINT_FOLDER.mkdir(parents=True, exist_ok=True)

    checkpoint_path = BASE_CHECKPOINT_FOLDER / f'epoch_{epoch}_{val_loss:.2f}.pth'
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

    if BASE_CHECKPOINT_FOLDER.exists():
        shutil.rmtree(BASE_CHECKPOINT_FOLDER)
        print(f"Cleared base checkpoint folder: {BASE_CHECKPOINT_FOLDER}")


def load_latest_checkpoint(model, optimizer, scheduler):
    checkpoints = list(BASE_CHECKPOINT_FOLDER.glob('*.pth'))
    if not checkpoints:
        print("No checkpoints found. Starting from scratch.")
        return 0, 0, 0
    
    latest_checkpoint = max(checkpoints, key=os.path.getctime)
    print(f"Loading checkpoint: {latest_checkpoint}")
    
    checkpoint = torch.load(latest_checkpoint, weights_only=True)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])
    
    return checkpoint['epoch'] + 1, checkpoint['train_loss'], checkpoint['val_loss']


def main():
    reset()
    initialise_wandb()

    print(f"Training on device: {DEVICE}")

    train_loader, val_loader = create_dataloaders(batch_size=BATCH_SIZE)

    model = PixelTransformer().to(DEVICE)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=MIN_LEARNING_RATE)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=PAD_ID)

    start_epoch, last_train_loss, last_val_loss = load_latest_checkpoint(model, optimizer, scheduler)

    start_time = time.time()
    global_step = start_epoch * len(train_loader)

    for epoch in range(start_epoch, EPOCHS):
        epoch_start_time = time.time()
        epoch_loss = 0

        with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{EPOCHS}") as pbar:
            for batch in train_loader:
                loss = train(model, batch, optimizer, criterion)
                epoch_loss += loss
                global_step += 1

                wandb.log({
                    'step': global_step,
                    'train_loss': loss,
                    'learning_rate': scheduler.get_last_lr()[0]
                })

                pbar.update(1)
                pbar.set_postfix({'loss': f'{loss:.4f}'})

        avg_train_loss = epoch_loss / len(train_loader)
        val_loss = validate(model, val_loader, criterion)

        epoch_time = time.time() - epoch_start_time
        total_time = time.time() - start_time
        estimated_time_remaining = (EPOCHS - epoch - 1) * epoch_time

        wandb.log({
            'epoch': epoch,
            'avg_train_loss': avg_train_loss,
            'val_loss': val_loss,
            'epoch_time': epoch_time,
            'total_time': total_time,
            'estimated_time_remaining': estimated_time_remaining
        })

        print(f"Epoch {epoch+1}/{EPOCHS}, Avg Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}")
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