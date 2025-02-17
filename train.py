from main_model import fullmodel
from dataload import train_loader, test_loader
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DataParallel
import os
from transformers import BertTokenizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
import multiprocessing

# Initialize tokenizer
TOKENIZER = BertTokenizer.from_pretrained('bert-base-uncased')
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Initialize model
model = fullmodel(
    embed_size=300,
    attention_dim=256,
    encoder_dim=2048,
    decoder_dim=512
).to(DEVICE)

# Define loss function (ignoring padding token)
criterion = nn.CrossEntropyLoss(ignore_index=TOKENIZER.pad_token_id)

# Define optimizer with single learning rate definition
LEARNING_RATE = 5e-4  # Initial learning rate
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=0.01,
    betas=(0.9, 0.999)
)

def get_latest_checkpoint(directory):
    checkpoint_files = [f for f in os.listdir(directory) if f.startswith("model_epoch_") and f.endswith(".pth")]
    if not checkpoint_files:
        return None  # No checkpoint found
    checkpoint_files.sort(key=lambda f: int(f.split('_')[-1].split('.')[0]))  # Sort by epoch number
    return os.path.join(directory, checkpoint_files[-1])  # Return latest checkpoint path

# Add learning rate scheduler - removing verbose parameter
scheduler = ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,
    patience=3,
    min_lr=1e-6
)

def load_checkpoint(model, optimizer, scheduler, checkpoint_path):
    """
    Load checkpoint with compatibility for both old and new formats.
    Old format: Just model state dict
    New format: Dictionary with model, optimizer, and scheduler states
    """
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)  # Added weights_only=True
    
    # Check if checkpoint is old format (direct state dict) or new format (dictionary)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        # New format
        model.load_state_dict(checkpoint['model_state_dict'])
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        epoch = checkpoint.get('epoch', -1)
    else:
        # Old format - just model state dict
        model.load_state_dict(checkpoint)
        epoch = -1
    
    return epoch

def train(epochs, model, train_loader, test_loader, optimizer, criterion, tokenizer, scheduler,start_epoch):
    """
    Train the image captioning model.
    """
    os.makedirs('checkpoints', exist_ok=True)
    best_loss = float('inf')
    
    for epoch in range(start_epoch,epochs):
        model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        current_lr = optimizer.param_groups[0]['lr']
        print(f"\nEpoch {epoch+1}/{epochs} - Learning Rate: {current_lr:.6f}")
        
        for idx, (image, caption) in enumerate(train_loader):
            optimizer.zero_grad()
            image, caption = image.to(DEVICE), caption.to(DEVICE)

            # Forward pass
            outputs, attention_weights = model(image, caption)
            outputs = outputs[:, :-1, :]
            targets = caption[:, 1:]
            outputs_flat = outputs.reshape(-1, tokenizer.vocab_size)
            targets_flat = targets.reshape(-1)
            
            # Compute loss
            loss = criterion(outputs_flat, targets_flat)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            # Compute accuracy ignoring padding
            _, predicted = outputs.max(2)
            mask = targets != tokenizer.pad_token_id
            correct_predictions += (predicted == targets).masked_select(mask).sum().item()
            total_predictions += mask.sum().item()
            
            # Print training stats
            if (idx + 1) % 200 == 0:
                avg_loss = running_loss / 200
                accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
                print(f"Step [{idx+1}/{len(train_loader)}], "
                      f"Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
                running_loss = 0.0
                correct_predictions = 0
                total_predictions = 0
        
        # Validation phase
        model.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0
        
        with torch.inference_mode():
            for image, caption in test_loader:
                image, caption = image.to(DEVICE), caption.to(DEVICE)

                outputs, attention_weights = model(image, caption)
                outputs = outputs[:, :-1, :]
                targets = caption[:, 1:]
                outputs_flat = outputs.reshape(-1, tokenizer.vocab_size)
                targets_flat = targets.reshape(-1)
                
                loss = criterion(outputs_flat, targets_flat)
                test_loss += loss.item()
                
                _, predicted = outputs.max(2)
                mask = targets != tokenizer.pad_token_id
                test_correct += (predicted == targets).masked_select(mask).sum().item()
                test_total += mask.sum().item()
        
        avg_test_loss = test_loss / len(test_loader)
        test_accuracy = test_correct / test_total if test_total > 0 else 0
        
        # Step the scheduler based on validation loss
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(avg_test_loss)
        new_lr = optimizer.param_groups[0]['lr']
        
        if new_lr != old_lr:
            print(f"Learning rate decreased from {old_lr:.6f} to {new_lr:.6f}")
        
        print(f"Validation - Loss: {avg_test_loss:.4f}, Accuracy: {test_accuracy:.4f}")
        
        # Save checkpoint with full state
        if avg_test_loss < best_loss:
            best_loss = avg_test_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': avg_test_loss,
            }, 'checkpoints/best_model.pth')
            print("Saved new best model")
            
        if (epoch + 1) % 5 == 0:  # Save regular checkpoint every 5 epochs
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': avg_test_loss,
            }, f"checkpoints/model_epoch_{epoch+1}.pth")
            print(f"Saved checkpoint at epoch {epoch+1}")

# Main execution
EPOCHS = 80

if __name__ == '__main__':
    multiprocessing.freeze_support()
    latest_checkpoint = get_latest_checkpoint("checkpoints")
    
    if latest_checkpoint:
        print(f"Loading checkpoint: {latest_checkpoint}")
        start_epoch = load_checkpoint(model, optimizer, scheduler, latest_checkpoint)
        if start_epoch >= 0:
            print(f"Resuming from epoch {start_epoch + 1}")
        else:
            print("Loaded model state only (old format checkpoint)")
    else:
        print("No checkpoint found. Training from scratch.")
    
    train(EPOCHS, model, train_loader, test_loader, optimizer, criterion, TOKENIZER, scheduler,start_epoch)