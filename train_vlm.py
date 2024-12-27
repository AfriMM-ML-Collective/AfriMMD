import wandb
import torch
from torch import nn
from PIL import Image
from tqdm.auto import tqdm
from typing import Dict, Any
from siglip_nllb import SiglipNllb, Tokenizer
from transformers import AdamW, get_scheduler
import argparse, json, datetime, math, os, csv
from datasets import load_dataset, DatasetDict
from data_preprocessing import transform_dataset, data_collator, Tokenize_function

def get_args_parser():
    parser = argparse.ArgumentParser('AFRIMMD Pretraining script', add_help=True)
    parser.add_argument('--batch-size', default=16, type=int)
    parser.add_argument('--epochs', default=80, type=int)
    parser.add_argument('--device', default='mps', type=str)
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--max_grad_norm', default=1.0, type=float)
    parser.add_argument('--warmup_steps', default=1000, type=int)
    parser.add_argument('--save_steps', default=1000, type=int)
    parser.add_argument('--wandb_logging', action='store_true')
    parser.add_argument('--output_dir', type=str, required=True, 
                       help='Directory where model checkpoints and logs will be saved')
    parser.add_argument('--freeze_vision_encoder', action='store_true')
    parser.add_argument('--freeze_language_decoder', action='store_true')
    return parser

def compute_training_metrics(logits: torch.Tensor, targets: torch.Tensor, loss: torch.Tensor) -> Dict[str, float]:
    """
    Compute pretraining-specific metrics
    """
    with torch.no_grad():
        # Perplexity
        perplexity = torch.exp(loss)
        
        # Vision-Language alignment score (cosine similarity between vision and language features)
        pred_probs = torch.softmax(logits, dim=-1)
        accuracy = (torch.argmax(pred_probs, dim=-1) == targets).float().mean()
        
        # Vocabulary usage statistics
        vocab_usage = torch.sum(pred_probs > 0.1, dim=-1).float().mean()
        
    return {
        'loss': loss.item(),
        'perplexity': perplexity.item(),
        'accuracy': accuracy.item(),
        'vocab_usage': vocab_usage.item()
    }
    
def setup_logging(output_dir: str):
    """Setup training and validation log files"""
    os.makedirs(output_dir, exist_ok=True)
    
    train_log_path = os.path.join(output_dir, 'train_metrics.csv')
    val_log_path = os.path.join(output_dir, 'val_metrics.csv')
    
    # Create CSV files with headers
    metrics = ['epoch', 'loss', 'perplexity', 'accuracy', 'vocab_usage', 'grad_norm', 'learning_rate']
    
    for path in [train_log_path, val_log_path]:
        if not os.path.exists(path):
            with open(path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(metrics)
    
    return train_log_path, val_log_path

def log_metrics(log_path: str, epoch: int, metrics: Dict[str, float], lr: float = None):
    """Log metrics to CSV file"""
    row = [epoch, 
           metrics['loss'], 
           metrics['perplexity'], 
           metrics['accuracy'], 
           metrics['vocab_usage'],
           metrics.get('grad_norm', 0),
           lr if lr is not None else 0]
    
    with open(log_path, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(row)

def save_checkpoint(output_dir: str, model, optimizer, epoch: int, metrics: Dict[str, float], 
                   is_best: bool = False):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics
    }
    
    if is_best:
        save_path = os.path.join(output_dir, 'best_model.pt')
        print(f"\n=== Saving new best model at epoch {epoch+1} with validation loss: {metrics['loss']:.4f} ===")
    else:
        save_path = os.path.join(output_dir, f'checkpoint_epoch_{epoch+1}.pt')
        print(f"\nSaving checkpoint for epoch {epoch+1}")
        
    torch.save(checkpoint, save_path)
    
def train_one_epoch(args, model, train_data, epoch, optimizer, lr_scheduler):
    model.train()
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    epoch_metrics = {
        'loss': 0,
        'perplexity': 0,
        'accuracy': 0,
        'vocab_usage': 0,
        'grad_norm': 0}
    num_steps = len(train_data)
    
    pbar = tqdm(enumerate(train_data), total=num_steps, 
                desc=f"Epoch {epoch+1}/{args.epochs}")
    
    for step, batch in pbar:
        batch = {k: v.to(args.device) for k, v in batch.items()}
        
        # Forward pass
        outputs = model(batch)
        logits = outputs.view(-1, outputs.size(-1))
        targets = batch["input_ids"].view(-1)
        
        loss = loss_fn(logits, targets)
        
        # Compute training metrics
        step_metrics = compute_training_metrics(logits, targets, loss)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        step_metrics['grad_norm'] = grad_norm.item()
        
        optimizer.step()
        lr_scheduler.step()
        
        # Update metrics
        for k, v in step_metrics.items():
            epoch_metrics[k] += v
        
        # MPS synchronization
        if args.device == 'mps':
            torch.mps.synchronize()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f"{step_metrics['loss']:.4f}",
            'perp': f"{step_metrics['perplexity']:.4f}",
            'lr': f"{lr_scheduler.get_last_lr()[0]:.7f}"})
        
        # Logging
        if args.wandb_logging:
            wandb.log(step_metrics)
    
    # Average epoch metrics
    epoch_metrics = {k: v/num_steps for k, v in epoch_metrics.items()}
    return epoch_metrics, lr_scheduler.get_last_lr()[0]

def evaluate(args, model, eval_dataloader):
    model.eval()
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    eval_metrics = {
        'loss': 0,
        'perplexity': 0,
        'accuracy': 0,
        'vocab_usage': 0
    }
    num_steps = len(eval_dataloader)
    
    with torch.no_grad():
        pbar = tqdm(eval_dataloader, desc="Evaluating")
        for batch in pbar:
            batch = {k: v.to(args.device) for k, v in batch.items()}
            
            outputs = model(batch)
            logits = outputs.view(-1, outputs.size(-1))
            targets = batch["input_ids"].view(-1)
            
            loss = loss_fn(logits, targets)
            
            # Compute metrics
            step_metrics = compute_training_metrics(logits, targets, loss)
            
            # Update metrics
            for k, v in step_metrics.items():
                eval_metrics[k] += v
            
            if args.device == 'mps':
                torch.mps.synchronize()
    
    # Average metrics
    eval_metrics = {k: v/num_steps for k, v in eval_metrics.items()}
    return eval_metrics

def main(args):
    # Initialize wandb if enabled
    if args.wandb_logging:
        wandb.init(
            project="afrimmd-pretraining",
            config=vars(args),
            dir=args.output_dir
        )
    
    # Setup logging
    train_log_path, val_log_path = setup_logging(args.output_dir)
    
    # Model initialization
    model = SiglipNllb()
    model.to(args.device)
    
    # Freeze parts of the model for pretraining
    if args.freeze_vision_encoder:
        for param in model.vit.parameters():
            param.requires_grad = False
    if args.freeze_language_decoder:
        for param in model.lm.parameters():
            param.requires_grad = False
    
    # Always keep connector layer trainable
    model.connector.requires_grad = True
    
    # Data loading
    raw_data = load_dataset("AfriMM/AfriMMD")
    raw_data = raw_data['train'].select(range(2))  # Remove this line when using script
    data = transform_dataset(raw_data)
    tokenized_data = data.map(Tokenize_function)
    tokenized_data.set_format("pt", columns=["pixel_values", "input_ids", "attention_mask"])
    
    # Create dataloaders
    train_data = torch.utils.data.DataLoader(
        tokenized_data["train"],
        shuffle=True,
        batch_size=args.batch_size,
        collate_fn=data_collator
    )
    val_data = torch.utils.data.DataLoader(
        tokenized_data["validation"],
        batch_size=args.batch_size,
        collate_fn=data_collator
    )
    test_data = torch.utils.data.DataLoader(
        tokenized_data["test"],
        batch_size=args.batch_size,
        collate_fn=data_collator
    )
    
    # Optimization setup
    optimizer = AdamW(model.parameters(), lr=5e-5)
    num_training_steps = args.epochs * len(train_data)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=num_training_steps
    )
    
    print(f"Starting SIGLIP and NLLB Pretraining on AFRIMMD dataset")
    print(f"Outputs will be saved to: {args.output_dir}")
    best_val_loss = float('inf')
    
    # Training loop
    for epoch in range(args.epochs):
        # Training
        train_metrics, lr = train_one_epoch(args, model, train_data, epoch, optimizer, lr_scheduler)
        
        # Log training metrics
        log_metrics(train_log_path, epoch + 1, train_metrics, lr)
        
        # Validation
        val_metrics = evaluate(args, model, val_data)
        
        # Log validation metrics
        log_metrics(val_log_path, epoch + 1, val_metrics)
        
        # Logging
        print(f"\nEpoch {epoch+1} results:")
        print(f"Train metrics: {train_metrics}")
        print(f"Val metrics: {val_metrics}")
        
        # Save best model if validation loss improved
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            save_checkpoint(args.output_dir, model, optimizer, epoch, val_metrics, is_best=True)
        
        # Save last epoch model
        if epoch == args.epochs - 1:
            save_checkpoint(args.output_dir, model, optimizer, epoch, val_metrics, is_best=False)
    
    # Final evaluation
    if args.eval:
        test_metrics = evaluate(args, model, test_data)
        print(f"Final test metrics: {test_metrics}")
        
        # Save test metrics
        test_log_path = os.path.join(args.output_dir, 'test_metrics.json')
        with open(test_log_path, 'w') as f:
            json.dump(test_metrics, f, indent=4)

if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)

