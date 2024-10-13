import torch, os
from dataset import load_data
from torch.utils.data import DataLoader, Dataset
from torch import nn, optim
from torch.optim.lr_scheduler import StepLR
from PIL import Image
import pandas as pd
from siglip_nllb import SiglipNllb, Tokenizer

# Define a custom dataset class for image-caption pairs using DataFrames
class MultilingualImageCaptionDataset(Dataset):
    def __init__(self, df, image_folder, tokenizer_dict):
        self.data = df
        self.image_folder = image_folder
        self.tokenizer_dict = tokenizer_dict

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image_path = os.path.join(self.image_folder, row['image_id'])   
        caption = row['caption']
        language = row['language_code']

        tokenizer = self.tokenizer_dict[language]
        image = Image.open(image_path).convert("RGB")
        tokenized_data = tokenizer(image, caption)

        pixel_values = tokenized_data["pixel_values"].squeeze(0)
        input_ids = tokenized_data["input_ids"].squeeze(0)
        attention_mask = tokenized_data["attention_mask"].squeeze(0)
        labels = input_ids.clone()

        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }
# Training function with gradient accumulation
def train(model, dataloader, optimizer, scheduler, device, grad_accumulation_steps):
    model.train()
    total_loss = 0

    optimizer.zero_grad()
    for step, batch in enumerate(dataloader):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(batch)
        loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        loss = loss_fct(outputs.view(-1, outputs.size(-1)), batch["labels"].view(-1))
        loss = loss / grad_accumulation_steps
        loss.backward()

        if (step + 1) % grad_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        total_loss += loss.item() * grad_accumulation_steps

    return total_loss / len(dataloader)

# Evaluation function with metrics computation using `pycocoevalcap` for METEOR and CIDEr
def evaluate(model, dataloader, tokenizer_dict, device):
    model.eval()
    total_loss = 0
    generated_captions = []
    ground_truth_captions = []

    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(batch)

            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(outputs.view(-1, outputs.size(-1)), batch["labels"].view(-1))
            total_loss += loss.item()

            predictions = torch.argmax(outputs, dim=-1)
            for i in range(predictions.size(0)):
                language = batch['language'][i]
                tokenizer = tokenizer_dict[language]

                pred_caption = tokenizer.detokenize(predictions[i])
                generated_captions.append(pred_caption)
                ground_truth = tokenizer.detokenize(batch['labels'][i])
                ground_truth_captions.append(ground_truth)

    return total_loss / len(dataloader)

if __name__ == "__main__":
    # Configuration
    image_folder = "data/Images"
    batch_size = 8
    epochs = 3
    learning_rate = 5e-5
    grad_accumulation_steps = 4

    # Set of unique languages from your dataset
    languages = {'eng', 'dyu', 'kmb', 'ibo', 'lin', 'cjk', 'kab', 'kon', 'bem', 'lug', 'dik', 'kam', 'afr', 'lua',
                 'kin', 'yor', 'hau', 'kik', 'ewe', 'amh', 'fuv'}
    tokenizer_dict = {lang: Tokenizer(target_lang=lang) for lang in languages}

    # Load datasets using your custom `load_data` function
    dataset_splits = load_data("AfriMM/AfriMMD") 

    train_df = pd.DataFrame(dataset_splits['train'])
    val_df = pd.DataFrame(dataset_splits['validation'])
    test_df = pd.DataFrame(dataset_splits['test'])

    # Initialize datasets and dataloaders using the DataFrames
    train_dataset = MultilingualImageCaptionDataset(train_df, image_folder, tokenizer_dict)
    val_dataset = MultilingualImageCaptionDataset(val_df, image_folder, tokenizer_dict)
    test_dataset = MultilingualImageCaptionDataset(test_df, image_folder, tokenizer_dict)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SiglipNllb()
    model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    total_steps = (len(train_loader) // grad_accumulation_steps) * epochs
    scheduler =  StepLR(optimizer, step_size=10, gamma=0.1)

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        train_loss = train(model, train_loader, optimizer, scheduler, device, grad_accumulation_steps)
        val_loss = evaluate(model, val_loader, tokenizer_dict, device)
        print(f"Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")
       

    test_loss = evaluate(model, test_loader, tokenizer_dict, device)

    torch.save(model.state_dict(), "siglip_nllb_image_captioning_model.pth")
