import torch
from PIL import Image
from siglip_nllb import Tokenizer
from datasets import DatasetDict
from typing import Dict, List, Any

class DatasetProcessor:
    def __init__(self, train_size: float = 0.8, test_size: float = 0.1):
        self.train_size = train_size
        self.test_size = test_size
        self.tokenizers = {}  # Cache for tokenizers
        
    def get_tokenizer(self, lang_code: str) -> Tokenizer:
        """Get or create tokenizer for a language"""
        if lang_code not in self.tokenizers:
            self.tokenizers[lang_code] = Tokenizer(lang_code)
        return self.tokenizers[lang_code]
        
    def tokenize(self, example: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Tokenize a single example"""
        tokenizer = self.get_tokenizer(example['lang_code'])
        image = Image.open(f"data/Images/{example['image_id']}").convert("RGB")
        return tokenizer(image, example['caption'])

    def collate_batch(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Collate batch with proper padding per language"""
        pixel_values = torch.stack([item["pixel_values"] for item in batch]).squeeze(1)
        input_ids = [item["input_ids"].squeeze(0) for item in batch]
        attention_mask = [item["attention_mask"].squeeze(0) for item in batch]
        
        # Get language codes and pad tokens for this batch
        lang_codes = [item.get("lang_code", "eng_Latn") for item in batch]  # Default to English if not found
        pad_tokens = [self.get_tokenizer(lang).text_tokenizer.pad_token_id for lang in lang_codes]
        
        # Pad sequences using language-specific pad tokens
        max_len = max(len(ids) for ids in input_ids)
        padded_input_ids = torch.stack([
            torch.cat([ids, torch.full((max_len - len(ids),), pad_tokens[i])]) 
            for i, ids in enumerate(input_ids)
        ])
        
        padded_attention_mask = torch.nn.utils.rnn.pad_sequence(
            attention_mask, batch_first=True, padding_value=0
        )

        return {
            "pixel_values": pixel_values,
            "input_ids": padded_input_ids,
            "attention_mask": padded_attention_mask
        }

    def transform_dataset(self, dataset) -> DatasetDict:
        """Transform dataset to include language codes and captions"""
        dataset = dataset.remove_columns(["id"])
        lang_columns = [col for col in dataset.column_names if col not in ["image_id"]]

        def transform_row(batch):
            new_image_ids = []
            new_lang_codes = []
            new_captions = []

            for image_id, row in zip(batch["image_id"], zip(*[batch[col] for col in lang_columns])):
                for lang_code, caption in zip(lang_columns, row):
                    new_image_ids.append(image_id)
                    new_lang_codes.append(lang_code)
                    new_captions.append(caption)

            return {
                "image_id": new_image_ids,
                "lang_code": new_lang_codes,
                "caption": new_captions,
            }

        dataset = dataset.map(transform_row, batched=True, remove_columns=lang_columns)
        
        # Split dataset
        train_test = dataset.train_test_split(train_size=self.train_size, test_size=self.test_size)
        test_valid = train_test["test"].train_test_split(train_size=0.5)

        return DatasetDict({
            "train": train_test["train"],
            "test": test_valid["train"],
            "validation": test_valid["test"]
        })
    def process(self, dataset) -> DatasetDict:
        """Complete dataset processing pipeline"""
        transformed = self.transform_dataset(dataset)
        tokenized = transformed.map(self.tokenize)
        # Convert to PyTorch tensors
        for split in tokenized.keys():
            tokenized[split].set_format("torch", columns=["pixel_values", "input_ids", "attention_mask"])
        return tokenized