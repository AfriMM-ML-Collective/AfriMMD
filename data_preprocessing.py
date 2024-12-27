import torch
from PIL import Image
from siglip_nllb import Tokenizer
from datasets import DatasetDict

def Tokenize_function(example):
    tokenizer = Tokenizer(example['lang_code'])
    image_ = Image.open(f"data/Images/{example['image_id']}").convert("RGB")
    tokenized_input = tokenizer(image_, example['caption'])
    return tokenized_input

def data_collator(batch):
    """
    Collates a batch of examples into a dictionary of tensors (tokens).
    """
    # Ensure pixel_values have consistent dimensions
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    pixel_values = pixel_values.squeeze(1) 

    # Convert input_ids and attention_mask to tensors and Remove the singleton dimension from input_ids and attention_mask
    input_ids = [item["input_ids"].squeeze(0) for item in batch]
    attention_mask = [item["attention_mask"].squeeze(0) for item in batch]

    # Pad input_ids and attention_mask to the same length
    padded_input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=0)  # Replace 0 with tokenizer.pad_token_id
    padded_attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)

    # Create the tokens dictionary
    tokens = {
        "pixel_values": pixel_values,
        "input_ids": padded_input_ids,
        "attention_mask": padded_attention_mask}

    return tokens

def transform_dataset(dataset):
    """
    Transforms a Hugging Face dataset by dropping 'id', retaining 'image_id',
    and converting language-specific columns to rows with 'lang_code' and 'caption'.

    Args:
        dataset (Dataset): Hugging Face Dataset object.

    Returns:
        Dataset: Transformed Hugging Face Dataset object.
    """
    # Drop the 'id' column
    dataset = dataset.remove_columns(["id"])
    train_size, test_size= 0.8, 0.1
    # Extract language-specific columns
    lang_columns = [col for col in dataset.column_names if col not in ["image_id"]]

    # Create new rows for each language code
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

    # Apply the transformation
    dataset = dataset.map(
        transform_row,
        batched=True,
        remove_columns=lang_columns)
    
    # Split the dataset
    train_test_split = dataset.train_test_split(train_size=train_size, test_size=test_size)
    test_valid_split = train_test_split["test"].train_test_split(train_size=0.5)

    # Combine splits into a DatasetDict
    datasets = DatasetDict({
        "train": train_test_split["train"],
        "test": test_valid_split["train"],
        "validation": test_valid_split["test"]})

    return datasets