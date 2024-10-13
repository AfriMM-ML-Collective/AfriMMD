import pandas as pd, os
from sklearn.utils import shuffle
from datasets import Dataset, load_dataset

def transform_and_shuffle_dataset(df):
    # Convert from wide format to long format
    df = df.drop(columns=["id"])
    df_long = pd.melt(df,id_vars=["image_id"], var_name="language_code",value_name="caption")

    # Extract language code from the original column names
    df_long["language_code"] = df_long["language_code"]

    # Shuffle the dataframe
    df_shuffled = shuffle(df_long, random_state=42)

    return df_shuffled

def load_data(dataset_name):
    data = load_dataset(dataset_name)
    dataset = data['train'].train_test_split(test_size=0.05)
    dataset['validation'] = dataset['test'][:1]
    da = dataset['train'].train_test_split(test_size=0.1)
    dataset['train'] = da['train'][:2]
    dataset['test'] = da['test'][:1]
    #Convert Hugging Face Dataset splits to pandas DataFrames
    train_df = pd.DataFrame(dataset['train'])
    val_df = pd.DataFrame(dataset['validation'])
    test_df = pd.DataFrame(dataset['test'])

    # Use the `transform_and_shuffle_dataset` function to reformat the data
    train_transformed = transform_and_shuffle_dataset(train_df)
    val_transformed = transform_and_shuffle_dataset(val_df)
    test_transformed = transform_and_shuffle_dataset(test_df)
 
    # Reset the index to ensure correct alignment before converting to a Dataset
    train_transformed.reset_index(drop=True, inplace=True)
    val_transformed.reset_index(drop=True, inplace=True)
    test_transformed.reset_index(drop=True, inplace=True)

    # Convert the transformed DataFrames back into Hugging Face Dataset objects
    train_dataset = Dataset.from_pandas(train_transformed)
    val_dataset = Dataset.from_pandas(val_transformed)
    test_dataset = Dataset.from_pandas(test_transformed)
    # Return the datasets in a dictionary format
    return {
        "train": train_dataset,
        "validation": val_dataset,
        "test": test_dataset
    }