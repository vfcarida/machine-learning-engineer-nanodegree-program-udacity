import re
import pandas as pd
import operator
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_datatype(df: pd.DataFrame, 
                    train_value: int, 
                    test_value: int, 
                    datatype_var: str, 
                    compare_dfcolumn: str, 
                    operator_of_compare: Any, 
                    value_of_compare: Any,
                    sampling_number: int, 
                    sampling_seed: int) -> None:
    """
    Labels records in the dataframe as training or test based on stratified random sampling.
    
    Args:
        df: The main dataframe to modify.
        train_value: Value to assign for training records.
        test_value: Value to assign for test records.
        datatype_var: Column name where labels will be stored.
        compare_dfcolumn: Column to use for subsetting.
        operator_of_compare: Operator (e.g. operator.gt) to compare compare_dfcolumn with value_of_compare.
        value_of_compare: Value to compare against.
        sampling_number: Number of samples to take per group for the test set.
        sampling_seed: Random seed for reproducibility.
    """
    # Subsets dataframe by condition
    df_subset = df[operator_of_compare(df[compare_dfcolumn], value_of_compare)].copy()
    
    if datatype_var in df_subset.columns:
        df_subset = df_subset.drop(columns=[datatype_var])
    
    # Sets all datatype to value for training initially
    df_subset.loc[:, datatype_var] = train_value
    
    # Performs stratified random sample to identify test cases
    df_sampled = df_subset.groupby(['Task', compare_dfcolumn], group_keys=False).apply(
        lambda x: x.sample(min(len(x), sampling_number), random_state=sampling_seed)
    )
    
    # Update the main dataframe with the labels
    for index in df_subset.index:
        label = test_value if index in df_sampled.index else train_value
        df.at[index, datatype_var] = label

def train_test_dataframe(clean_df: pd.DataFrame, random_seed: int = 100) -> pd.DataFrame:
    """
    Creates training and test labels for the plagiarism dataset.
    
    Args:
        clean_df: Dataframe containing the cleaned file information.
        random_seed: Random seed for sampling.
        
    Returns:
        pd.DataFrame: Dataframe with an added 'Datatype' column.
    """
    new_df = clean_df.copy()

    # Initialize datatype as 0 (original wiki answers)
    new_df.loc[:, 'Datatype'] = 0

    # Creates test & training datatypes for plagiarized answers (1,2,3)
    create_datatype(new_df, 1, 2, 'Datatype', 'Category', operator.gt, 0, 1, random_seed)

    # Creates test & training datatypes for NON-plagiarized answers (0)
    create_datatype(new_df, 1, 2, 'Datatype', 'Category', operator.eq, 0, 2, random_seed)
    
    # Map numerical labels to descriptive strings
    mapping = {0: 'orig', 1: 'train', 2: 'test'}
    new_df['Datatype'] = new_df['Datatype'].map(mapping)

    return new_df

def process_file(file: Any) -> str:
    """
    Standardizes text by converting to lowercase and removing non-alphanumeric characters.
    
    Args:
        file: A file object to read from.
        
    Returns:
        str: Processed text.
    """
    # Put text in all lower case letters 
    all_text = file.read().lower()

    # Remove all non-alphanumeric chars and collapse whitespace
    all_text = re.sub(r"[^a-zA-Z0-9]", " ", all_text)
    all_text = re.sub(r"\s+", " ", all_text).strip()
    
    return all_text

def create_text_column(df: pd.DataFrame, file_directory: str = 'data/') -> pd.DataFrame:
    """
    Reads in files listed in a dataframe and adds a 'Text' column with processed content.
    
    Args:
        df: Dataframe with a 'File' column.
        file_directory: Directory where files are stored.
        
    Returns:
        pd.DataFrame: Dataframe with an additional 'Text' column.
    """
    text_df = df.copy()
    text_list = []
    
    base_path = Path(file_directory)
    
    for _, row in df.iterrows():
        file_path = base_path / row['File']
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                processed_text = process_file(file)
                text_list.append(processed_text)
        except FileNotFoundError:
            logger.error(f"File not found: {file_path}")
            text_list.append("")
    
    text_df['Text'] = text_list
    return text_df
