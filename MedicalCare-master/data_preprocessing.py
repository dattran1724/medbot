import pandas as pd
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict, load_dataset

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('wordnet')

def load_and_preprocess_data():
    """Load and preprocess the medical QA dataset"""
    # Load dataset
    # dataset = load_dataset("keivalya/MedQuad-MedicalQnADataset")
    dataset = load_dataset("medalpaca/medical_meadow_wikidoc")
    df = dataset['train'].to_pandas()
    
    # Clean text columns
    df = clean_text_columns(df)
    
    # Analyze data (optional steps)
    # analyze_question_types(df)
    # analyze_answer_lengths(df)
    
    # Clean the dataset
    df_cleaned, removed_records = remove_missing_records(df)
    df_cleaned, removed_records = remove_duplicates_and_log(df_cleaned, removed_records)
    
    # Remove stopwords
    df_cleaned = remove_stopwords_from_df(df_cleaned)
    
    return df_cleaned

def clean_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Clean string columns by lowercasing and normalizing spaces"""
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].str.lower().str.split().str.join(' ')
    return df

def analyze_question_types(df: pd.DataFrame) -> None:
    """Analyze and visualize question types"""
    unique_qtypes = df['qtype'].unique()
    
    # Display the distribution of question types
    qtype_distribution = df['qtype'].value_counts()
    
    # Plot the distribution
    plt.figure(figsize=(8, 5))
    qtype_distribution.plot(kind='bar', color='skyblue')
    plt.title('Distribution of Question Types')
    plt.xlabel('Question Type')
    plt.ylabel('Number of Questions')
    plt.xticks(rotation=45, ha='right')
    plt.show()
    
    # Display the unique question types
    print("Unique Question Types:", unique_qtypes)

def analyze_answer_lengths(df: pd.DataFrame) -> None:
    """Analyze and visualize answer lengths"""
    df['Answer_Length_Words'] = df['Answer'].str.split().apply(len)
    
    # Visualize the distribution of answer lengths
    plt.figure(figsize=(6, 4))
    plt.hist(df['Answer_Length_Words'], bins=100, color='salmon', edgecolor='black')
    plt.title('Answer Length Distribution (Words)')
    plt.xlabel('Number of Words')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.show()

def remove_missing_records(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Remove records with missing values"""
    removed_records = df[df.isnull().any(axis=1)].copy()
    df_cleaned = df.dropna().copy()
    return df_cleaned, removed_records

def remove_duplicates_and_log(df: pd.DataFrame, removed_records: pd.DataFrame,
                              subset_cols: list = ['Question', 'Answer'],
                              log_file: str = 'removed_records_audit.csv') -> tuple[pd.DataFrame, pd.DataFrame]:
    """Remove duplicate records and log them"""
    duplicates = df[df.duplicated(subset=subset_cols, keep=False)].copy()
    removed_records = pd.concat([removed_records, duplicates], ignore_index=True)
    
    df_cleaned = df.drop_duplicates(subset=subset_cols).copy()
    
    if not removed_records.empty:
        removed_records.to_csv(log_file, index=False)
    
    return df_cleaned, removed_records

def remove_stopwords_from_df(df: pd.DataFrame) -> pd.DataFrame:
    """Remove stopwords from Question and Answer columns"""
    # Define a set of question words to retain
    question_words = {'who', 'what', 'where', 'when', 'why', 'how', 'is', 'are'}
    
    # Define stopwords excluding question words
    stop_words = set(stopwords.words('english')) - question_words
    
    # Remove stopwords
    def remove_stopwords(text):
        return " ".join([word for word in text.split() if word not in stop_words])
    
    df['Question'] = df['Question'].apply(remove_stopwords)
    df['Answer'] = df['Answer'].apply(remove_stopwords)
    return df

def combine_text(df):
    """Combine question and answer into a single text field"""
    combined_sequences = df.apply(lambda row: f"<question>{row['Question']}<answer>{row['Answer']}<end>", axis=1)
    return combined_sequences

def prepare_datasets(df_cleaned):
    """Split the data and prepare datasets for training"""
    train_df, temp_df = train_test_split(df_cleaned, test_size=0.2, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
    
    print(f"Training set size: {train_df.shape[0]}")
    print(f"Validation set size: {val_df.shape[0]}")
    
    train_df["text"] = combine_text(train_df)
    val_df["text"] = combine_text(val_df)
    test_df["text"] = combine_text(test_df)
    
    train_dataset = Dataset.from_pandas(train_df[["text", "Question", "Answer"]])
    val_dataset = Dataset.from_pandas(val_df[["text", "Question", "Answer"]])
    test_dataset = Dataset.from_pandas(test_df[["text", "Question", "Answer"]])
    
    dataset_dict = DatasetDict({
        "train": train_dataset,
        "validation": val_dataset,
        "test": test_dataset
    })
    
    return dataset_dict, dataset_dict
