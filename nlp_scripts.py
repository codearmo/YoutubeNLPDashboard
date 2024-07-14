import nltk
import spacy
import pandas as pd
from collections import Counter

# Ensure nltk datasets are downloaded
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
nltk.download('wordnet')

# Load spacy language model
nlp = spacy.load('en_core_web_sm')


def perform_named_entity_recognition(text):
    """
    Perform named entity recognition (NER) on the given text.

    Args:
        text (str): The input text on which to perform NER.

    Returns:
        list: A list of tuples, where each tuple contains the entity text and its label.
    """
    # Use the Spacy language model to process the text
    doc = nlp(text)
    
    # Extract entities and their labels from the processed text
    # ent.text gives the entity text and ent.label_ gives the entity label (e.g., PERSON, ORG, GPE, etc.)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    
    # Return the list of entities and their corresponding labels
    return entities


def count_ner(entities):
    """
    Count the occurrences of each named entity type in the provided list of entities.

    Args:
        entities (list): A list of tuples, where each tuple contains the entity text and its label.

    Returns:
        dict: A dictionary with entity labels as keys and their counts as values.
    """
    ner_count = {}
    for entity, label in entities:
        ner_count[label] = ner_count.get(label, 0) + 1
    return ner_count


def count_entities(df, n=30):

    combined_ner = [] 

    for ner_list in df['ner_list']:
        combined_ner.extend(ner_list)

    counts_ner = Counter(combined_ner)
    top_entities = counts_ner.most_common(n=n)
    entities = [e for e,c in top_entities]
    counter = [c for e,c in top_entities]
    return entities, counter
    

def count_entity_type(df):
    # Combine counts from all rows
    combined_counts = {}
    for ner_count in df['ner_count']:
        for label, count in ner_count.items():
            combined_counts[label] = combined_counts.get(label, 0) + count

    labels = list(combined_counts.keys())
    counts = list(combined_counts.values())

    return labels, counts



# Apply NER counting to each row in the DataFrame
def extract_ner(entities):
    """
    Extract and lowercase named entities from the provided list of entities.

    Args:
        entities (list): A list of tuples, where each tuple contains the entity text and its label.

    Returns:
        list: A list of named entities in lowercase.
    """
    ner_list = [ent[0].lower() for ent in entities]  # Extract named entities
    return ner_list


def apply_ner_functions(df: pd.DataFrame, col='textDisplay') -> dict:
    """
    Apply NER-related functions to a specified column in the DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame.
        col (str): The name of the column on which to perform NER (default is 'textDisplay').

    Returns:
        pd.DataFrame: The DataFrame with additional columns for NER results.
    """
    # Apply the perform_named_entity_recognition function to the specified column
    df['ner'] = df[col].apply(perform_named_entity_recognition)
    
    # Apply the count_ner function to the 'ner' column to count entity occurrences
    df['ner_count'] = df['ner'].apply(count_ner)
    
    # Apply the extract_ner function to the 'ner' column to extract named entities
    df['ner_list'] = df['ner'].apply(extract_ner)
    
    # Return the modified DataFrame
    return df
