import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from typing import Tuple, List
import numpy as np

def drop_columns(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """
    Drop specified columns from the dataframe.
    """
    return df.drop(columns=columns)

def split_data(df: pd.DataFrame, target: str, test_size: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Split the dataframe into training and validation sets.
    """
    train_df, val_df = train_test_split(df, test_size=test_size, random_state=random_state)
    train_targets = train_df[target]
    val_targets = val_df[target]
    return train_df, val_df, train_targets, val_targets

def scale_numeric_features(df: pd.DataFrame, numeric_features: List[str], scaler: StandardScaler = None) -> Tuple[pd.DataFrame, StandardScaler]:
    """
    Scale numeric features using StandardScaler.

    Args:
    - df (pd.DataFrame): The dataframe with numeric features.
    - numeric_features (List[str]): List of numeric feature names.
    - scaler (StandardScaler): The scaler to use. If None, a new scaler will be created and fitted.

    Returns:
    - df_scaled (pd.DataFrame): The dataframe with scaled numeric features.
    - scaler (StandardScaler): The fitted scaler.
    """
    if scaler is None:
        scaler = StandardScaler()
        df[numeric_features] = scaler.fit_transform(df[numeric_features])
    else:
        df[numeric_features] = scaler.transform(df[numeric_features])
    return df, scaler

def encode_categorical_features(df: pd.DataFrame, categorical_features: List[str], encoder: OneHotEncoder = None) -> Tuple[pd.DataFrame, OneHotEncoder]:
    """
    Encode categorical features using OneHotEncoder.

    Args:
    - df (pd.DataFrame): The dataframe with categorical features.
    - categorical_features (List[str]): List of categorical feature names.
    - encoder (OneHotEncoder): The encoder to use. If None, a new encoder will be created and fitted.

    Returns:
    - df_encoded (pd.DataFrame): The dataframe with encoded categorical features.
    - encoder (OneHotEncoder): The fitted encoder.
    """
    if encoder is None:
        encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False) #drop='first'
        encoded_features = encoder.fit_transform(df[categorical_features])
    else:
        encoded_features = encoder.transform(df[categorical_features])
    
    encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(categorical_features))
    df = df.drop(columns=categorical_features).reset_index(drop=True)
    df_encoded = pd.concat([df, encoded_df], axis=1)
    return df_encoded, encoder

def get_input_columns(df: pd.DataFrame, target: str) -> Tuple[List[str], List[str], List[str]]:
    """
    Get numeric and categorical feature names from the dataframe.
    """
    numeric_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    numeric_features.remove(target)
    categorical_features = df.select_dtypes(include=['object']).columns.tolist()
    features = numeric_features + categorical_features
    return features, numeric_features, categorical_features

def preprocess_data(raw_df: pd.DataFrame, scaler_numeric: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str], StandardScaler, OneHotEncoder]:
    """
    Preprocess the raw dataframe and return train and validation sets along with
    scalers and encoders used.

    Args:
    - raw_df (pd.DataFrame): The raw data frame containing the bank churn data.
    - scaler_numeric (bool): Whether to scale numeric features. Default is True.

    Returns:
    - X_train (np.ndarray): Preprocessed training features.
    - train_targets (np.ndarray): Training targets.
    - X_val (np.ndarray): Preprocessed validation features.
    - val_targets (np.ndarray): Validation targets.
    - input_cols (List[str]): List of input columns used for the features.
    - scaler (StandardScaler): Scaler used for numeric features (None if not used).
    - encoder (OneHotEncoder): Encoder used for categorical features.
    """
    # Drop the 'Surname' column
    df = drop_columns(raw_df, ['Surname','CustomerId'])

    # Define target and features
    target = 'Exited'
    features, numeric_features, categorical_features = get_input_columns(df, target)

    # Split the data into training and validation sets
    train_df, val_df, train_targets, val_targets = split_data(df, target)

    # Scale numeric features
    if scaler_numeric:
        train_df, scaler = scale_numeric_features(train_df, numeric_features)
        val_df, _ = scale_numeric_features(val_df, numeric_features, scaler)
    else:
        scaler = None

    # Encode categorical features
    train_df, encoder = encode_categorical_features(train_df, categorical_features)
    val_df, _ = encode_categorical_features(val_df, categorical_features, encoder)

    # Prepare the final datasets
    X_train = train_df.drop(columns=[target])
    X_val = val_df.drop(columns=[target])

    return {
        'train_X': X_train,
        'train_y': train_targets,
        'val_X': X_val,
        'val_y': val_targets,
        'numeric_cols': numeric_features, 
        'categorical_cols': categorical_features,
        'scaler': scaler, 
        'encoder': encoder
    }

def preprocess_new_data(new_df: pd.DataFrame, input_cols: List[str], numeric_features: List[str], categorical_features: List[str], scaler: StandardScaler, encoder: OneHotEncoder, scaler_numeric: bool = True) -> np.ndarray:
    """
    Preprocess new data using the provided scalers and encoders.

    Args:
    - new_df (pd.DataFrame): The new data to preprocess.
    - input_cols (List[str]): List of input columns used for the features.
    - numeric_features (List[str]): List of numeric feature names.
    - categorical_features (List[str]): List of categorical feature names.
    - scaler (StandardScaler): Scaler used for numeric features (None if not used).
    - encoder (OneHotEncoder): Encoder used for categorical features.

    Returns:
    - new_data (np.ndarray): Preprocessed new data.
    """

    # Ensure the new data contains the input columns
    new_df = new_df[input_cols]

    #Drop the 'Surname' column
    new_df = drop_columns(new_df, ['Surname','CustomerId'])
    
    # Scale numeric features
    if scaler_numeric:
        new_df, _ = scale_numeric_features(new_df, numeric_features, scaler)
    else:
        scaler = None

    # Encode categorical features
    new_df, _ = encode_categorical_features(new_df, categorical_features, encoder)

    return {'new_df': new_df
    }
