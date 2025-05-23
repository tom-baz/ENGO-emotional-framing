import pandas as pd
import numpy as np
from typing import Dict, List, Tuple

def load_and_filter_data(cop26_path: str, cop27_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load COP datasets and filter for relevant tweets only."""
    df_cop26 = pd.read_excel(cop26_path)
    df_cop27 = pd.read_excel(cop27_path)
    
    # Filter for relevant tweets only
    df_cop26_rel = df_cop26[df_cop26['Relevant'] == 1].copy()
    df_cop27_rel = df_cop27[df_cop27['Relevant'] == 1].copy()
    
    return df_cop26_rel, df_cop27_rel

def process_datetime_and_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """Process datetime and create sentiment indicator columns."""
    df = df.copy()
    
    # Parse datetime
    df[['date', 'hour']] = df['created_at'].str.split(" ", expand=True)
    df['date'] = pd.to_datetime(df['date'])
    
    # Create sentiment indicator columns
    df['Positive'] = (df['label'] == 'positive').astype(int)
    df['Negative'] = (df['label'] == 'negative').astype(int)
    df['Neutral'] = (df['label'] == 'neutral').astype(int)
    
    return df

def aggregate_by_group_and_date(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate sentiment data by date, name, and group."""
    # Group by date, name, and group
    grouped = df.groupby(['date', 'Name', 'Group']).agg({
        'Positive': 'sum',
        'Negative': 'sum',
        'Neutral': 'sum'
    }).reset_index()
    
    # Calculate percentages and derived metrics
    grouped['sum_twits'] = grouped['Positive'] + grouped['Negative'] + grouped['Neutral']
    
    # Calculate percentages
    for sentiment in ['negative', 'neutral', 'positive']:
        col_name = sentiment.capitalize()
        grouped[f'{sentiment}_percent'] = (grouped[col_name] / grouped['sum_twits']) * 100
    
    # Calculate sentiment index
    grouped['Sentiment Index'] = ((grouped['Positive'] - grouped['Negative']) / 
                                 grouped['sum_twits'])
    
    # Add day formatting and numbering
    grouped['day'] = grouped['date'].dt.strftime('%d-%m')
    grouped['day_number'] = grouped['date'].rank(method='dense').astype(int)
    
    return grouped

def create_group_datasets(grouped_df: pd.DataFrame) -> List[pd.DataFrame]:
    """Create separate datasets for each group and aggregate by date."""
    group_datasets = []
    
    # Define which columns to average (only numeric columns)
    numeric_columns = ['Positive', 'Negative', 'Neutral', 'negative_percent', 
                      'neutral_percent', 'positive_percent', 'Sentiment Index', 
                      'sum_twits', 'day_number']
    
    for group_num in range(1, 5):  # Groups 1-4
        group_data = grouped_df[grouped_df['Group'] == group_num]
        # Only average the numeric columns
        group_aggregated = group_data.groupby('date')[numeric_columns].mean().reset_index()
        group_datasets.append(group_aggregated)
    
    return group_datasets

def process_cop_data(cop26_path: str, cop27_path: str) -> Tuple[List[pd.DataFrame], List[pd.DataFrame], pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Complete data processing pipeline for COP26 and COP27 data.
    
    Returns:
        - COP26 group datasets (list of 4 DataFrames)
        - COP27 group datasets (list of 4 DataFrames) 
        - COP26 overall aggregated data
        - COP27 overall aggregated data
        - COP26 grouped data (organization-level daily data)
        - COP27 grouped data (organization-level daily data)
        - COP26 filtered relevance data (tweet-level data)
        - COP27 filtered relevance data (tweet-level data)
    """
    # Load and filter data
    df_cop26, df_cop27 = load_and_filter_data(cop26_path, cop27_path)
    
    # Process datetime and sentiment
    df_cop26_processed = process_datetime_and_sentiment(df_cop26)
    df_cop27_processed = process_datetime_and_sentiment(df_cop27)
    
    # Aggregate by group and date
    cop26_grouped = aggregate_by_group_and_date(df_cop26_processed)
    cop27_grouped = aggregate_by_group_and_date(df_cop27_processed)
    
    # Define numeric columns for aggregation
    numeric_columns = ['Positive', 'Negative', 'Neutral', 'negative_percent', 
                      'neutral_percent', 'positive_percent', 'Sentiment Index', 
                      'sum_twits', 'day_number']
    
    # Create group datasets
    cop26_group_datasets = create_group_datasets(cop26_grouped)
    cop27_group_datasets = create_group_datasets(cop27_grouped)
    
    # Create overall aggregated datasets
    cop26_overall = cop26_grouped.groupby('date')[numeric_columns].mean().reset_index()
    cop27_overall = cop27_grouped.groupby('date')[numeric_columns].mean().reset_index()
    
    # Align COP27 dates with COP26 for comparison
    cop27_overall['date'] = cop26_overall['date']
    
    return (cop26_group_datasets, cop27_group_datasets, 
            cop26_overall, cop27_overall, 
            cop26_grouped, cop27_grouped,
            df_cop26_processed, df_cop27_processed)

# Define events for plotting
EVENTS_COP26 = {
    '2021-10-31': 'COP26 start',
    '2021-11-06': 'Demonstrations', 
    '2021-11-13': 'COP26 ends'
}

EVENTS_COP27 = {
    '2022-11-06': 'COP27 start',
    '2022-11-12': 'Demonstrations',
    '2022-11-20': 'COP27 ends'
}

LEGEND_NAMES = [
    'NGOs that came with states delegations',
    'NGOs sent their own delegation', 
    'NGOs that came under other NGOs delegations',
    "NGOs that didn't sent NGO Representatives"
]