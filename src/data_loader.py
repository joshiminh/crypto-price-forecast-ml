import pandas as pd

def load_data(path="data/crypto_statistics_data.csv"):
    """
    Load the main statistics dataset.
    """
    df = pd.read_csv(path, parse_dates=["date"]) 
    return df

def load_raw_data(hourly_path='data/Crypto_Hourly_Refined.csv', stats_path='data/Crypto_Stats_Refined.csv'):
    """
    Load raw hourly and stats data if available.
    """
    try:
        df_hourly = pd.read_csv(hourly_path)
        df_stats = pd.read_csv(stats_path)
        
        # Convert to datetime
        df_hourly['date'] = pd.to_datetime(df_hourly['date'])
        df_stats['date'] = pd.to_datetime(df_stats['date'])
        
        # Sort by date
        df_hourly = df_hourly.sort_values('date').reset_index(drop=True)
        df_stats = df_stats.sort_values('date').reset_index(drop=True)
        
        # Merge daily stats into hourly data
        df_hourly['date_day'] = df_hourly['date'].dt.floor('D')
        df_stats['date_day'] = df_stats['date']
        
        df = df_hourly.merge(
            df_stats[['date_day', 'average', 'range', 'pct_change']],
            on='date_day',
            how='left',
            suffixes=('', '_daily')
        )
        df.drop(columns=['date_day'], inplace=True)
        return df
    except FileNotFoundError as e:
        print(f"File not found: {e}")
        return pd.DataFrame()
