import pandas as pd, os

def clean_events():
    df = pd.read_csv('data/raw/events/events_2025.csv')
    df['event_date'] = pd.to_datetime(df['start_date_time'], errors='coerce').dt.date
    df['hour'] = pd.to_datetime(df['start_date_time'], errors='coerce').dt.hour
    df = df.dropna(subset=['event_date'])
    df['event_type'] = df.get('event_type', pd.Series('Other', index=df.index)).fillna('Other')
    # All events ingested already impact streets, so all are "major"
    df['is_major_event'] = 1
    df.to_csv('data/intermediate/clean_events.csv', index=False)
    print(f'[OK] Events bersih: {len(df):,} event')
    return df

if __name__ == '__main__':
    clean_events()