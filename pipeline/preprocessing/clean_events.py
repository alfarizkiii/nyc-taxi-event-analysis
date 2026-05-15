import pandas as pd, os

MAJOR_TYPES = ['Concert','Festival','Sports','Parade','Fair','Exhibition']

def clean_events():
    df = pd.read_csv('data/raw/events/events_2025.csv')
    df['event_date'] = pd.to_datetime(df['start_date_time'], errors='coerce').dt.date
    df['hour'] = pd.to_datetime(df['start_date_time'], errors='coerce').dt.hour
    df = df.dropna(subset=['event_date'])
    df['event_type'] = df.get('event_type', pd.Series('Other', index=df.index)).fillna('Other')
    df['is_major_event'] = df['event_type'].apply(
        lambda x: 1 if any(t.lower() in str(x).lower() for t in MAJOR_TYPES) else 0)
    df.to_csv('data/intermediate/clean_events.csv', index=False)
    print(f'[OK] Events bersih: {len(df):,} event')
    return df

if __name__ == '__main__':
    clean_events()