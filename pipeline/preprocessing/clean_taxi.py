import pandas as pd, glob, os

def clean_taxi():
    files = glob.glob('data/raw/taxi/*.parquet')
    dfs = []
    for f in files:
        df = pd.read_parquet(f)
        df = df.dropna(subset=['tpep_pickup_datetime','tpep_dropoff_datetime','PULocationID','DOLocationID'])
        df = df[(df['fare_amount'] > 0) & (df['total_amount'] > 0)]
        df = df[(df['trip_distance'] > 0) & (df['trip_distance'] < 200)]
        df = df[df['passenger_count'] > 0]
        df['pickup_dt'] = pd.to_datetime(df['tpep_pickup_datetime'])
        df['dropoff_dt'] = pd.to_datetime(df['tpep_dropoff_datetime'])
        df['trip_duration_min'] = (df['dropoff_dt'] - df['pickup_dt']).dt.total_seconds() / 60
        df = df[(df['trip_duration_min'] > 1) & (df['trip_duration_min'] < 300)]
        # Filter hanya Jan-Jun 2025
        df = df[(df['pickup_dt'] >= '2025-01-01') & (df['pickup_dt'] < '2025-07-01')]
        # Kolom turunan
        df['pickup_date']      = df['pickup_dt'].dt.date
        df['pickup_hour']      = df['pickup_dt'].dt.hour
        df['pickup_dayofweek'] = df['pickup_dt'].dt.dayofweek
        df['pickup_month']     = df['pickup_dt'].dt.month
        df['is_weekend']       = (df['pickup_dayofweek'] >= 5).astype(int)
        dfs.append(df)
    result = pd.concat(dfs, ignore_index=True)
    os.makedirs('data/intermediate', exist_ok=True)
    result.to_parquet('data/intermediate/clean_taxi.parquet', index=False)
    print(f'[OK] Data bersih: {len(result):,} baris')
    return result

if __name__ == '__main__':
    clean_taxi()
