import pandas as pd
import os

BASE_URL = 'https://d37ci6vzurychx.cloudfront.net/trip-data'
MONTHS = ['2025-01','2025-02','2025-03','2025-04','2025-05','2025-06']
COLS = ['tpep_pickup_datetime','tpep_dropoff_datetime','passenger_count',
        'trip_distance','PULocationID','DOLocationID','fare_amount',
        'tip_amount','total_amount','payment_type']

def ingest_taxi():
    os.makedirs('data/raw/taxi', exist_ok=True)
    for month in MONTHS:
        url = f'{BASE_URL}/yellow_tripdata_{month}.parquet'
        out  = f'data/raw/taxi/yellow_{month}.parquet'
        if os.path.exists(out):
            print(f'[SKIP] {out} sudah ada')
            continue
        print(f'Mengunduh {month}...')
        df = pd.read_parquet(url, columns=COLS)
        df.to_parquet(out, index=False)
        print(f'[OK] Disimpan: {out}')

if __name__ == '__main__':
    ingest_taxi()
