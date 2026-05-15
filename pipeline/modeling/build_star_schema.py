import pandas as pd, duckdb, os

def build_star_schema(clean_taxi_path='data/intermediate/clean_taxi.parquet',
                      clean_events_path='data/intermediate/clean_events.csv'):
    taxi = pd.read_parquet(clean_taxi_path)
    events = pd.read_csv(clean_events_path)

    # ── dim_datetime ──
    dt_df = taxi[['pickup_dt','pickup_date','pickup_hour','pickup_dayofweek',
                  'pickup_month','is_weekend']].drop_duplicates().copy()
    dt_df.reset_index(drop=True, inplace=True)
    dt_df['datetime_id'] = dt_df.index + 1
    dt_df.to_parquet('data/intermediate/dim_datetime.parquet', index=False)

    # ── dim_location (dari taxi zone lookup) ──
    loc_url = 'https://d37ci6vzurychx.cloudfront.net/misc/taxi+_zone_lookup.csv'
    try:
        loc_df = pd.read_csv(loc_url)
    except:
        loc_df = pd.DataFrame({'LocationID': taxi['PULocationID'].unique(),
                              'Borough':'Unknown','Zone':'Unknown','service_zone':'Unknown'})
    loc_df.columns = [c.lower() for c in loc_df.columns]
    loc_df.rename(columns={'locationid':'location_id'}, inplace=True)
    loc_df.to_parquet('data/intermediate/dim_location.parquet', index=False)

    # ── dim_event ──
    events['event_id'] = range(1, len(events)+1)
    events.to_parquet('data/intermediate/dim_event.parquet', index=False)

    # ── Gabungkan taxi dengan datetime & event ──
    taxi = taxi.merge(dt_df[['pickup_dt','datetime_id']], on='pickup_dt', how='left')
    event_dates = set(events['event_date'].astype(str))
    taxi['has_event_nearby'] = taxi['pickup_date'].astype(str).isin(event_dates).astype(int)
    taxi['trip_id'] = range(1, len(taxi)+1)

    fact = taxi[['trip_id','datetime_id','PULocationID','DOLocationID',
                 'fare_amount','tip_amount','trip_distance','trip_duration_min',
                 'passenger_count','has_event_nearby']].copy()
    fact.rename(columns={'PULocationID':'pickup_loc_id','DOLocationID':'dropoff_loc_id'}, inplace=True)
    fact.to_parquet('data/intermediate/fact_trips.parquet', index=False)

    # ── Simpan ke DuckDB ──
    con = duckdb.connect('data/warehouse.duckdb')
    con.execute("CREATE OR REPLACE TABLE fact_trips AS SELECT * FROM fact")
    con.execute("CREATE OR REPLACE TABLE dim_datetime AS SELECT * FROM dt_df")
    con.execute("CREATE OR REPLACE TABLE dim_location AS SELECT * FROM loc_df")
    con.execute("CREATE OR REPLACE TABLE dim_event AS SELECT * FROM events")
    con.close()
    print('[OK] Star schema berhasil dibuat di DuckDB!')
    return fact

if __name__ == '__main__':
    build_star_schema()