import pandas as pd, joblib, duckdb, os

FEATURES = [
    'pickup_hour','pickup_dayofweek','is_weekend','pickup_month',
    'has_event_nearby','event_count',
    'parade_count','festival_count','race_count',
    'rush_hour',
    'hour_x_weekend','hour_x_event','weekend_x_event',
    'borough_avg_fare','borough_avg_distance','loc_pop_decile'
]

def predict():
    model = joblib.load('ml/surge_model.pkl')
    con = duckdb.connect('data/warehouse.duckdb', read_only=True)
    df = con.execute('''
        SELECT DISTINCT d.pickup_hour, d.pickup_dayofweek, d.is_weekend, d.pickup_month,
               f.has_event_nearby,
               COALESCE(ec.event_count, 0) AS event_count,
               COALESCE(ec.parade_count, 0) AS parade_count,
               COALESCE(ec.festival_count, 0) AS festival_count,
               COALESCE(ec.race_count, 0) AS race_count,
               CAST(AVG(f.fare_amount) OVER (PARTITION BY l.borough) AS FLOAT) AS borough_avg_fare,
               CAST(AVG(f.trip_distance) OVER (PARTITION BY l.borough) AS FLOAT) AS borough_avg_distance,
               COUNT(*) OVER (PARTITION BY f.pickup_loc_id) AS loc_total
        FROM fact_trips f
        JOIN dim_datetime d ON f.datetime_id = d.datetime_id
        LEFT JOIN dim_location l ON f.pickup_loc_id = l.location_id
        LEFT JOIN (
            SELECT event_date, event_borough,
                   COUNT(*) AS event_count,
                   SUM(CASE WHEN event_type LIKE '%Parade%' THEN 1 ELSE 0 END) AS parade_count,
                   SUM(CASE WHEN event_type LIKE '%Festival%' THEN 1 ELSE 0 END) AS festival_count,
                   SUM(CASE WHEN event_type LIKE '%Race%' OR event_type LIKE '%Athletic%' THEN 1 ELSE 0 END) AS race_count
            FROM dim_event
            GROUP BY event_date, event_borough
        ) ec ON ec.event_date = d.pickup_date AND ec.event_borough = l.borough
    ''').df()
    con.close()

    df['loc_pop_decile'] = pd.qcut(df['loc_total'], q=10, labels=False, duplicates='drop')
    df['rush_hour'] = df['pickup_hour'].isin([7,8,9,16,17,18]).astype(int)
    df['hour_x_weekend'] = df['pickup_hour'] * df['is_weekend']
    df['hour_x_event'] = df['pickup_hour'] * df['has_event_nearby']
    df['weekend_x_event'] = df['is_weekend'] * df['has_event_nearby']

    X = df[FEATURES].fillna(0)
    df['is_surge'] = model.predict(X)
    df['surge_probability'] = model.predict_proba(X)[:, 1]
    os.makedirs('data/final', exist_ok=True)
    df.to_parquet('data/final/ml_predictions.parquet', index=False)
    print('[OK] Prediksi disimpan ke data/final/ml_predictions.parquet')
    return df

if __name__ == '__main__':
    predict()
