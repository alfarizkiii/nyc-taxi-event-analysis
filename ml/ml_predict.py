import pandas as pd, joblib, duckdb, os

FEATURES = ['pickup_hour','pickup_dayofweek','is_weekend','pickup_month', 'pickup_loc_id','has_event_nearby','avg_fare','avg_distance']

def predict():
    model = joblib.load('ml/surge_model.pkl')
    con = duckdb.connect('data/warehouse.duckdb', read_only=True)
    df = con.execute('''
        SELECT DISTINCT d.pickup_hour, d.pickup_dayofweek, d.is_weekend, d.pickup_month,
               f.pickup_loc_id, f.has_event_nearby,
               AVG(f.fare_amount) OVER (PARTITION BY f.pickup_loc_id) AS avg_fare,
               AVG(f.trip_distance) OVER (PARTITION BY f.pickup_loc_id) AS avg_distance
        FROM fact_trips f JOIN dim_datetime d ON f.datetime_id = d.datetime_id
    ''').df()
    con.close()
    X = df[FEATURES].fillna(0)
    df['is_surge'] = model.predict(X)
    df['surge_probability'] = model.predict_proba(X)[:, 1]
    os.makedirs('data/final', exist_ok=True)
    df.to_parquet('data/final/ml_predictions.parquet', index=False)
    print('[OK] Prediksi disimpan ke data/final/ml_predictions.parquet')
    return df

if __name__ == '__main__':
    predict()
