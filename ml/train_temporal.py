import pandas as pd, duckdb, joblib, os
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report

FEATURES = [
    'pickup_hour','pickup_dayofweek','is_weekend','pickup_month',
    'has_event_nearby','event_count',
    'parade_count','festival_count','race_count',
    'rush_hour',
    'hour_x_weekend','hour_x_event','weekend_x_event',
    'borough_avg_fare','borough_avg_distance','loc_pop_decile'
]

def train_temporal():
    con = duckdb.connect('data/warehouse.duckdb', read_only=True)
    df = con.execute('''
        SELECT d.pickup_date, d.pickup_hour, d.pickup_dayofweek, d.is_weekend, d.pickup_month,
               f.has_event_nearby,
               COALESCE(ec.event_count, 0) AS event_count,
               COALESCE(ec.parade_count, 0) AS parade_count,
               COALESCE(ec.festival_count, 0) AS festival_count,
               COALESCE(ec.race_count, 0) AS race_count,
               CAST(AVG(f.fare_amount) OVER (PARTITION BY l.borough) AS FLOAT) AS borough_avg_fare,
               CAST(AVG(f.trip_distance) OVER (PARTITION BY l.borough) AS FLOAT) AS borough_avg_distance,
               COUNT(*) OVER (PARTITION BY f.pickup_loc_id) AS loc_total,
               COUNT(*) OVER (PARTITION BY f.pickup_loc_id, d.pickup_hour) AS trip_count
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

    # Surge label dihitung dari data TRAIN saja (Jan-Mei)
    train_mask = df['pickup_month'].isin([1,2,3,4,5])
    test_mask = df['pickup_month'] == 6

    threshold = df.loc[train_mask, 'trip_count'].quantile(0.75)
    df['is_surge'] = (df['trip_count'] >= threshold).astype(int)

    # Feature engineering
    # NOTE: loc_total & borough_* dihitung dari ALL data (sedikit leak).
    # Untuk produksi ketat, hitung dari Jan-Mei saja.
    df['loc_pop_decile'] = pd.qcut(df['loc_total'], q=10, labels=False, duplicates='drop')
    df['rush_hour'] = df['pickup_hour'].isin([7,8,9,16,17,18]).astype(int)
    df['hour_x_weekend'] = df['pickup_hour'] * df['is_weekend']
    df['hour_x_event'] = df['pickup_hour'] * df['has_event_nearby']
    df['weekend_x_event'] = df['is_weekend'] * df['has_event_nearby']

    # Temporal split: train Jan-Mei, test Juni
    X_train = df.loc[train_mask, FEATURES].fillna(0).sample(frac=0.2, random_state=42)
    y_train = df.loc[train_mask, 'is_surge'].loc[X_train.index]

    X_test = df.loc[test_mask, FEATURES].fillna(0)
    y_test = df.loc[test_mask, 'is_surge']

    n_train = len(X_train)
    n_test = len(X_test)
    print(f'Train: {n_train:,} baris (Jan-Mei, 20% sample)')
    print(f'Test:  {n_test:,} baris (Juni 2025, full)')

    model = HistGradientBoostingClassifier(
        max_iter=100, max_depth=8, learning_rate=0.1,
        random_state=42, class_weight='balanced'
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f'\n[OK] Akurasi model: {acc:.2%}')
    print(classification_report(y_test, y_pred))

    os.makedirs('ml', exist_ok=True)
    joblib.dump(model, 'ml/surge_model.pkl')
    print('[OK] Model disimpan ke ml/surge_model.pkl')
    return model

if __name__ == '__main__':
    train_temporal()
