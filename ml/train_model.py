import pandas as pd, duckdb, joblib, os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

FEATURES = ['pickup_hour','pickup_dayofweek','is_weekend','pickup_month',
            'pickup_loc_id','has_event_nearby','avg_fare','avg_distance']

def train_model():
    con = duckdb.connect('data/warehouse.duckdb', read_only=True)
    df = con.execute('''
        SELECT d.pickup_hour, d.pickup_dayofweek, d.is_weekend, d.pickup_month,
               f.pickup_loc_id, f.has_event_nearby,
               AVG(f.fare_amount) OVER (PARTITION BY f.pickup_loc_id) AS avg_fare,
               AVG(f.trip_distance) OVER (PARTITION BY f.pickup_loc_id) AS avg_distance,
               COUNT(*) OVER (PARTITION BY f.pickup_loc_id, d.pickup_hour) AS trip_count
        FROM fact_trips f JOIN dim_datetime d ON f.datetime_id = d.datetime_id
    ''').df()
    con.close()

    # Definisi surge: trip count >= persentil 75
    threshold = df['trip_count'].quantile(0.75)
    df['is_surge'] = (df['trip_count'] >= threshold).astype(int)

    X = df[FEATURES].fillna(0)
    y = df['is_surge']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f'[OK] Akurasi model: {acc:.2%}')
    print(classification_report(y_test, y_pred))

    os.makedirs('ml', exist_ok=True)
    joblib.dump(model, 'ml/surge_model.pkl')
    print('[OK] Model disimpan ke ml/surge_model.pkl')
    return model

if __name__ == '__main__':
    train_model()