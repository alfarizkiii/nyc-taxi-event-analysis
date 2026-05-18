import duckdb
import pandas as pd
import os

def run_queries(db_path='data/warehouse.duckdb'):
    # 1. Proteksi Pipeline: Pastikan database warehouse sudah ada
    if not os.path.exists(db_path):
        print(f"[ERROR] Database {db_path} tidak ditemukan. Jalankan tahap Build Star Schema terlebih dahulu.")
        return None

    print("Terhubung ke DuckDB Warehouse...")
    con = duckdb.connect(db_path, read_only=True)
    os.makedirs('data/final', exist_ok=True)

    print("Mengeksekusi Query 1: Trip per jam...")
    q1 = con.execute('''
        SELECT d.pickup_hour AS hour,
               COUNT(CASE WHEN f.has_event_nearby = 1 THEN 1 END) AS trips_event,
               COUNT(CASE WHEN f.has_event_nearby = 0 THEN 1 END) AS trips_normal
        FROM fact_trips f
        JOIN dim_datetime d ON f.datetime_id = d.datetime_id
        GROUP BY d.pickup_hour 
        ORDER BY d.pickup_hour
    ''').df()
    q1.to_parquet('data/final/trips_per_hour.parquet', index=False, engine='pyarrow')

    print("Mengeksekusi Query 2: Top 10 zona pickup saat event...")
    q2 = con.execute('''
        SELECT l.zone, l.borough, COUNT(*) AS total_trips,
               AVG(f.fare_amount) AS avg_fare
        FROM fact_trips f
        JOIN dim_location l ON f.pickup_loc_id = l.location_id
        WHERE f.has_event_nearby = 1
        GROUP BY l.zone, l.borough
        ORDER BY total_trips DESC
        LIMIT 10
    ''').df()
    q2.to_parquet('data/final/top_zones_event.parquet', index=False, engine='pyarrow')

    print("Mengeksekusi Query 3: Perbandingan Rata-rata Fare & Durasi...")
    q3 = con.execute('''
        SELECT f.has_event_nearby,
               ROUND(AVG(f.fare_amount), 2) AS avg_fare,
               ROUND(AVG(f.trip_duration_min), 2) AS avg_duration
        FROM fact_trips f
        GROUP BY f.has_event_nearby
    ''').df()
    q3.to_parquet('data/final/fare_comparison.parquet', index=False, engine='pyarrow')

    print("Mengeksekusi Query 4: Dampak per tipe event...")
    q4 = con.execute('''
        SELECT e.event_type,
               COUNT(DISTINCT f.trip_id) AS total_trips,
               ROUND(AVG(f.tip_amount), 2) AS avg_tip
        FROM fact_trips f
        JOIN dim_datetime d ON f.datetime_id = d.datetime_id
        JOIN dim_location l ON f.pickup_loc_id = l.location_id
        JOIN dim_event e ON CAST(d.pickup_date AS VARCHAR) = CAST(e.event_date AS VARCHAR)
                        AND l.borough = e.event_borough
        WHERE f.has_event_nearby = 1
        GROUP BY e.event_type
        ORDER BY total_trips DESC
    ''').df()
    q4.to_parquet('data/final/event_type_impact.parquet', index=False, engine='pyarrow')

    print("Mengeksekusi Query 5: Tren Harian (Time Series)...")
    q5 = con.execute('''
        SELECT d.pickup_date,
               COUNT(f.trip_id) AS total_trips,
               SUM(f.has_event_nearby) AS event_trips
        FROM fact_trips f
        JOIN dim_datetime d ON f.datetime_id = d.datetime_id
        GROUP BY d.pickup_date
        ORDER BY d.pickup_date
    ''').df()
    q5.to_parquet('data/final/daily_trend.parquet', index=False, engine='pyarrow')

    print("Mengeksekusi Query 6: Heatmap Jam vs Hari...")
    q6 = con.execute('''
        SELECT d.pickup_dayofweek, d.pickup_hour,
               COUNT(f.trip_id) AS total_trips
        FROM fact_trips f
        JOIN dim_datetime d ON f.datetime_id = d.datetime_id
        GROUP BY d.pickup_dayofweek, d.pickup_hour
    ''').df()
    q6.to_parquet('data/final/heatmap_data.parquet', index=False, engine='pyarrow')

    print("Mengeksekusi Query 7: Sampel Boxplot Fare & Tip...")
    q7 = con.execute('''
        SELECT has_event_nearby, fare_amount, tip_amount
        FROM fact_trips
        USING SAMPLE 5%
    ''').df()
    q7.to_parquet('data/final/boxplot_sample.parquet', index=False, engine='pyarrow')

    con.close()
    print('[OK] Semua query selesai dan disimpan ke dalam folder data/final/')

    con.close()
    print('[OK] Semua query selesai dan disimpan ke dalam folder data/final/')
    
    return q1, q2, q3, q4, q5, q6, q7

if __name__ == '__main__':
    run_queries()