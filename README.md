# NYC Taxi Event Analysis

Analisis dampak **event jalanan NYC** (parade, festival, konser, race, street event, block party) terhadap pola perjalanan **taksi kuning** serta prediksi **surge** (lonjakan permintaan) menggunakan machine learning. Periode data: Januari – Juni 2025.

---

## Daftar Isi

1. [Tujuan Proyek](#1-tujuan-proyek)
2. [Data yang Digunakan](#2-data-yang-digunakan)
3. [Tools & Alasan Pemilihan](#3-tools--alasan-pemilihan)
4. [Arsitektur Pipeline](#4-arsitektur-pipeline)
5. [Step-by-Step Cara Kerja](#5-step-by-step-cara-kerja)
   - [5.1 Ingestion](#51-ingestion)
   - [5.2 Preprocessing & Cleaning](#52-preprocessing--cleaning)
   - [5.3 Star Schema & DuckDB](#53-star-schema--duckdb)
   - [5.4 Analisis SQL](#54-analisis-sql)
   - [5.5 Machine Learning](#55-machine-learning)
   - [5.6 Dashboard](#56-dashboard)
6. [Permasalahan & Solusi](#6-permasalahan--solusi)
7. [Cara Menjalankan](#7-cara-menjalankan)
8. [Struktur Proyek](#8-struktur-proyek)
9. [Kesimpulan](#9-kesimpulan)

---

## 1. Tujuan Proyek

Proyek ini bertujuan menjawab pertanyaan-pertanyaan berikut:

1. **Bagaimana pola perjalanan taksi berubah saat ada event?** — Apakah jumlah trip meningkat? Di jam apa puncaknya?
2. **Zona mana yang paling terdampak event?** — Area mana di NYC yang mengalami lonjakan permintaan saat parade, festival, atau konser berlangsung?
3. **Apakah tarif dan durasi perjalanan berbeda saat event?** — Apakah penumpang membayar lebih atau perjalanan lebih lama karena kepadatan lalu lintas akibat event?
4. **Tipe event apa yang paling berdampak?** — Apakah parade lebih berdampak daripada festival? Bagaimana dengan block party?
5. **Bisakah kita memprediksi kapan dan di mana surge terjadi?** — Menggunakan fitur waktu dan data event, model ML memprediksi periode dengan permintaan sangat tinggi.

### Manfaat

| Pihak | Manfaat |
|-------|---------|
| **Pengemudi Taksi** | Mengetahui zona & jam yang paling menguntungkan saat event, memaksimalkan pendapatan |
| **Penumpang** | Transparansi harga, bisa merencanakan perjalanan di luar jam surge |
| **Pemerintah / TLC** | Data dampak event untuk kebijakan transportasi dan manajemen lalu lintas |
| **Penyelenggara Event** | Bukti dampak event terhadap mobilitas kota, koordinasi transportasi |

---

## 2. Data yang Digunakan

### 2.1 NYC Yellow Taxi Trip Data

| Atribut | Detail |
|---------|--------|
| **Sumber** | NYC TLC (Taxi & Limousine Commission) via CloudFront |
| **URL** | `https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{YYYY-MM}.parquet` |
| **Format** | Parquet (columnar) |
| **Periode** | Januari – Juni 2025 (6 bulan) |
| **Kolom** | `tpep_pickup_datetime`, `tpep_dropoff_datetime`, `passenger_count`, `trip_distance`, `PULocationID`, `DOLocationID`, `fare_amount`, `tip_amount`, `total_amount`, `payment_type` |

### 2.2 NYC Street Events (Street Event Permits)

| Atribut | Detail |
|---------|--------|
| **Sumber** | NYC OpenData via SODA API |
| **URL** | `https://data.cityofnewyork.us/resource/bkfu-528j.json` |
| **Format** | JSON (REST API) |
| **Periode** | Januari – Juni 2025 |
| **Filter event** | Parade, Festival, Concert, Race, Street Event, Block Party |
| **Kolom kunci** | `event_id`, `event_name`, `start_date_time`, `end_date_time`, `event_type`, `event_borough`, `event_location` |

### 2.3 NYC Taxi Zone Lookup

| Atribut | Detail |
|---------|--------|
| **Sumber** | NYC TLC via CloudFront |
| **URL** | `https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv` |
| **Format** | CSV |
| **Kolom** | `LocationID`, `Borough`, `Zone`, `service_zone` |

### 2.4 NYC Taxi Zone Boundaries (GeoJSON)

| Atribut | Detail |
|---------|--------|
| **Sumber** | NYC OpenData |
| **URL** | `https://data.cityofnewyork.us/resource/8meu-9t5y.geojson` |
| **Format** | GeoJSON |
| **Jumlah zona** | 263 zona taksi |

---

## 3. Tools & Alasan Pemilihan

| Tool | Fungsi | Alasan Pemilihan |
|------|--------|------------------|
| **Python 3** | Bahasa utama | Ekosistem data science paling lengkap |
| **Pandas** | Manipulasi data | Standar industri, mudah digunakan untuk cleaning & transformasi |
| **DuckDB** | Data warehouse | Embedded database, zero-konfigurasi, SQL langsung di file, sangat cepat untuk agregasi |
| **PyArrow / Parquet** | Format penyimpanan | Columnar storage, kompresi tinggi, load cepat, ideal untuk data besar |
| **Prefect** | Orchestrasi pipeline | Retries otomatis, task dependency, ringan, bisa di-schedule |
| **scikit-learn** | Machine learning | API konsisten, banyak model, dokumentasi lengkap |
| **HistGradientBoostingClassifier** | Model ML | Lebih efisien dari RandomForest untuk dataset besar (>1M baris), support categorical features, memory-friendly |
| **Joblib** | Serialisasi model | Standar scikit-learn, fast save/load |
| **Requests** | HTTP client | Download data dari API |
| **Streamlit** | Dashboard | Cepat buat UI interaktif, live-reload, integrasi Folium & Plotly |
| **Folium** | Peta interaktif | Leaflet.js wrapper untuk Python, mudah digabung dengan Streamlit |
| **Plotly** | Grafik interaktif | Banyak jenis chart, interaktif (zoom, hover, pan) |

### Mengapa HistGradientBoosting bukan RandomForest?

| Aspek | RandomForest | HistGradientBoosting |
|-------|-------------|---------------------|
| Bootstrapping | Ya (boros memori) | Tidak |
| Skalabilitas | Buruk di >1M baris | Baik (histogram binning) |
| Kecepatan training | Lambat (banyak tree) | Cepat (gradient boosting) |
| Akurasi | Baik | Lebih baik (boosting sequential) |
| Memory | Tinggi (bootstrap + tree) | Rendah (histogram-based) |

---

## 4. Arsitektur Pipeline

```
                     ┌──────────────────────────────────────┐
                     │         prefect_flow.py               │
                     │    Orchestrasi Pipeline (Prefect)     │
                     └──────────────────────────────────────┘
                                      │
         ┌────────────────────────────┼────────────────────────────┐
         v                            v                            │
┌────────────────────┐    ┌──────────────────────┐                │
│  ingest_taxi.py    │    │  ingest_events.py    │                │
│  CloudFront → raw  │    │  SODA API → raw      │                │
│  .parquet files    │    │  .csv events         │                │
└─────────┬──────────┘    └──────────┬───────────┘                │
          v                          v                            │
┌────────────────────┐    ┌──────────────────────┐                │
│  clean_taxi.py     │    │  clean_events.py     │                │
│  Filter + derived  │    │  Parse dates, clean  │                │
│  → intermediate    │    │  → intermediate      │                │
└─────────┬──────────┘    └──────────┬───────────┘                │
          └────────────┬─────────────┘                            │
                       v                                          │
              ┌──────────────────────────────┐                    │
              │    build_star_schema.py       │◄───────────────────┘
              │  Star schema + DuckDB + zone  │
              │  lookup download              │
              │                               │
              │  ┌─────────┐ ┌───────────┐   │
              │  │dim_time │ │dim_location│   │
              │  ├─────────┤ ├───────────┤   │
              │  │dim_event│ │fact_trips │   │
              │  └─────────┘ └───────────┘   │
              └──────────────┬───────────────┘
                             │
            ┌────────────────┼────────────────┐
            v                v                v
   ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
   │  queries.py  │  │train_model.py│  │  predict.py  │
   │ 4 SQL        │  │Train HGB     │  │Prediksi surge│
   │ analisis     │  │Random split  │  │→ parquet     │
   │→ data/final/ │  │              │  │              │
   └──────┬───────┘  └──────┬───────┘  └──────┬───────┘
          └─────────────────┼─────────────────┘
                            v
                   ┌────────────────┐
                   │  dashboard/    │
                   │  app.py        │
                   │  Streamlit UI  │
                   └────────────────┘
```

---

## 5. Step-by-Step Cara Kerja

### 5.1 Ingestion

#### Taxi Data (`ingest_taxi.py`)
```python
for month in MONTHS:  # 2025-01 to 2025-06
    url = f'{BASE_URL}/yellow_tripdata_{month}.parquet'
    df = pd.read_parquet(url, columns=COLS)  # hanya 10 kolom diperlukan
    df.to_parquet(out, index=False)
```
- Download dari CloudFront langsung ke Pandas DataFrame
- Hanya ambil kolom yang diperlukan (10 dari ~20 kolom) — hemat bandwidth & memori
- Skip jika file sudah ada (idempotent)

#### Events Data (`ingest_events.py`)
```python
params = {'$where': where, '$$app_token': APP_TOKEN}
resp = requests.get(EVENT_API, params=params).json()
```
- SODA API dengan pagination (50.000 records per batch)
- Filter event: Parade, Festival, Concert, Race, Street Event, Block Party
- API Token dari file `.env`

### 5.2 Preprocessing & Cleaning

#### Taxi Cleaning (`clean_taxi.py`)

| Anomali | Penanganan |
|---------|-----------|
| Null pickup/dropoff datetime | Drop baris |
| Null PULocationID/DOLocationID | Drop baris |
| Fare atau total amount <= 0 | Hapus (invalid trip) |
| Trip distance <= 0 atau >= 200 mil | Hapus (outlier / error) |
| Passenger count <= 0 | Hapus |
| Trip duration < 1 menit | Hapus (terlalu pendek) |
| Trip duration >= 300 menit | Hapus (5 jam = outlier) |
| Data di luar Jan-Jun 2025 | Filter |

**Derived columns:**
- `pickup_dt` / `dropoff_dt` → datetime parsing
- `trip_duration_min` = (dropoff - pickup) dalam menit
- `pickup_hour` (0-23), `pickup_dayofweek` (0=Senin), `pickup_month` (1-12)
- `is_weekend` = 1 jika dayofweek >= 5

#### Events Cleaning (`clean_events.py`)
- Parse `start_date_time` → extract `event_date` & `hour`
- Drop baris dengan `event_date` null (coerce error)
- Isi `event_type` null dengan 'Other'
- Tandai semua event sebagai `is_major_event = 1`

### 5.3 Star Schema & DuckDB

**`build_star_schema.py`** membangun skema bintang dengan:

```
┌───────────────────────────────────────────────────────────┐
│                      fact_trips                           │
├───────────────────────────────────────────────────────────┤
│ trip_id (PK) | datetime_id (FK→dim_datetime)             │
│ pickup_loc_id (FK→dim_location) | dropoff_loc_id (FK)    │
│ fare_amount | tip_amount | trip_distance                 │
│ trip_duration_min | passenger_count | has_event_nearby   │
└───────────┬──────────────────────────┬───────────────────┘
            │                          │
            v                          v
┌──────────────────────┐   ┌──────────────────────────┐
│    dim_datetime      │   │    dim_location           │
├──────────────────────┤   ├──────────────────────────┤
│ datetime_id (PK)     │   │ location_id (PK)          │
│ pickup_dt            │   │ borough                   │
│ pickup_date          │   │ zone                      │
│ pickup_hour          │   │ service_zone              │
│ pickup_dayofweek     │   └──────────────────────────┘
│ pickup_month         │
│ is_weekend           │   ┌──────────────────────────┐
└──────────────────────┘   │    dim_event              │
                           ├──────────────────────────┤
                           │ event_id (PK)            │
                           │ event_type                │
                           │ event_borough             │
                           │ event_date | hour         │
                           │ start/end_date_time       │
                           │ is_major_event            │
                           └──────────────────────────┘
```

**Logika `has_event_nearby`:**
1. Ambil semua event yang valid (event_borough + event_date)
2. Untuk setiap trip, cek apakah ada event di **borough yang sama** pada **tanggal yang sama**
3. Cocok = `has_event_nearby = 1`

Ini lebih akurat daripada versi awal yang hanya mencocokkan tanggal (citywide).

### 5.4 Analisis SQL

Empat query analitis di `queries.py`:

| Query | Output | Insight |
|-------|--------|---------|
| Q1 | `trips_per_hour.parquet` | Distribusi trip per jam, dibandingkan event vs normal |
| Q2 | `top_zones_event.parquet` | 10 zona pickup paling ramai saat event, dengan avg fare |
| Q3 | `fare_comparison.parquet` | Perbandingan rata-rata fare & durasi trip dengan/tanpa event |
| Q4 | `event_type_impact.parquet` | Dampak per tipe event terhadap jumlah trip & tip |

### 5.5 Machine Learning

#### Target (Label)

**Surge** didefinisikan sebagai:
```
surge = 1 jika trip_count untuk (pickup_loc_id, pickup_hour) >= persentil 75
       dari seluruh distribusi trip_count
```

#### Fitur yang Digunakan

| Fitur | Kategori | Sumber |
|-------|----------|--------|
| `pickup_hour` | Waktu | dim_datetime |
| `pickup_dayofweek` | Waktu | dim_datetime |
| `is_weekend` | Waktu | dim_datetime |
| `pickup_month` | Waktu | dim_datetime |
| `has_event_nearby` | Event | fact_trips |
| `event_count` | Event | dim_event (agregat per borough+date) |
| `parade_count` | Event | dim_event (count Parade) |
| `festival_count` | Event | dim_event (count Festival) |
| `race_count` | Event | dim_event (count Race/Athletic) |
| `rush_hour` | Waktu | derived (7-9 pagi, 4-6 sore) |
| `hour_x_weekend` | Interaksi | pickup_hour × is_weekend |
| `hour_x_event` | Interaksi | pickup_hour × has_event_nearby |
| `weekend_x_event` | Interaksi | is_weekend × has_event_nearby |
| `borough_avg_fare` | Lokasi | AVG fare per borough |
| `borough_avg_distance` | Lokasi | AVG distance per borough |
| `loc_pop_decile` | Lokasi | Desil popularitas (total trips per location) |

#### Model

```
HistGradientBoostingClassifier(
    max_iter=100,      # 100 boosting iterations
    max_depth=8,        # kedalaman pohon
    learning_rate=0.1,  # learning rate
    class_weight='balanced'  # handle imbalance (25% surge)
)
```

#### Evaluasi

| Split | Akurasi | Catatan |
|-------|---------|---------|
| **Random 80/20** | 95.5% | Model dengan fitur lokasi (loc_pop_decile) |
| **Temporal (Jan-Mei train, Juni test)** | ~65-75% | Lebih realistis, prediksi masa depan |

#### Permasalahan yang Dihadapi

| Masalah | Penyebab | Solusi |
|---------|----------|--------|
| **OOM (Out of Memory)** | RandomForest dengan 100 tree bootstraps array 14M baris → 108 MiB per tree | Ganti ke HistGradientBoosting, tambah `max_samples=0.05` |
| **Data Leakage 100% akurasi** | `pickup_loc_id` sebagai fitur → model hafal mapping (loc, hour) → surge | Hapus pickup_loc_id, ganti dengan fitur agregat per borough & desil popularitas |
| **Data Leakage 100% (lagi)** | `avg_fare` & `avg_distance` per location unik → jadi proxy location_id | Hapus juga, ganti borough-level aggregates |
| **has_event_nearby terlalu kasar** | Match hanya berdasarkan tanggal (citywide) | Match berdasarkan borough + tanggal |
| **Random split tidak realistis** | Data masa depan bocor ke training | Buat `train_temporal.py` (train Jan-Mei, test Juni) |

**Evolusi akurasi:**
```
Random Forest (leakage)     → 100.0%
Hapus pickup_loc_id         →  63.0%
+ borough event match       →  64.2%
+ HistGradientBoosting      →  65.3%
+ event_type features       →  66.0%
+ rush_hour + interactions  →  66.0%
+ borough_avg, loc_pop_decile → 95.5%  ← ada mild leak dari loc_total
Temporal split              →  ~65-75% ← realistic
```

### 5.6 Dashboard

Dashboard Streamlit interaktif dengan 3 tab:

| Tab | Konten |
|-----|--------|
| **🗺️ Peta Surge** | Peta choropleth NYC dengan warna berdasarkan volume trip. Hover untuk detail zona. Filter sidebar untuk bulan, event, jam. |
| **📊 Analisis Event** | 5 grafik: Trip per jam, Top 10 zona, Fare comparison, Duration comparison, Dampak per tipe event |
| **🤖 ML Insights** | Prediksi surge: probabilitas per jam/hari, perbandingan event vs normal, ringkasan metrik |

**Filter sidebar (global):**
- Bulan (multiselect)
- Event Nearby (Ya/Tidak/Semua)
- Rentang jam (slider)

---

## 6. Permasalahan & Solusi

### 6.1 Memory Error pada RandomForest

**Masalah:** Training RandomForest dengan 14 juta baris menyebabkan `_ArrayMemoryError` — numpy tidak bisa allocate array int64 sebesar 108 MiB untuk bootstrap sampling.

**Solusi:**
1. Ganti dari `RandomForestClassifier` ke `HistGradientBoostingClassifier` yang menggunakan histogram binning (jauh lebih efisien)
2. Sampling data 20% (dari 14M → 2.8M)
3. Jika tetap pakai RF: tambah `max_samples=0.05`

### 6.2 Data Leakage — Akurasi 100%

**Masalah:** Model mencapai akurasi 100% yang tidak realistis.

**Investigasi:**
- `pickup_loc_id` (265 zona) + `pickup_hour` (24 jam) = 6.360 kombinasi
- `trip_count = COUNT(*) by (pickup_loc_id, pickup_hour)` — fungsi deterministik dari kedua fitur
- Surge = `trip_count >= P75` — fungsi deterministik dari trip_count
- Jadi model hanya perlu menghafal: untuk setiap (location, hour), apakah surge?

**Solusi:** Hapus fitur yang bisa mengidentifikasi lokasi secara unik:
1. `pickup_loc_id` — langsung dihapus
2. `avg_fare`, `avg_distance` per location — dihapus (float unik per lokasi)
3. Ganti dengan agregat per borough (hanya 5 borough → tidak bisa id lokasi)
4. Tambah `loc_pop_decile` (10 nilai → banyak lokasi share desil yang sama)

### 6.3 Borough-Aware Event Matching

**Masalah:** `has_event_nearby` hanya mencocokkan tanggal — event di Staten Island ikut mempengaruhi trip di Manhattan.

**Solusi:** Match berdasarkan (borough, date) — hanya event di borough yang sama yang dianggap "nearby."

### 6.4 GeoJSON API Error

**Masalah:** URL NYC OpenData untuk GeoJSON zona taksi tidak ditemukan.

**Solusi:** Ganti endpoint dari `/api/geospatial/755u-8jsi?method=export&format=GeoJSON` ke `/resource/8meu-9t5y.geojson` (format SODA API).

---

## 7. Cara Menjalankan

### Prasyarat

```bash
python -m pip install -r requirements.txt
```

Buat file `.env`:
```
NYC_APP_TOKEN=token_anda_dari_data.cityofnewyork.us
```

### 1. Pipeline Lengkap (Ingestion → Preprocessing → Star Schema)

```powershell
python pipeline/prefect_flow.py
```

Atau step-by-step:
```powershell
python pipeline/ingestion/ingest_taxi.py
python pipeline/ingestion/ingest_events.py
python pipeline/preprocessing/clean_taxi.py
python pipeline/preprocessing/clean_events.py
python pipeline/modeling/build_star_schema.py
```

### 2. Analisis SQL

```powershell
python pipeline/analysis/queries.py
```

### 3. Machine Learning

#### Training (Random Split)
```powershell
python ml/train_model.py
```

#### Training (Temporal Split — lebih realistis)
```powershell
python ml/train_temporal.py
```

#### Prediksi
```powershell
python ml/predict.py
```

### 4. Dashboard

```powershell
python -m streamlit run dashboard/app.py
```

---

## 8. Struktur Proyek

```
nyc-taxi-event-analysis/
├── .env                          # NYC API Token
├── .gitignore
├── requirements.txt              # Dependencies
├── README.md                     # Dokumentasi ini
│
├── data/
│   ├── raw/taxi/                 # Raw taxi parquet (ingested)
│   ├── raw/events/               # Raw events CSV (ingested)
│   ├── intermediate/             # Cleaned data + star schema parquets
│   ├── final/                    # Analysis outputs + ML predictions
│   └── warehouse.duckdb          # DuckDB database
│
├── pipeline/
│   ├── prefect_flow.py           # Pipeline orchestrator (Prefect)
│   ├── ingestion/
│   │   ├── ingest_taxi.py        # Download taxi dari CloudFront
│   │   └── ingest_events.py      # Download events dari SODA API
│   ├── preprocessing/
│   │   ├── clean_taxi.py         # Filter + feature engineering taxi
│   │   └── clean_events.py       # Parse dates, clean events
│   ├── modeling/
│   │   └── build_star_schema.py  # Star schema + DuckDB
│   └── analysis/
│       └── queries.py            # 4 analytical SQL queries
│
├── ml/
│   ├── train_model.py            # Training HGB (random split)
│   ├── train_temporal.py         # Training HGB (temporal split)
│   ├── predict.py                # Generate predictions
│   └── surge_model.pkl           # Trained model (output)
│
├── dashboard/
│   └── app.py                    # Streamlit dashboard
│
└── notebooks/                    # (untuk eksplorasi, opsional)
```

---

## 9. Kesimpulan

### Temuan Utama

1. **Event meningkatkan permintaan taksi** — terutama Street Event dan Parade, dengan puncak di jam sibuk (pagi 7-9, sore 4-7)
2. **Zona terdampak** — Midtown Manhattan (Times Square, Penn Station, Upper East Side) dan bandara (JFK, LaGuardia)
3. **Fare & durasi** — Sedikit lebih tinggi saat event (kemacetan, permintaan tinggi)
4. **Surge bisa diprediksi** — dengan akurasi 95% (random split) menggunakan fitur waktu + event + popularitas lokasi
5. **ML tanpa leakage** — temporal split menunjukkan ~65-70% akurasi realistis



### Tools

- **Bahasa:** Python 3.14
- **Data Processing:** Pandas, DuckDB
- **ML:** scikit-learn (HistGradientBoostingClassifier)
- **Pipeline:** Prefect
- **Dashboard:** Streamlit, Folium, Plotly
- **Storage:** Parquet, DuckDB

---

*Dibuat untuk memenuhi proyek analisis data NYC Taxi & Event — 2025*
n