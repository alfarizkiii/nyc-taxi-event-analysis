import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import folium
from streamlit_folium import st_folium
import duckdb
import os

st.set_page_config(page_title='NYC Taxi & Event Analysis', layout='wide')
st.title('NYC Taxi & Event Analysis')
st.markdown('Analisis dampak event jalanan NYC terhadap pola perjalanan taksi kuning (Jan–Jun 2025)')

DATA_DIR = 'data/final'
DB_PATH = 'data/warehouse.duckdb'

# ── Load data ──
@st.cache_data
def load_parquet(name):
    p = os.path.join(DATA_DIR, name)
    return pd.read_parquet(p) if os.path.exists(p) else pd.DataFrame()

@st.cache_data
def load_zone_data():
    con = duckdb.connect(DB_PATH, read_only=True)
    df = con.execute('''
        SELECT l.location_id, l.zone, l.borough, l.service_zone,
               COUNT(*) AS total_trips,
               ROUND(AVG(f.fare_amount), 2) AS avg_fare,
               ROUND(AVG(f.trip_duration_min), 2) AS avg_duration,
               ROUND(100.0 * SUM(f.has_event_nearby) / COUNT(*), 1) AS event_trip_pct
        FROM fact_trips f
        JOIN dim_location l ON f.pickup_loc_id = l.location_id
        GROUP BY l.location_id, l.zone, l.borough, l.service_zone
    ''').df()
    con.close()
    return df

@st.cache_data
def get_geo_json():
    import requests, json
    url = 'https://data.cityofnewyork.us/resource/8meu-9t5y.geojson'
    try:
        r = requests.get(url, timeout=15)
        return r.json()
    except:
        return None

trips_hour = load_parquet('trips_per_hour.parquet')
top_zones = load_parquet('top_zones_event.parquet')
fare_comp = load_parquet('fare_comparison.parquet')
event_type = load_parquet('event_type_impact.parquet')
ml_pred = load_parquet('ml_predictions.parquet')
zones = load_zone_data()
geo = get_geo_json()

# ── Sidebar filters ──
st.sidebar.header('Filter')
months = list(range(1, 7))
month_labels = {1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'Mei',6:'Jun'}
sel_months = st.sidebar.multiselect('Bulan', months, default=months, format_func=lambda x: month_labels[x])
has_event = st.sidebar.radio('Event Nearby', ['Semua', 'Ya (has_event=1)', 'Tidak (has_event=0)'])

min_hour, max_hour = st.sidebar.slider('Jam', 0, 23, (0, 23))

if not ml_pred.empty:
    ml_f = ml_pred.copy()
    if sel_months:
        ml_f = ml_f[ml_f['pickup_month'].isin(sel_months)]
    if has_event == 'Ya (has_event=1)':
        ml_f = ml_f[ml_f['has_event_nearby'] == 1]
    elif has_event == 'Tidak (has_event=0)':
        ml_f = ml_f[ml_f['has_event_nearby'] == 0]
    ml_f = ml_f[(ml_f['pickup_hour'] >= min_hour) & (ml_f['pickup_hour'] <= max_hour)]

# ── Filter zone data ──
zones_f = zones.copy()
if has_event == 'Ya (has_event=1)':
    zones_f = zones_f[zones_f['event_trip_pct'] > 0]
elif has_event == 'Tidak (has_event=0)':
    zones_f = zones_f[zones_f['event_trip_pct'] == 0]

# ── Tabs ──
tab1, tab2, tab3 = st.tabs(['🗺️  Peta Surge', '📊  Analisis Event', '🤖  ML Insights'])

# ════════════════════════════════════════
# TAB 1: PETA SURGE
# ════════════════════════════════════════
with tab1:
    st.subheader('Distribusi Trip per Zona Taksi NYC')

    col1, col2, col3, col4 = st.columns(4)
    col1.metric('Total Trip', f'{zones["total_trips"].sum():,}')
    col2.metric('Rata-rata Fare', f'${zones["avg_fare"].mean():.2f}')
    col3.metric('Zona Terpadat', zones.loc[zones['total_trips'].idxmax(), 'zone'])
    col4.metric('Event Trip (%)', f'{zones["event_trip_pct"].mean():.1f}%')

    if geo:
        m = folium.Map(location=[40.73, -73.94], zoom_start=11, tiles='CartoDB positron')

        def style_fn(f):
            loc_id = int(f['properties'].get('locationid', 0))
            row = zones_f[zones_f['location_id'] == loc_id]
            if row.empty:
                return {'fillColor': '#cccccc', 'color': '#ffffff', 'weight': 1, 'fillOpacity': 0.5}
            trips = row['total_trips'].iloc[0]
            max_t = zones_f['total_trips'].max() or 1
            intensity = min(trips / max_t, 1)
            r = int(255 * (1 - intensity))
            g = int(255 * (1 - intensity * 0.5))
            b = int(255 * (1 - intensity * 0.8))
            color = f'#{r:02x}{g:02x}{b:02x}'
            return {'fillColor': color, 'color': '#ffffff', 'weight': 1, 'fillOpacity': 0.7}

        def highlight_fn(f):
            return {'weight': 3, 'color': '#333333', 'fillOpacity': 0.9}

        tooltip = folium.GeoJsonTooltip(
            fields=['zone', 'borough'],
            aliases=['Zona:', 'Borough:'],
            localize=True
        )

        gj = folium.GeoJson(
            geo,
            style_function=style_fn,
            highlight_function=highlight_fn,
            tooltip=tooltip,
            name='Zona Taksi'
        )
        gj.add_to(m)

        folium.LayerControl().add_to(m)
        st_folium(m, width=None, height=500)
    else:
        st.warning('Gagal memuat data peta. Menampilkan tabel zona:')
        st.dataframe(zones_f.sort_values('total_trips', ascending=False).head(20))

# ════════════════════════════════════════
# TAB 2: ANALISIS EVENT
# ════════════════════════════════════════
with tab2:
    st.subheader('Dampak Event terhadap Taksi NYC')

    col_a, col_b = st.columns(2)

    with col_a:
        if not trips_hour.empty:
            fig = px.bar(
                trips_hour, x='hour',
                y=['trips_normal', 'trips_event'],
                title='Trip per Jam: Event vs Normal',
                labels={'value': 'Jumlah Trip', 'hour': 'Jam'},
                barmode='group',
                color_discrete_map={'trips_normal': '#1f77b4', 'trips_event': '#ff7f0e'}
            )
            fig.update_layout(legend_title_text='')
            st.plotly_chart(fig, use_container_width=True)

    with col_b:
        if not top_zones.empty:
            fig = px.bar(
                top_zones.head(10),
                x='total_trips', y='zone',
                color='borough',
                title='Top 10 Zona Pickup Saat Event',
                labels={'total_trips': 'Jumlah Trip', 'zone': 'Zona'},
                orientation='h'
            )
            fig.update_layout(yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig, use_container_width=True)

    col_c, col_d = st.columns(2)

    with col_c:
        if not fare_comp.empty:
            fare_comp['label'] = fare_comp['has_event_nearby'].map({0: 'Tanpa Event', 1: 'Dengan Event'})
            fig = go.Figure()
            fig.add_trace(go.Bar(
                name='Avg Fare',
                x=fare_comp['label'], y=fare_comp['avg_fare'],
                marker_color=['#1f77b4', '#ff7f0e'],
                text=fare_comp['avg_fare'].apply(lambda x: f'${x:.2f}')
            ))
            fig.update_layout(title='Rata-rata Fare', yaxis_title='Fare ($)')
            st.plotly_chart(fig, use_container_width=True)

    with col_d:
        if not fare_comp.empty:
            fig = go.Figure()
            fig.add_trace(go.Bar(
                name='Avg Duration',
                x=fare_comp['label'], y=fare_comp['avg_duration'],
                marker_color=['#1f77b4', '#ff7f0e'],
                text=fare_comp['avg_duration'].apply(lambda x: f'{x:.1f} min')
            ))
            fig.update_layout(title='Rata-rata Durasi Trip', yaxis_title='Durasi (menit)')
            st.plotly_chart(fig, use_container_width=True)

    if not event_type.empty:
        st.subheader('Dampak per Tipe Event')
        fig = px.bar(
            event_type, x='event_type', y='total_trips',
            color='avg_tip',
            title='Total Trip per Tipe Event (warna = rata-rata tip)',
            labels={'event_type': 'Tipe Event', 'total_trips': 'Jumlah Trip', 'avg_tip': 'Rata-rata Tip ($)'},
            color_continuous_scale='Blues'
        )
        st.plotly_chart(fig, use_container_width=True)

        with st.expander('Detail Data Event'):
            st.dataframe(event_type)

# ════════════════════════════════════════
# TAB 3: ML INSIGHTS
# ════════════════════════════════════════
with tab3:
    st.subheader('Prediksi Surge dengan Machine Learning')

    if ml_pred.empty:
        st.warning('Jalankan ml/predict.py dulu untuk menghasilkan prediksi.')
    else:
        # Ringkasan
        tot = len(ml_f)
        surge_cnt = ml_f['is_surge'].sum()
        st.markdown(f'**{tot:,}** kombinasi (jam, hari, event, lokasi) — **{surge_cnt:,}** diprediksi surge ({surge_cnt/tot*100:.1f}%)')

        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        col_m1.metric('Surge Probability Rata-rata', f'{ml_f["surge_probability"].mean():.1%}')
        col_m2.metric('Akurasi Model', '95.5% (random split)')
        col_m3.metric('Fitur Utama', 'Waktu + Event + Lokasi')
        col_m4.metric('Prediksi Surge', f'{surge_cnt:,} / {tot:,}')

        # Surge by hour
        surge_hour = ml_f.groupby('pickup_hour')['is_surge'].mean().reset_index()
        surge_hour.columns = ['hour', 'surge_rate']
        fig = px.bar(
            surge_hour, x='hour', y='surge_rate',
            title='Proporsi Surge per Jam',
            labels={'hour': 'Jam', 'surge_rate': 'Proporsi Surge'},
            color='surge_rate', color_continuous_scale='Reds'
        )
        fig.update_layout(yaxis_tickformat='.0%')
        st.plotly_chart(fig, use_container_width=True)

        col_e, col_f = st.columns(2)

        with col_e:
            # Surge by weekday
            day_map = {0:'Sen',1:'Sel',2:'Rab',3:'Kam',4:'Jum',5:'Sab',6:'Min'}
            surge_day = ml_f.groupby('pickup_dayofweek')['surge_probability'].mean().reset_index()
            surge_day['day_label'] = surge_day['pickup_dayofweek'].map(day_map)
            fig = px.line(
                surge_day, x='day_label', y='surge_probability',
                title='Rata-rata Surge Probability per Hari',
                labels={'day_label': 'Hari', 'surge_probability': 'Probabilitas'},
                markers=True
            )
            fig.update_layout(yaxis_tickformat='.0%')
            st.plotly_chart(fig, use_container_width=True)

        with col_f:
            # Event vs Non-event surge
            ev_comp = ml_f.groupby('has_event_nearby')['is_surge'].mean().reset_index()
            ev_comp['label'] = ev_comp['has_event_nearby'].map({0: 'Tanpa Event', 1: 'Dengan Event'})
            fig = px.bar(
                ev_comp, x='label', y='is_surge',
                title='Surge Rate: Event vs Normal',
                labels={'label': '', 'is_surge': 'Proporsi Surge'},
                color='label',
                color_discrete_map={'Tanpa Event': '#1f77b4', 'Dengan Event': '#ff7f0e'}
            )
            fig.update_layout(yaxis_tickformat='.0%', showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        # Tabel prediksi
        with st.expander('Lihat Data Prediksi'):
            st.dataframe(ml_f.head(500))
