import requests, os, csv
from dotenv import load_dotenv
load_dotenv()

EVENT_API = 'https://data.cityofnewyork.us/resource/bkfu-528j.json'
APP_TOKEN = os.getenv('NYC_APP_TOKEN', '')

def ingest_events():
    os.makedirs('data/raw/events', exist_ok=True)
    all_events = []
    offset = 0
    while True:
        params = {
            '$limit': 1000, '$offset': offset,
            '$where': "start_date_time >= '2025-01-01'",
            '$$app_token': APP_TOKEN
        }
        resp = requests.get(EVENT_API, params=params).json()
        if not resp: break
        all_events.extend(resp)
        offset += 1000
    # Simpan ke CSV
    if all_events:
        keys = all_events[0].keys()
        with open('data/raw/events/events_2025.csv', 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader(); w.writerows(all_events)
    print(f'[OK] {len(all_events)} events diunduh')

if __name__ == '__main__':
    ingest_events()
