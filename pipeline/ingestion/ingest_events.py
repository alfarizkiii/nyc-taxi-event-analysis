import requests, os, csv, math
from dotenv import load_dotenv
load_dotenv()

EVENT_API = 'https://data.cityofnewyork.us/resource/bkfu-528j.json'
APP_TOKEN = os.getenv('NYC_APP_TOKEN', '')

EVENT_FILTER = "(event_type LIKE '%Parade%' OR event_type LIKE '%Festival%' OR event_type LIKE '%Concert%' OR event_type LIKE '%Race%' OR event_type = 'Street Event' OR event_type = 'Block Party')"

def ingest_events():
    os.makedirs('data/raw/events', exist_ok=True)
    all_events = []
    offset = 0
    limit = 50000
    # Sync date range with taxi data: Jan-Jun 2025
    where = f"start_date_time >= '2025-01-01' AND start_date_time < '2025-07-01' AND ({EVENT_FILTER})"
    # First pass: get first batch to estimate total
    params = {
        '$select': 'COUNT(*)',
        '$where': where,
        '$$app_token': APP_TOKEN
    }
    resp = requests.get(EVENT_API, params=params).json()
    total = int(resp[0]['COUNT']) if resp else 0
    print(f'[INFO] Total events sesuai filter: {total:,}')

    for offset in range(0, total, limit):
        params = {
            '$limit': limit, '$offset': offset,
            '$where': where,
            '$$app_token': APP_TOKEN
        }
        resp = requests.get(EVENT_API, params=params).json()
        if not resp: break
        all_events.extend(resp)
        print(f'[INFO] Diunduh {len(all_events):,} / {total:,} events...')

    if all_events:
        keys = all_events[0].keys()
        with open('data/raw/events/events_2025.csv', 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader(); w.writerows(all_events)
    print(f'[OK] {len(all_events)} events diunduh')

if __name__ == '__main__':
    ingest_events()
