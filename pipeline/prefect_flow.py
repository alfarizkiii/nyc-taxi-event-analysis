from prefect import flow, task
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from ingestion.ingest_taxi   import ingest_taxi
from ingestion.ingest_events import ingest_events
from preprocessing.clean_taxi   import clean_taxi
from preprocessing.clean_events import clean_events
from modeling.build_star_schema import build_star_schema

@task(name='Ingest Taxi Data', retries=2, retry_delay_seconds=30)
def task_ingest_taxi(): ingest_taxi()

@task(name='Ingest Events Data', retries=2, retry_delay_seconds=30)
def task_ingest_events(): ingest_events()

@task(name='Clean Taxi Data')
def task_clean_taxi(): return clean_taxi()

@task(name='Clean Events Data')
def task_clean_events(): return clean_events()

@task(name='Build Star Schema')
def task_build_schema(): return build_star_schema()

@flow(name='NYC Taxi Event Pipeline')
def main_flow():
    task_ingest_taxi()
    task_ingest_events()
    task_clean_taxi()
    task_clean_events()
    task_build_schema()
    print('[DONE] Pipeline selesai!')

if __name__ == '__main__':
    main_flow()
