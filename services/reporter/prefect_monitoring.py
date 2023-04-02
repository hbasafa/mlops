import os
import json
import pickle
import pandas
import pyarrow.parquet as pq
from datetime import datetime
from prefect import flow, task
from pymongo import MongoClient
from predictor import ColumnMapping
from predictor.dashboard import Dashboard
from predictor.model_profile import Profile
from predictor.dashboard.tabs import ClassificationPerformanceTab
from predictor.model_profile.sections import ClassificationPerformanceProfileSection


MONGO_CLIENT_ADDRESS = "mongodb://localhost:27018/"
MONGO_DATABASE = "prediction_service"
PREDICTION_COLLECTION = "data"
REPORT_COLLECTION = "report"
REFERENCE_DATA_FILE = "./predictor_service/datasets/sample_test_data.csv"
TARGET_DATA_FILE = "target.csv"
MODEL_FILE = os.getenv('MODEL_FILE', './prediction_service/model.pkl')


@task
def upload_target(filename):
    client = MongoClient(MONGO_CLIENT_ADDRESS)
    collection = client.get_database(MONGO_DATABASE).get_collection(
        PREDICTION_COLLECTION
    )
    with open(filename) as f_target:
        for line in f_target.readlines():
            row = line
            collection.update_one({"id": row[0]}, {"$set": {"target": str(row[1])}})


@task
def load_reference_data(filename):

    with open(MODEL_FILE, 'rb') as f_in:
        model = pickle.load(f_in)
    reference_data = pandas.read_csv(filename, delimiter=',')
    
    # add target column
    reference_data['target'] = reference_data['Pass/Fail']
    reference_data = reference_data.drop('Pass/Fail', axis=1, inplace=False)
    reference_data['prediction'] = model.predict(
        reference_data.drop('target', axis=1, inplace=False)
    )
    return reference_data


@task
def fetch_data():
    client = MongoClient(MONGO_CLIENT_ADDRESS)
    data = (
        client.get_database(MONGO_DATABASE).get_collection(PREDICTION_COLLECTION).find()
    )
    df = pandas.DataFrame(list(data))
    df['target'] = df['Pass/Fail']
    return df


@task
def run_predictor(ref_data, data):

    profile = Profile(sections=[ClassificationPerformanceProfileSection()])
    mapping = ColumnMapping(
        prediction="prediction",
        numerical_features=[
            '0',
            '1',
            '2',
            '3',
            '4',
            '5',
            '6',
            '7',
            '8',
            '9',
        ],
        datetime_features=[],
    )
    # print(ref_data)
    # print(data)
    profile.calculate(ref_data, data, mapping)

    dashboard = Dashboard(tabs=[ClassificationPerformanceTab()])
    dashboard.calculate(ref_data, data, mapping)
    return json.loads(profile.json()), dashboard


@task
def save_report(result):
    """Save evidendtly profile for ride prediction to mongo server"""

    client = MongoClient(MONGO_CLIENT_ADDRESS)
    collection = client.get_database(MONGO_DATABASE).get_collection(REPORT_COLLECTION)
    collection.insert_one(result)


@task
def save_html_report(result, filename_suffix=None):
    """Create predictor html report file for ride prediction"""

    if filename_suffix is None:
        filename_suffix = datetime.now().strftime('%Y-%m-%d-%H-%M')

    result.save(f"semicon_class_report_{filename_suffix}.html")


@flow
def batch_analyze():
    upload_target(TARGET_DATA_FILE)
    ref_data = load_reference_data(REFERENCE_DATA_FILE).result()
    data = fetch_data().result()
    profile, dashboard = run_predictor(ref_data, data).result()
    save_report(profile)
    save_html_report(dashboard)


batch_analyze()
