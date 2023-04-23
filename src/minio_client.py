import gzip
import io
import os
import shutil
from contextlib import redirect_stdout

import pandas as pd
from minio import Minio

MINIO_HOST_NAME = os.environ["MINIO_HOST_NAME"]
MINIO_PORT = os.environ["MINIO_PORT"]
MINIO_ACCESS_KEY = os.environ["MINIO_ACCESS_KEY"]
MINIO_SECRET_KEY = os.environ["MINIO_PRIVATE_KEY"]
DATA_BUCKET_NAME = os.environ["TRAINING_DATA_BUCKET_NAME"]
MODEL_BUCKET_NAME = os.environ["MODEL_BUCKET_NAME"]

TMP_COMPRESSED_FILE_NAME = "training-data.csv.gz"
TMP_UNCOMPRESSED_FILE_NAME = "training-data.csv"

minio_client = Minio(
    f"{MINIO_HOST_NAME}:{MINIO_PORT}",
    access_key=MINIO_ACCESS_KEY,
    secret_key=MINIO_SECRET_KEY,
    secure=False,
)


def extract_data(training_data_filename):
    response = minio_client.get_object(DATA_BUCKET_NAME, training_data_filename)
    with open(TMP_COMPRESSED_FILE_NAME, "wb") as file_data:
        for d in response.stream(32 * 1024):
            file_data.write(d)
    with gzip.open(TMP_COMPRESSED_FILE_NAME, "rb") as compressed_file:
        with open(TMP_UNCOMPRESSED_FILE_NAME, "wb") as uncompressed_file:
            shutil.copyfileobj(compressed_file, uncompressed_file)
    response.close()
    response.release_conn()
    print("Finsihed downloading and extracting data")

    return pd.concat(
        [
            chunk
            for chunk in pd.read_csv(
                TMP_UNCOMPRESSED_FILE_NAME, sep=",", chunksize=1000
            )
        ]
    )


def clean_up_files():
    os.remove(TMP_UNCOMPRESSED_FILE_NAME)
    os.remove(TMP_COMPRESSED_FILE_NAME)
    shutil.rmtree("./model")


def upload_model_to_bucket(application_name, model_storage_name):
    print("Starting upload")
    for root, _, files in os.walk("./model"):
        for file in files:
            file_path = os.path.join(root, file)
            minio_client.fput_object(
                MODEL_BUCKET_NAME,
                application_name + "/" + model_storage_name + "/" + file,
                file_path,
            )
