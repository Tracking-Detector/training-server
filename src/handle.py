import gzip
import io
import os
import shutil
from contextlib import redirect_stdout

import pandas as pd
import tensorflow as tf
import tensorflowjs as tfjs

from minio_client import clean_up_files, extract_data, upload_model_to_bucket

MINIO_ACCESS_KEY = os.environ["MINIO_ACCESS_KEY"]
MINIO_SECRET_KEY = os.environ["MINIO_PRIVATE_KEY"]
DATA_BUCKET_NAME = os.environ["TRAINING_DATA_BUCKET_NAME"]
MODEL_BUCKET_NAME = os.environ["MODEL_BUCKET_NAME"]


def train_and_save_model(data_frame, model, vector_length, batch_size, epochs):
    print("Started Training")
    X = data_frame.to_numpy()[:, 0:vector_length]
    y = data_frame.to_numpy()[:, -1]
    model.fit(X, y, batch_size=batch_size, epochs=epochs)
    evaluation = model.evaluate(X, y)
    print("Model evaluation ", evaluation)
    tfjs.converters.save_keras_model(model, "model")
    return evaluation[1]


def handler(
    applicationName,
    modelStorageName,
    trainingDataFilename,
    modelJson,
    vectorLength,
    batchSize,
    epochs,
):
    f = io.StringIO()
    with redirect_stdout(f):
        try:
            training_data = extract_data(trainingDataFilename)
            model = tf.keras.models.model_from_json(modelJson)
            model.compile(
                loss="mse",
                optimizer=tf.keras.optimizers.RMSprop(learning_rate=1e-2),
                metrics=["accuracy"],
            )
            print(model.summary())
            acc = train_and_save_model(
                training_data, model, vectorLength, batchSize, epochs
            )
            upload_model_to_bucket(applicationName, modelStorageName)
            delete_temporary_files()
            return {"accuracy": acc, "logs": f.getvalue()}
        except Exception as e:
            print(e)
            delete_temporary_files()
            clean_up_files()
            return {"error": e, "logs": f.getvalue()}
