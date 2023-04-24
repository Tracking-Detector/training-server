import io
# from contextlib import redirect_stdout

import tensorflow as tf
import tensorflowjs as tfjs

import minio_client


def train_and_save_model(data_frame, model, vector_length, batch_size, epochs):
    print("Started Training")
    X = data_frame.to_numpy()[:, 0:vector_length]
    y = data_frame.to_numpy()[:, -1]
    model.fit(X, y, batch_size=batch_size, epochs=epochs)
    evaluation = model.evaluate(X, y)
    print("Model evaluation ", evaluation)
    tfjs.converters.save_keras_model(model, "model")
    return evaluation[1]


def status():
    print("Status was called")
    return {"status": "RUNNING"}


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
    # with redirect_stdout(f):
    try:
        training_data = minio_client.extract_data(trainingDataFilename)
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
        minio_client.upload_model_to_bucket(applicationName, modelStorageName)
        minio_client.clean_up_files()
        return {"accuracy": acc, "logs": f.getvalue()}
    except Exception as e:
        print(e)
        minio_client.clean_up_files()
        return {"error": str(e), "logs": f.getvalue()}
