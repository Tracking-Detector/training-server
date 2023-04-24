import json
import os

import pandas as pd
import pytest
import tensorflow as tf
from tensorflow.keras import Input, Sequential
from tensorflow.keras.layers import Dense

from handle import handler


@pytest.fixture(scope="session", autouse=True)
def set_env():
    os.environ["MINIO_HOST_NAME"] = "minio"
    os.environ["MINIO_PORT"] = "9000"
    os.environ["MINIO_ACCESS_KEY"] = "some-key"
    os.environ["MINIO_PRIVATE_KEY"] = "some-pw"
    os.environ["TRAINING_DATA_BUCKET_NAME"] = "some-bucket-1"
    os.environ["MODEL_BUCKET_NAME"] = "some-bucket-2"


def test_should_train_model_from_model_json(mocker):
    # given
    mock_app_name = "MockApplication"
    mock_model_storage_name = "MockStorageName"
    mock_training_file_name = "MockTrainingFile"
    mock_model = json.dumps(
        tf.keras.layers.serialize(
            Sequential(
                layers=[
                    Input(shape=(4,)),
                    Dense(10, input_shape=(4,)),
                    Dense(1, input_shape=(10,)),
                ]
            )
        )
    )
    mock_vector_length = 4
    mock_batch_size = 1
    mock_epochs = 10
    mock_data = [[1, 0.5, 7, 10, -1], [0, 0.5, 1, 2, 1], [1, 0.5, 1, 2, 1]]
    mock_extract_data = mocker.patch(
        "minio_client.extract_data", return_value=pd.DataFrame(mock_data)
    )
    mock_upload = mocker.patch("minio_client.upload_model_to_bucket", return_value=None)
    mock_cleanup = mocker.patch("minio_client.clean_up_files", return_value=None)

    # when
    res = handler(
        mock_app_name,
        mock_model_storage_name,
        mock_training_file_name,
        mock_model,
        mock_vector_length,
        mock_batch_size,
        mock_epochs,
    )

    # then
    mock_extract_data.assert_called_once_with(mock_training_file_name)
    mock_upload.assert_called_once_with(mock_app_name, mock_model_storage_name)
    mock_cleanup.assert_called_once()
    assert len(res) == 2
    assert res["accuracy"] >= 0.0
    assert res["accuracy"] <= 1.0
    assert len(res["logs"]) > 0


def test_should_fail_when_model_json_not_correct(mocker):
    # given
    mock_app_name = "MockApplication"
    mock_model_storage_name = "MockStorageName"
    mock_training_file_name = "MockTrainingFile"
    mock_model = '{"test":1}'
    mock_vector_length = 4
    mock_batch_size = 1
    mock_epochs = 10
    mock_data = [[1, 0.5, 7, 10, -1], [0, 0.5, 1, 2, 1], [1, 0.5, 1, 2, 1]]
    mock_extract_data = mocker.patch(
        "minio_client.extract_data", return_value=pd.DataFrame(mock_data)
    )
    mock_cleanup = mocker.patch("minio_client.clean_up_files", return_value=None)

    # when
    res = handler(
        mock_app_name,
        mock_model_storage_name,
        mock_training_file_name,
        mock_model,
        mock_vector_length,
        mock_batch_size,
        mock_epochs,
    )

    # then
    mock_extract_data.assert_called_once_with(mock_training_file_name)

    mock_cleanup.assert_called_once()
    assert len(res) == 2
    assert (
        res["error"]
        == "Improper config format for {'test': 1}. Expecting python dict contains `class_name` and `config` as keys"
    )
    assert (
        res["logs"]
        == "Improper config format for {'test': 1}. Expecting python dict contains `class_name` and `config` as keys\n"
    )
