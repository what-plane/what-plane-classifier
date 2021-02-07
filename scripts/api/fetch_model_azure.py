import os
from pathlib import Path
import argparse

from azure.storage.blob import BlobServiceClient

MODEL_CONTAINER = "models"
CONN_STR_NAME = "AZURE_STORAGE_CONNECTION_STRING"
SECRET_PATH = Path("/run/secrets/ENV")


def fetch_model(blob_name, out_filepath, connection_string):

    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    model_blob = blob_service_client.get_blob_client(container=MODEL_CONTAINER, blob=blob_name)

    with open(out_filepath, "wb") as outfile:
        outfile.write(model_blob.download_blob().readall())

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_name", type=str, help="Name of the model blob in Azure")
    parser.add_argument("out_filepath", type=str, help="Folder to save model file")
    args = parser.parse_args()

    out_filepath = Path(args.out_filepath) / args.model_name

    if os.getenv(CONN_STR_NAME, default="") != "":
        connect_str = os.getenv(CONN_STR_NAME)
    elif SECRET_PATH.exists():
        try:
            parsed_env = [
                line.replace(CONN_STR_NAME + "=", "")
                for line in SECRET_PATH.read_text().split("\n")
                if line.startswith(CONN_STR_NAME)
            ]
            assert len(parsed_env) == 1
            connect_str = "".join(parsed_env)

        except:
            raise RuntimeError(f"Unable to correctly parse secrets file {SECRET_PATH}")

    else:
        raise RuntimeError(
            (
                f"Please set the {CONN_STR_NAME} environment variable "
                f"or add a {CONN_STR_NAME} file in /run/secrets"
            )
        )

    print(f"Downloading {args.model_name}...")
    fetch_model(args.model_name, out_filepath, connect_str)
    print(f"Downloaded {args.model_name} to {args.out_filepath}")
