import os
from pathlib import Path
import argparse

from azure.storage.blob import BlobServiceClient

MODEL_CONTAINER = "models"
CONNECT_STR = os.getenv("AZURE_STORAGE_CONNECTION_STRING")


def fetch_model(blob_name, out_filepath):

    blob_service_client = BlobServiceClient.from_connection_string(CONNECT_STR)
    model_blob = blob_service_client.get_blob_client(container=MODEL_CONTAINER, blob=blob_name)

    with open(out_filepath, "wb") as outfile:
        outfile.write(model_blob.download_blob().readall())

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_name", type=str, help="Name of the model blob in Azure")
    parser.add_argument("out_filepath", type=str, help="Location to save model file")
    args = parser.parse_args()

    out_filepath = Path(args.out_filepath)

    if os.getenv("AZURE_STORAGE_CONNECTION_STRING", default="") == "":
        raise RuntimeError("Please set the AZURE_STORAGE_CONNECTION_STRING environment variable")

    print(f"Downloading {args.model_name}...")
    fetch_model(args.model_name, out_filepath)
    print(f"Downloaded {args.model_name} to {args.out_filepath}")
