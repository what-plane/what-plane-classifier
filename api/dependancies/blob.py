import os
import io
from uuid import UUID

from fastapi import HTTPException
from PIL import Image
from azure.storage.blob import BlobServiceClient

IMAGE_UPLOAD_CONTAINTER = "uploaded-images"
CLASSIFIED_IMAGE_CONTAINER = "uploaded-images-airliners"
CONNECT_STR = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
blob_service_client = BlobServiceClient.from_connection_string(CONNECT_STR)


def is_valid_uuid(uuid_to_test, version=4):
    """
    Check if uuid_to_test is a valid UUID.

     Parameters
    ----------
    uuid_to_test : str
    version : {1, 2, 3, 4}

     Returns
    -------
    `True` if uuid_to_test is a valid UUID, otherwise `False`.

     Examples
    --------
    >>> is_valid_uuid('c9bf9e57-1685-4c89-bafb-ff5af830be8a')
    True
    >>> is_valid_uuid('c9bf9e58')
    False
    """

    try:
        uuid_obj = UUID(uuid_to_test, version=version)
    except ValueError:
        return False
    return str(uuid_obj) == uuid_to_test


class ImageBlobClient:
    def __init__(self, uuid):
        self.UUID = self._validate_uuid(uuid)
        self.uploaded_blob = self._validate_uploaded_blob()

    @staticmethod
    def _validate_uuid(uuid):
        if not is_valid_uuid(uuid):
            raise HTTPException(status_code=400, detail="Invalid UUID")

        return uuid

    def _validate_uploaded_blob(self):
        uploaded_blob = blob_service_client.get_blob_client(
            container=IMAGE_UPLOAD_CONTAINTER, blob=self.UUID
        )

        if not uploaded_blob.exists():
            raise HTTPException(status_code=404, detail="Resource not found")

        content_type = uploaded_blob.get_blob_properties()["content_settings"]["content_type"]

        if content_type not in ["image/jpeg", "image/png"]:
            raise HTTPException(status_code=415, detail=f"File type {content_type} not supported")

        return uploaded_blob

    def get_uploaded_image(self):

        img_bytes = io.BytesIO(self.uploaded_blob.download_blob().readall())
        image = Image.open(img_bytes).convert("RGB")

        return image

    def copy_classified_blob(self, likely_class):
        airliner_blob = blob_service_client.get_blob_client(
            container=CLASSIFIED_IMAGE_CONTAINER, blob="/".join([likely_class, self.UUID])
        )
        airliner_blob.start_copy_from_url(self.uploaded_blob.url)

        return self
