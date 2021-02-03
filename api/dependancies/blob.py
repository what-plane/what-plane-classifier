import os
import io

from fastapi import HTTPException
from PIL import Image
from azure.storage.blob import BlobServiceClient

IMAGE_UPLOAD_CONTAINTER = "uploaded-images"
CLASSIFIED_IMAGE_CONTAINER = "uploaded-images-airliners"
CONNECT_STR = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
blob_service_client = BlobServiceClient.from_connection_string(CONNECT_STR)


class ImageBlobClient:
    def __init__(self, uuid):
        self.UUID = uuid
        self.uploaded_blob = self._return_uploaded_blob()

    def _return_uploaded_blob(self):
        uploaded_blob = blob_service_client.get_blob_client(
            container=IMAGE_UPLOAD_CONTAINTER, blob=self.UUID
        )

        if not uploaded_blob.exists():
            raise HTTPException(status_code=404, detail="Resource not found")

        content_type = uploaded_blob.get_blob_properties()["content_settings"]["content_type"]

        if content_type not in ["image/jpeg", "image/png"]:
            raise HTTPException(status_code=404, detail=f"File type {content_type} not supported")

        return uploaded_blob

    def get_uploaded_image(self):

        image = Image.open(io.BytesIO(self.uploaded_blob.download_blob().readall()))

        return image

    def copy_classified_blob(self, likely_class):
        airliner_blob = blob_service_client.get_blob_client(
            container=CLASSIFIED_IMAGE_CONTAINER, blob="/".join([likely_class, self.uuid])
        )
        airliner_blob.start_copy_from_url(self.uploaded_blob.url)

        return self
