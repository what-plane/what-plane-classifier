import os
import io
from uuid import UUID

from fastapi import HTTPException
from PIL import Image
from azure.storage.blob import BlobServiceClient, BlobClient

IMAGE_UPLOAD_CONTAINTER = "uploaded-images"
CLASSIFIED_IMAGE_CONTAINER = "uploaded-images-airliners"
CONNECT_STR = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
blob_service_client = BlobServiceClient.from_connection_string(CONNECT_STR)


def is_valid_uuid(uuid_to_test: str, version: int = 4) -> bool:
    """Function to check if uuid_to_test is a valid UUID.

    Args:
        uuid_to_test (str): UUID string to be tested
        version (int, optional): UUID Version. Defaults to 4.

    Returns:
        bool: Is UUID valid or not
    """

    try:
        uuid_obj = UUID(uuid_to_test, version=version)
    except ValueError:
        return False
    return str(uuid_obj) == uuid_to_test


class ImageBlobClient:
    """Class to interact with images stored in Azure Blob Storage that have
    been uploaded by the React frontend

    Attributes:
        UUID: The UUID associated with the Blob
        uploaded_blob: The Azure BlobClient associated with the Blob
    """

    def __init__(self, uuid: str):
        """Inits ImageBlobClient with UUID and BlobClient for uploaded blob

        Args:
            uuid (str): UUID of blob
        """
        self.UUID = self._validate_uuid(uuid)
        self.uploaded_blob = self._validate_uploaded_blob()

    @staticmethod
    def _validate_uuid(uuid: str) -> str:
        """Set UUID if valid, otherwise raise an error

        Args:
            uuid (str): UUID of blob

        Raises:
            HTTPException: HTTP Error 400 for Invalid UUID

        Returns:
            str: UUID of blob
        """
        if not is_valid_uuid(uuid):
            raise HTTPException(status_code=400, detail="Invalid UUID")

        return uuid

    def _validate_uploaded_blob(self) -> BlobClient:
        """Validate the blob exists and is the right content type before
        returning a BlobClient connected to the Blob with name UUID

        Raises:
            HTTPException: HTTP Error 404 if Blob not found
            HTTPException: HTTP Error 415 if Blob is the wrong content type

        Returns:
            BlobClient: BlobClient for the Blob with name UUID
        """
        uploaded_blob = blob_service_client.get_blob_client(
            container=IMAGE_UPLOAD_CONTAINTER, blob=self.UUID
        )

        if not uploaded_blob.exists():
            raise HTTPException(status_code=404, detail="Resource not found")

        content_type = uploaded_blob.get_blob_properties()["content_settings"]["content_type"]

        if content_type not in ["image/jpeg", "image/png"]:
            raise HTTPException(status_code=415, detail=f"File type {content_type} not supported")

        return uploaded_blob

    def get_uploaded_image(self) -> Image.Image:
        """Load image data from Blob, convert to RGB (for .png) and return as PIL Image

        Raises:
            HTTPException: HTTP Error 400 if the Blob could not be read in as an Image

        Returns:
            Image.Image: PIL Image Data from Blob
        """

        try:
            img_bytes = io.BytesIO(self.uploaded_blob.download_blob().readall())
            image = Image.open(img_bytes).convert("RGB")
        except Exception:
            raise HTTPException(
                status_code=400, detail=f"Unable to read image associated with provided UUID"
            )
        return image

    def copy_classified_blob(self, likely_class: str):
        """Copy a image (blob) that has been classified into a longer term storage Container
        (uploaded-images-airliners) to be used for model re-training

        Args:
            likely_class (str): The most probable class predicted (the folder the blob should
            be moved into)

        Returns:
            self: The ImageBlobClient object instance
        """
        airliner_blob = blob_service_client.get_blob_client(
            container=CLASSIFIED_IMAGE_CONTAINER, blob="/".join([likely_class, self.UUID])
        )
        airliner_blob.start_copy_from_url(self.uploaded_blob.url)

        return self
