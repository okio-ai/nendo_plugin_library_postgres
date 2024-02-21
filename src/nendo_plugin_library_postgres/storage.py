# -*- encoding: utf-8 -*-
"""Additional storage drivers for the postgres library plugin."""

import hashlib
import json
import logging
import os
import pickle
import uuid
from enum import Enum
from importlib import metadata
from tempfile import NamedTemporaryFile
from typing import Any, List, Optional

import numpy as np
import soundfile as sf
from google.cloud import storage
from google.cloud.exceptions import NotFound
from sqlalchemy import MetaData
from sqlalchemy.orm import declarative_base

from nendo import NendoStorage, ResourceLocation

from .config import PostgresConfig

plugin_package = metadata.metadata(__package__ or __name__)
plugin_config = PostgresConfig()
Base = declarative_base(metadata=MetaData())
logger = logging.getLogger("nendo")


class TrackType(str, Enum):
    """Enum representing different types of `NendoTrack`s."""

    voice: str = "voice"
    track: str = "track"
    loop: str = "loop"
    singleshot: str = "singleshot"


class CollectionType(str, Enum):
    """Enum representing different types of Collections in Nendo."""

    playlist: str = "playlist"
    favorites: str = "favorites"
    stems: str = "stems"
    loops: str = "loops"
    generic: str = "generic"
    quantized: str = "quantized"
    temp: str = "temp"


class NendoStorageGCS(NendoStorage):
    environment: str = "local"
    storage_client: storage.Client = None

    class Config:
        arbitrary_types_allowed = True

    def __init__(
        self,
        environment: str,
        credentials_json: str,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        logger.info(f"Initializing storage for environment {environment}")
        self.environment = environment
        try:
            self.storage_client = storage.Client.from_service_account_info(
                info=json.loads(credentials_json),
            )
        except Exception as e:
            logger.error(
                "Error initializing GCS. Please check that your have "
                "properly configured the google_storage_credentials config "
                "variable to contain a json key with access to your "
                "cloud storage. Error: %s",
                e,
            )

    def _get_bucket_name_for_user(self, user_id: Optional[str] = None) -> str:
        """Returns the name of the bucket for the user with the given user_id.

        Args:
            user_id (str, optional): ID of the user for which the bucket name
                should be returned.

        Returns:
            str: The name of the user bucket.
        """
        if user_id is None:
            return f"no-user-nendo-{self.environment}"

        return f"{user_id}-nendo-{self.environment}"

    def _get_user_bucket(self, user_id: str) -> storage.bucket.Bucket:
        bucket_name = self._get_bucket_name_for_user(user_id)
        try:
            return self.storage_client.bucket(bucket_name)
        except NotFound:
            logger.warning(
                "Bucket with name %s was not found, creating now",
                bucket_name,
            )
            return self.init_storage_for_user(user_id=user_id)

    def init_storage_for_user(self, user_id: str) -> storage.bucket.Bucket:
        try:
            """Create a new bucket in specific location with storage class"""
            bucket_name = self._get_bucket_name_for_user(user_id)
            try:
                return self.storage_client.get_bucket(bucket_name)
            except NotFound:
                try:
                    bucket = self.storage_client.create_bucket(bucket_name)

                    # Make the objects within the bucket publicly readable
                    policy = bucket.get_iam_policy(requested_policy_version=3)
                    policy.version = 3

                    # Modify bindings using the bindings property
                    policy.bindings.append(
                        {"role": "roles/storage.objectViewer", "members": ["allUsers"]},
                    )

                    bucket.set_iam_policy(policy)

                    logger.info(
                        "Bucket %s created and objects are publicly readable.",
                        bucket.name,
                    )

                    return bucket
                except Exception as e:
                    logger.error(f"Error creating bucket: {bucket_name} error: {e}")

            return bucket
        except Exception as e:
            logger.error(f"Error initializing the storage client: {e}")

    def generate_filename(self, filetype: str, user_id: str) -> str:
        """Generate a unique filename."""
        return f"{uuid.uuid4()!s}.{filetype}"

    def file_exists(self, file_name: str, user_id: str) -> bool:
        bucket = self._get_user_bucket(user_id=user_id)
        blob = bucket.blob(file_name)
        return blob.exists()

    def as_local(self, file_name: str, user_id: str) -> str:
        bucket = self._get_user_bucket(user_id=user_id)
        blob = bucket.blob(file_name)
        _, file_extension = os.path.splitext(file_name)
        with NamedTemporaryFile(suffix=file_extension, delete=False) as tmpfile:
            # TODO this ugly hack is related to seeding. Move it to nendo_server
            # or find another way to get rid of it
            try:
                blob.download_to_file(tmpfile)
            except Exception:
                bucket = self.storage_client.bucket(
                    "00110279-7d74-467a-8324-9bafe96878da-nendo",
                )
                blob = bucket.blob(file_name)
                blob.download_to_file(tmpfile)
            temp_file_path = tmpfile.name
        return temp_file_path

    def save_file(self, file_name: str, file_path: str, user_id: str) -> str:
        """Uploads the given file to the GCS bucket."""
        bucket = self._get_user_bucket(user_id=user_id)
        blob = bucket.blob(file_name)
        blob.upload_from_filename(file_path)
        blob.make_public()
        return self.get_file(file_name=file_name, user_id=user_id)

    def save_signal(
        self, file_name: str, signal: np.ndarray, sr: int, user_id: str,
    ) -> str:
        bucket = self._get_user_bucket(user_id=user_id)
        blob = bucket.blob(file_name)

        # Create a temporary file
        with NamedTemporaryFile(suffix=".wav", delete=True) as tmpfile:
            sf.write(tmpfile.name, signal, sr, subtype="PCM_16")

            # NOTE the blob.upload_from_file() call below failed for files that were
            # even slightly larger than the test assets. According to the first SO
            # thread found on google, there is the option to increase the timeout
            # (which I did) or, alternatively, to change the chunksize.
            # Should we keep running into the same problem, it might provide a fix:
            # storage.blob._DEFAULT_CHUNKSIZE = 2097152 # 1024 * 1024 B * 2 = 2 MB
            # storage.blob._MAX_MULTIPART_SIZE = 2097152 # 2 MB

            # Open the temporary file and upload its contents to the blob
            with open(tmpfile.name, "rb") as file_data:
                blob.upload_from_file(file_data, content_type="audio/wav", timeout=300)

        return self.get_file(file_name=file_name, user_id=user_id)

    def save_bytes(self, file_name: str, data: bytes, user_id: str) -> str:
        bucket = self._get_user_bucket(user_id=user_id)
        blob = bucket.blob(file_name)
        serialized_data = pickle.dumps(data)
        blob.upload_from_string(serialized_data)
        return self.get_file(file_name=file_name, user_id=user_id)

    def remove_file(self, file_name: str, user_id: str) -> bool:
        bucket = self._get_user_bucket(user_id=user_id)
        blob = bucket.blob(os.path.basename(file_name))
        blob.delete()
        return True

    def get_file_path(self, src: str, user_id: str) -> str:
        """Returns the path to the GCS bucket, including the user bucket name."""
        return "https://storage.googleapis.com/" + self._get_bucket_name_for_user(
            user_id,
        )

    def get_file_namename(self, src: str, user_id: str) -> str:
        return src[
            len(
                "https://storage.googleapis.com/"
                + self._get_bucket_name_for_user(user_id),
            )
            + 1 :  # + 1 because of slash
        ]

    def get_file(self, file_name: str, user_id: str) -> str:
        return self.get_file_path(src="", user_id=user_id) + "/" + file_name

    def list_files(self, user_id: str) -> List[str]:
        return [
            blob.name
            for blob in self.storage_client.list_blobs(
                self._get_bucket_name_for_user(user_id=user_id),
            )
        ]

    def get_bytes(self, file_name: str, user_id: str) -> Any:
        bucket = self._get_user_bucket(user_id=user_id)
        blob = bucket.blob(file_name)
        serialized_data = blob.download_as_bytes()
        return pickle.loads(serialized_data)

    def get_checksum(self, file_name: str, user_id: str) -> str:
        bucket = self._get_user_bucket(user_id=user_id)
        blob = bucket.blob(file_name)

        blob_bytes = blob.download_as_bytes()

        hash_md5 = hashlib.md5()
        hash_md5.update(blob_bytes)
        return hash_md5.hexdigest()

    def get_location(self) -> str:
        return ResourceLocation.gcs
