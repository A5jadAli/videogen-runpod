import uuid
import boto3
from datetime import datetime
from botocore.exceptions import NoCredentialsError
import server.server_settings as settings

_client = None


def _get_client():
    """Lazy-initialize boto3 client to avoid crash if credentials are missing at import time."""
    global _client
    if _client is None:
        _client = boto3.client(
            "s3",
            region_name="sfo3",
            endpoint_url=settings.DIGITAL_OCEAN_ENDPOINT_URL,
            aws_access_key_id=settings.DIGITAL_OCEAN_BUCKET_ACCESS_KEY,
            aws_secret_access_key=settings.DIGITAL_OCEAN_BUCKET_SECRET_KEY,
        )
    return _client


def upload(path, object_name=None):
    """Upload a video file to Digital Ocean Spaces."""
    if object_name is None:
        object_name = f"videos/{datetime.now().strftime('%Y-%m-%d')}/{str(uuid.uuid4())}/"

    print("###########################  UPLOADING VIDEO ############################")
    file_name = path.split("/")[-1]
    upload_path = f"{object_name}{file_name}"
    bucket_name = settings.DIGITAL_OCEAN_BUCKET_NAME

    content_type = "video/mp4"
    if path.endswith(".webm"):
        content_type = "video/webm"

    try:
        client = _get_client()
        with open(path, "rb") as f:
            client.put_object(
                Bucket=bucket_name,
                Key=upload_path,
                Body=f,
                ACL="public-read",
                ContentType=content_type,
            )
        print(
            f"File '{path}' uploaded to bucket '{bucket_name}' as '{upload_path}'."
        )
        return f"{settings.DIGITAL_OCEAN_BUCKET_URL}{upload_path}"
    except FileNotFoundError:
        print("The file was not found.")
    except NoCredentialsError:
        print("Credentials not available.")
    except Exception as e:
        print(f"An error occurred: {e}")
    return None
