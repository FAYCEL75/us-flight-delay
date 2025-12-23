# src/s3_utils.py

import os
from pathlib import Path
import boto3
from botocore.exceptions import NoCredentialsError, ClientError

from src.config import (
    AWS_BUCKET_NAME,
    AWS_REGION,
    AWS_USE_S3,
    AWS_ACCESS_KEY_ENV,
    AWS_SECRET_KEY_ENV,
    AWS_SESSION_TOKEN_ENV,
    AWS_S3_PREFIX,
)


def _get_s3_client():
    access_key = os.getenv(AWS_ACCESS_KEY_ENV)
    secret_key = os.getenv(AWS_SECRET_KEY_ENV)
    session_token = os.getenv(AWS_SESSION_TOKEN_ENV) or None

    if not (access_key and secret_key):
        print("[S3] Credentials missing in env → client without explicit keys.")
        return boto3.client("s3", region_name=AWS_REGION)

    return boto3.client(
        "s3",
        region_name=AWS_REGION,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        aws_session_token=session_token,
    )


def upload_to_s3(file_path: Path, key: str):
    """
    Upload a local file to S3.
    key : chemin complet dans le bucket (ex: 'models/xgb_classifier.joblib')
    """
    if not AWS_USE_S3:
        print("[S3] Skipped upload (AWS_USE_S3 = False)")
        return

    s3 = _get_s3_client()

    try:
        s3.upload_file(str(file_path), AWS_BUCKET_NAME, key)
        print(f"[S3] Uploaded → s3://{AWS_BUCKET_NAME}/{key}")

    except FileNotFoundError:
        print(f"[S3 ERROR] File not found : {file_path}")

    except NoCredentialsError:
        print("[S3 ERROR] AWS credentials missing or invalid.")

    except Exception as e:
        print("[S3 ERROR]", e)


def download_from_s3(key: str, dest_path: Path):
    """
    Télécharger un fichier depuis S3 vers dest_path.
    """
    if not AWS_USE_S3:
        print("[S3] Skipped download (AWS_USE_S3 = False)")
        return

    s3 = _get_s3_client()
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        s3.download_file(AWS_BUCKET_NAME, key, str(dest_path))
        print(f"[S3] Downloaded → {dest_path} (from s3://{AWS_BUCKET_NAME}/{key})")
    except ClientError as e:
        if e.response["Error"]["Code"] == "404":
            print(f"[S3] Object not found: s3://{AWS_BUCKET_NAME}/{key}")
        else:
            print("[S3 ERROR]", e)
    except Exception as e:
        print("[S3 ERROR]", e)