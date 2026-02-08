import os
import time
import requests
from datetime import datetime


def webhook_response(webhook_url, status, code, message, data=None):
    """Final result webhook - same format as image generation service."""
    response_data = {
        "status": status,
        "code": code,
        "message": message,
        "data": data,
    }
    if webhook_url and "http" in webhook_url:
        try:
            requests.post(webhook_url, json=response_data, timeout=10)
            print(f"Final webhook sent: {message}")
        except Exception as e:
            print(f"Error sending final webhook: {e}")


def send_progress_webhook(webhook_url, job_id, progress, status):
    """
    Send progress update to the webhook URL.
    Same format as image generation service.
    """
    if not webhook_url or "http" not in webhook_url:
        return

    progress_data = {
        "type": "progress",
        "job_id": job_id,
        "progress": progress,
        "status": status,
    }

    try:
        requests.post(webhook_url, json=progress_data, timeout=5)
        print(f"Progress webhook sent: {progress}% - {status}")
    except Exception as e:
        print(f"Error sending progress webhook: {e}")


def download_reference_image(image_url, save_dir="/tmp"):
    """
    Download a reference image for image-to-video generation.
    Returns the local file path.
    """
    try:
        os.makedirs(save_dir, exist_ok=True)
        file_ext = image_url.split(".")[-1].split("?")[0]
        if file_ext not in ["jpg", "jpeg", "png", "webp"]:
            file_ext = "png"
        local_path = os.path.join(save_dir, f"ref_image.{file_ext}")
        response = requests.get(image_url, timeout=30)
        response.raise_for_status()
        with open(local_path, "wb") as f:
            f.write(response.content)
        print(f"Reference image downloaded to {local_path}")
        return local_path
    except Exception as e:
        print(f"Error downloading reference image: {e}")
        return None


def delete_old_files(directory_path):
    """
    Delete video and temp files older than 1 hour.
    """
    extensions = [".mp4", ".avi", ".mov", ".webm", ".tmp", ".png", ".jpg"]
    current_time = time.time()
    one_hour_ago = current_time - (60 * 60)

    deleted_count = 0
    deleted_files = []

    try:
        if not os.path.exists(directory_path):
            return 0, []

        for filename in os.listdir(directory_path):
            file_path = os.path.join(directory_path, filename)
            if os.path.isfile(file_path) and any(
                filename.lower().endswith(ext) for ext in extensions
            ):
                file_creation_time = os.path.getctime(file_path)
                if file_creation_time <= one_hour_ago:
                    os.remove(file_path)
                    deleted_count += 1
                    deleted_files.append(filename)
                    print(
                        f"Deleted: {filename} (Created: {datetime.fromtimestamp(file_creation_time)})"
                    )

        print(f"Deleted {deleted_count} file(s) that were at least 1 hour old.")
        return deleted_count, deleted_files

    except Exception as e:
        print(f"Error: {e}")
        return 0, []
