import os
import uuid
import time
import requests
from datetime import datetime


def webhook_response(webhook_url, status, code, message, data=None):
    """Final result webhook — same format as image generation service."""
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


def download_reference_image(image_url, save_dir="/tmp/ref_images"):
    """
    Download a reference image for image-to-video generation.

    Improvements over v1:
    - Unique filenames to avoid collisions with concurrent requests
    - Retry logic for transient failures
    - Better content-type detection
    - Size validation

    Returns the local file path or None on failure.
    """
    MAX_RETRIES = 3
    MAX_SIZE_MB = 50

    try:
        os.makedirs(save_dir, exist_ok=True)

        # Determine extension from URL
        file_ext = image_url.split(".")[-1].split("?")[0].lower()
        if file_ext not in ["jpg", "jpeg", "png", "webp", "bmp", "tiff"]:
            file_ext = "png"

        # Unique filename to avoid collisions
        local_path = os.path.join(save_dir, f"ref_{uuid.uuid4().hex[:12]}.{file_ext}")

        # Download with retry
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = requests.get(
                    image_url,
                    timeout=60,
                    stream=True,
                    headers={"User-Agent": "VideoGen-RunPod/2.0"},
                )
                response.raise_for_status()

                # Check content length if available
                content_length = response.headers.get("Content-Length")
                if content_length and int(content_length) > MAX_SIZE_MB * 1024 * 1024:
                    print(f"Reference image too large: {int(content_length) / 1e6:.1f}MB (max {MAX_SIZE_MB}MB)")
                    return None

                with open(local_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)

                # Validate it's actually an image
                from PIL import Image
                img = Image.open(local_path)
                img.verify()

                print(f"Reference image downloaded to {local_path} (attempt {attempt})")
                return local_path

            except requests.RequestException as e:
                if attempt < MAX_RETRIES:
                    wait_time = attempt * 2
                    print(f"Download attempt {attempt} failed ({e}), retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    print(f"All {MAX_RETRIES} download attempts failed: {e}")
                    return None

    except Exception as e:
        print(f"Error downloading reference image: {e}")
        return None


def delete_old_files(directory_path):
    """
    Delete video and temp files older than 1 hour.
    """
    extensions = [".mp4", ".avi", ".mov", ".webm", ".tmp", ".png", ".jpg", ".jpeg"]
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

        if deleted_count > 0:
            print(f"Deleted {deleted_count} file(s) that were at least 1 hour old.")
        return deleted_count, deleted_files

    except Exception as e:
        print(f"Error cleaning up files: {e}")
        return 0, []
