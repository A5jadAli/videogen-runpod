import os
import uuid
import time
import threading
import requests
from datetime import datetime


# ============================================================
# Webhook utilities — uses session pooling + background thread
#
# Research: progress webhooks can block the event loop for 10s+ each.
# Solution: use a persistent Session (connection pooling) and send
# progress updates from a background thread. Final webhooks are
# sent synchronously (must confirm delivery).
# ============================================================

_webhook_session = None
_session_lock = threading.Lock()


def _get_session():
    """Lazy-initialize a requests.Session with connection pooling."""
    global _webhook_session
    if _webhook_session is None:
        with _session_lock:
            if _webhook_session is None:
                s = requests.Session()
                adapter = requests.adapters.HTTPAdapter(
                    pool_connections=2,
                    pool_maxsize=5,
                    max_retries=1,
                )
                s.mount("http://", adapter)
                s.mount("https://", adapter)
                _webhook_session = s
    return _webhook_session


def webhook_response(webhook_url, status, code, message, data=None):
    """Final result webhook — synchronous to confirm delivery."""
    response_data = {
        "status": status,
        "code": code,
        "message": message,
        "data": data,
    }
    if webhook_url and "http" in webhook_url:
        try:
            session = _get_session()
            resp = session.post(webhook_url, json=response_data, timeout=15)
            print(f"Final webhook sent: {message} (HTTP {resp.status_code})")
            if resp.status_code >= 400:
                print(f"  WARNING: Webhook returned {resp.status_code}: {resp.text[:200]}")
        except Exception as e:
            print(f"CRITICAL: Error sending final webhook: {e}")


def send_progress_webhook(webhook_url, job_id, progress, status):
    """
    Send progress update in a background thread.

    Non-blocking: webhook failures don't delay generation.
    Uses session pooling to avoid DNS resolver bugs (SDK #422).
    """
    if not webhook_url or "http" not in webhook_url:
        return

    def _send():
        progress_data = {
            "type": "progress",
            "job_id": job_id,
            "progress": progress,
            "status": status,
        }
        try:
            session = _get_session()
            session.post(webhook_url, json=progress_data, timeout=5)
        except Exception as e:
            # Progress webhooks are best-effort — don't crash on failure
            print(f"Progress webhook failed (non-fatal): {e}")

    t = threading.Thread(target=_send, daemon=True)
    t.start()
    # Print locally regardless of send success
    print(f"Progress: {progress}% - {status}")


def download_reference_image(image_url, save_dir="/tmp/ref_images"):
    """
    Download a reference image for image-to-video generation.

    Features:
    - Unique filenames to avoid collisions
    - Retry logic for transient failures
    - Size validation (50MB max)
    - Image verification via PIL
    """
    MAX_RETRIES = 3
    MAX_SIZE_MB = 50

    try:
        os.makedirs(save_dir, exist_ok=True)

        # Determine extension from URL
        file_ext = image_url.split(".")[-1].split("?")[0].lower()
        if file_ext not in ["jpg", "jpeg", "png", "webp", "bmp", "tiff"]:
            file_ext = "png"

        local_path = os.path.join(save_dir, f"ref_{uuid.uuid4().hex[:12]}.{file_ext}")

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                session = _get_session()
                response = session.get(
                    image_url,
                    timeout=60,
                    stream=True,
                    headers={"User-Agent": "VideoGen-RunPod/2.0"},
                )
                response.raise_for_status()

                content_length = response.headers.get("Content-Length")
                if content_length and int(content_length) > MAX_SIZE_MB * 1024 * 1024:
                    print(f"Reference image too large: {int(content_length) / 1e6:.1f}MB")
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
    """Delete video and temp files older than 1 hour."""
    extensions = [".mp4", ".avi", ".mov", ".webm", ".tmp", ".png", ".jpg", ".jpeg"]
    current_time = time.time()
    one_hour_ago = current_time - (60 * 60)

    deleted_count = 0

    try:
        if not os.path.exists(directory_path):
            return 0, []

        deleted_files = []
        for filename in os.listdir(directory_path):
            file_path = os.path.join(directory_path, filename)
            if os.path.isfile(file_path) and any(
                filename.lower().endswith(ext) for ext in extensions
            ):
                # Use mtime (modification time) — more reliable than ctime on Linux
                file_mod_time = os.path.getmtime(file_path)
                if file_mod_time <= one_hour_ago:
                    os.remove(file_path)
                    deleted_count += 1
                    deleted_files.append(filename)

        if deleted_count > 0:
            print(f"Deleted {deleted_count} old file(s).")
        return deleted_count, deleted_files

    except Exception as e:
        print(f"Error cleaning up files: {e}")
        return 0, []
