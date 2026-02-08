import os
from server.request_queue import Job, VideoRequest, VideoResponse
from server.storage_utils import upload
from server.utils import webhook_response, send_progress_webhook, download_reference_image


async def process_job(job: Job, video_service):
    """
    Process a video generation job using Wan 2.2 A14B MoE pipeline.

    Handles:
    - Text-to-Video (no reference image)
    - Image-to-Video with identity preservation (reference image provided)
    - Camera motion control via prompt enhancement
    - Quality presets (draft/standard/high/ultra)
    """

    def progress_callback(progress_percent, status):
        send_progress_webhook(
            webhook_url=job.webhook_url,
            job_id=job.job_id,
            progress=progress_percent,
            status=status,
        )

    for idx, param in enumerate(job.job_request_params):
        print(f"\n{'='*60}")
        print(f"Processing video request {idx + 1}/{len(job.job_request_params)} for job {job.job_id}")
        print(f"  Mode: {'I2V' if param.reference_image_url else 'T2V'}")
        print(f"  Resolution: {param.width}x{param.height}")
        print(f"  Frames: {param.num_frames} @ {param.fps}fps ({param.num_frames/param.fps:.1f}s)")
        if param.camera_motion:
            print(f"  Camera: {param.camera_motion}")
        if param.quality_preset:
            print(f"  Quality: {param.quality_preset}")
        print(f"{'='*60}")

        # Download reference image if provided (for I2V / identity preservation)
        reference_image_path = None
        if param.reference_image_url:
            reference_image_path = download_reference_image(param.reference_image_url)
            if not reference_image_path:
                print("Failed to download reference image")
                webhook_response(
                    job.webhook_url,
                    False,
                    500,
                    "Failed to download reference image",
                    {"job_id": job.job_id},
                )
                continue

        # Generate video with Wan 2.2
        video_path = video_service.generate_video(
            prompt=param.prompt,
            negative_prompt=param.negative_prompt,
            height=param.height,
            width=param.width,
            num_frames=param.num_frames,
            guidance_scale=param.guidance_scale,
            num_inference_steps=param.num_inference_steps,
            seed=param.seed,
            flow_shift=param.flow_shift,
            fps=param.fps,
            reference_image_path=reference_image_path,
            camera_motion=param.camera_motion,
            enhance_prompt=param.enhance_prompt,
            quality_preset=param.quality_preset,
            progress_callback=progress_callback,
        )

        print(f"Video path: {video_path}")

        if not video_path:
            print("Video generation failed - no video path returned")
            webhook_response(
                job.webhook_url,
                False,
                500,
                "Video generation failed",
                {"job_id": job.job_id},
            )
            continue

        # Send uploading status
        send_progress_webhook(
            webhook_url=job.webhook_url,
            job_id=job.job_id,
            progress=100,
            status="uploading",
        )

        # Upload and send final webhook
        process_response(job, param, video_path)

        # Clean up reference image
        if reference_image_path and os.path.exists(reference_image_path):
            try:
                os.remove(reference_image_path)
            except Exception:
                pass


def process_response(job: Job, request: VideoRequest, video_path: str):
    """Upload video and send final webhook."""
    try:
        cloud_storage_path = upload(path=str(video_path), object_name=job.job_s3_folder)
        print(f"Cloud storage path: {cloud_storage_path}")

        if not cloud_storage_path:
            print("Upload failed - no cloud storage path returned")
            webhook_response(
                job.webhook_url,
                False,
                500,
                "Video upload failed",
                {"job_id": job.job_id},
            )
            return

        duration_seconds = request.num_frames / request.fps

        response = VideoResponse(
            job_id=job.job_id,
            video_url=cloud_storage_path,
            duration_seconds=round(duration_seconds, 2),
            num_frames=request.num_frames,
            resolution=f"{request.width}x{request.height}",
            model_version="wan2.2-i2v-a14b",
        )

        webhook_response(
            job.webhook_url,
            True,
            200,
            "Video Generated!",
            response.dict(),
        )
        print(f"Final webhook sent successfully for job {job.job_id}")

    except Exception as e:
        print(f"Error in process_response: {e}")
        webhook_response(
            job.webhook_url,
            False,
            500,
            f"Error processing response: {str(e)}",
            {"job_id": job.job_id},
        )

    finally:
        try:
            if video_path and os.path.exists(video_path):
                os.remove(video_path)
                print(f"Deleted local video: {video_path}")
        except Exception as e:
            print(f"Error deleting video: {e}")
