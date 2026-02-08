from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, field_validator


class VideoRequest(BaseModel):
    prompt: str
    negative_prompt: str = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"
    height: int = 480
    width: int = 832
    num_frames: int = 81
    guidance_scale: float = 5.0
    num_inference_steps: int = 50
    seed: int | None = None
    flow_shift: float | None = None
    fps: int = 15
    # For image-to-video (identity preservation)
    reference_image_url: str | None = None

    @field_validator("num_frames")
    @classmethod
    def validate_num_frames(cls, v):
        """Wan 2.1 requires (num_frames - 1) % 4 == 0"""
        if (v - 1) % 4 != 0:
            adjusted = ((v - 1) // 4) * 4 + 1
            if adjusted < 1:
                adjusted = 5
            return adjusted
        return v

    @field_validator("height")
    @classmethod
    def validate_height(cls, v):
        return (v // 16) * 16

    @field_validator("width")
    @classmethod
    def validate_width(cls, v):
        return (v // 16) * 16


class VideoResponse(BaseModel):
    job_id: str | None = None
    video_url: str | None = None
    duration_seconds: float | None = None
    num_frames: int | None = None
    resolution: str | None = None


class Job(BaseModel):
    job_id: str
    webhook_url: str
    job_request_params: List[VideoRequest] = []
    job_s3_folder: str | None = None

    def __init__(self, **data):
        super().__init__(**data)
        if self.job_id:
            self.job_s3_folder = (
                f"videos/{datetime.now().strftime('%Y-%m-%d')}/{self.job_id}/"
            )
