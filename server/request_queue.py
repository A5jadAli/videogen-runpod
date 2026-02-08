from datetime import datetime
from typing import List, Optional, Literal
from pydantic import BaseModel, field_validator


class VideoRequest(BaseModel):
    """
    Video generation request parameters for Wan 2.2.

    Supports three generation modes:
    1. Text-to-Video (T2V): Just provide a prompt
    2. Image-to-Video (I2V): Provide prompt + reference_image_url
    3. Controlled I2V: Provide prompt + reference_image_url + control options

    Camera motion can be controlled via:
    - camera_motion param (predefined camera movements)
    - Embedding camera instructions directly in the prompt

    Wan 2.2 prompt structure for best results:
      "Subject (description) + Scene (description) + Motion (description) + Aesthetic Control + Style"
    """

    prompt: str

    negative_prompt: str = (
        "Bright tones, overexposed, static, blurred details, subtitles, style, works, "
        "paintings, images, static, overall gray, worst quality, low quality, JPEG compression "
        "residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, "
        "deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, "
        "three legs, many people in the background, walking backwards, blurring, mutation, "
        "deformation, distortion, dark and solid, comics"
    )

    # Resolution — Wan 2.2 supports up to 720p natively
    height: int = 480
    width: int = 832

    # Frame count — Wan 2.2 requires (num_frames - 1) % 4 == 0
    num_frames: int = 81

    # Generation params
    guidance_scale: float = 5.0
    num_inference_steps: int = 50
    seed: int | None = None
    flow_shift: float | None = None
    fps: int = 16

    # ============================================================
    # Image-to-Video (identity preservation)
    # ============================================================
    reference_image_url: str | None = None

    # ============================================================
    # Camera motion control (Wan 2.2 feature)
    # Appended to prompt automatically for cinematic control.
    # Options: pan_left, pan_right, pan_up, pan_down,
    #          zoom_in, zoom_out, static, orbit_left, orbit_right,
    #          dolly_in, dolly_out, crane_up, crane_down, handheld
    # ============================================================
    camera_motion: str | None = None

    # ============================================================
    # Prompt enhancement — auto-enhance prompt for Wan 2.2 style
    # Adds cinematic descriptors, aesthetic tags, motion qualifiers
    # ============================================================
    enhance_prompt: bool = True

    # ============================================================
    # Quality preset — quick override for common configurations
    # "draft": 30 steps, lower guidance — fast preview
    # "standard": 50 steps — balanced quality/speed
    # "high": 80 steps, higher guidance — best quality
    # "ultra": 100 steps — maximum quality, slowest
    # ============================================================
    quality_preset: Literal["draft", "standard", "high", "ultra"] | None = None

    @field_validator("num_frames")
    @classmethod
    def validate_num_frames(cls, v):
        """Wan 2.2 requires (num_frames - 1) % 4 == 0"""
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


# ============================================================
# Camera motion → prompt suffix mapping for Wan 2.2
# These are optimized prompt suffixes based on Wan 2.2 docs.
# ============================================================
CAMERA_MOTION_PROMPTS = {
    "pan_left": "smooth camera pan left, lateral tracking shot",
    "pan_right": "smooth camera pan right, lateral tracking shot",
    "pan_up": "smooth camera tilt up, vertical pan upward",
    "pan_down": "smooth camera tilt down, vertical pan downward",
    "zoom_in": "smooth dolly in, camera slowly zooming in, push in shot",
    "zoom_out": "smooth dolly out, camera slowly zooming out, pull back shot",
    "static": "static shot, fixed camera, locked off tripod shot, no camera movement",
    "orbit_left": "smooth orbital camera movement to the left, rotating around subject",
    "orbit_right": "smooth orbital camera movement to the right, rotating around subject",
    "dolly_in": "dolly in, camera moves forward toward subject",
    "dolly_out": "dolly out, camera moves backward away from subject",
    "crane_up": "crane shot moving upward, ascending camera",
    "crane_down": "crane shot moving downward, descending camera",
    "handheld": "handheld camera, slight natural shake, documentary style",
}

# ============================================================
# Quality preset configurations
# ============================================================
QUALITY_PRESETS = {
    "draft": {
        "num_inference_steps": 30,
        "guidance_scale": 4.0,
    },
    "standard": {
        "num_inference_steps": 50,
        "guidance_scale": 5.0,
    },
    "high": {
        "num_inference_steps": 80,
        "guidance_scale": 5.5,
    },
    "ultra": {
        "num_inference_steps": 100,
        "guidance_scale": 6.0,
    },
}


class VideoResponse(BaseModel):
    job_id: str | None = None
    video_url: str | None = None
    duration_seconds: float | None = None
    num_frames: int | None = None
    resolution: str | None = None
    model_version: str = "wan2.2-i2v-a14b"


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
