from datetime import datetime
from typing import List, Optional, Literal
from pydantic import BaseModel, field_validator


class VideoRequest(BaseModel):
    """
    Video generation request parameters for Wan 2.2.

    Supports:
    1. Image-to-Video (I2V): Provide prompt + reference_image_url (primary mode)
    2. Text-to-Video (T2V): Just provide a prompt (requires T2V pipeline enabled)

    Post-processing pipeline (enabled by default):
    - Face restoration (CodeFormer) → Frame interpolation (RIFE) → Upscaling (Real-ESRGAN) → FFmpeg finishing
    """

    prompt: str

    # Research-backed negative prompt: includes temporal stability terms
    # (flicker, exposure flicker, frame hopping) and face morphing prevention
    negative_prompt: str = (
        "Bright tones, overexposed, static, blurred details, subtitles, style, works, "
        "paintings, images, static, overall gray, worst quality, low quality, JPEG compression "
        "residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, "
        "deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, "
        "three legs, many people in the background, walking backwards, blurring, mutation, "
        "deformation, distortion, dark and solid, comics, "
        "flicker, exposure flicker, frame hopping, face morphing, feature drift"
    )

    # Resolution — Wan 2.2 supports up to 720p natively
    height: int = 480
    width: int = 832

    # Frame count — Wan 2.2 requires (num_frames - 1) % 4 == 0
    num_frames: int = 81

    # Generation params — optimized for Wan 2.2 I2V
    # guidance_scale=3.5 is official recommendation (range: 3.0-5.0)
    guidance_scale: float = 3.5
    num_inference_steps: int = 40  # Official recommendation; 30 minimum
    seed: int | None = None
    flow_shift: float | None = None  # Auto: 3.0 for 480p, 5.0 for 720p
    fps: int = 16

    # ============================================================
    # Image-to-Video
    # ============================================================
    reference_image_url: str | None = None

    # ============================================================
    # Camera motion control
    # ============================================================
    camera_motion: str | None = None

    # ============================================================
    # Prompt enhancement
    # ============================================================
    enhance_prompt: Literal["none", "light", "full"] = "light"

    # ============================================================
    # Quality preset
    # ============================================================
    quality_preset: Literal["draft", "standard", "high", "ultra"] | None = None

    # ============================================================
    # Post-Processing Pipeline Controls
    # ============================================================
    enable_post_processing: bool = True
    enable_face_restore: bool = True
    enable_interpolation: bool = True
    enable_upscale: bool = True
    enable_ffmpeg_enhance: bool = True

    upscale_factor: float = 2.0
    target_fps: int = 24
    face_fidelity: float = 0.6  # CodeFormer w: 0.5-0.7 optimal for AI video

    @field_validator("num_frames")
    @classmethod
    def validate_num_frames(cls, v):
        """Wan 2.2 requires (num_frames - 1) % 4 == 0. Capped at ~10s at 16fps."""
        v = max(5, min(v, 161))
        if (v - 1) % 4 != 0:
            adjusted = ((v - 1) // 4) * 4 + 1
            if adjusted < 5:
                adjusted = 5
            return adjusted
        return v

    @field_validator("height")
    @classmethod
    def validate_height(cls, v):
        """Clamp to 256-720 range, align to 16px."""
        v = max(256, min(v, 720))
        return (v // 16) * 16

    @field_validator("width")
    @classmethod
    def validate_width(cls, v):
        """Clamp to 256-1280 range, align to 16px."""
        v = max(256, min(v, 1280))
        return (v // 16) * 16

    @field_validator("guidance_scale")
    @classmethod
    def validate_guidance_scale(cls, v):
        """Clamp guidance_scale to safe I2V range. >5.0 causes artifacts."""
        return max(1.0, min(v, 7.0))

    @field_validator("num_inference_steps")
    @classmethod
    def validate_num_inference_steps(cls, v):
        """Minimum 20 steps for any usable quality."""
        return max(20, min(v, 100))

    @field_validator("upscale_factor")
    @classmethod
    def validate_upscale_factor(cls, v):
        return max(1.0, min(v, 4.0))

    @field_validator("target_fps")
    @classmethod
    def validate_target_fps(cls, v):
        return max(16, min(v, 60))

    @field_validator("face_fidelity")
    @classmethod
    def validate_face_fidelity(cls, v):
        return max(0.0, min(v, 1.0))


# ============================================================
# Camera motion → prompt suffix mapping for Wan 2.2
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
# Quality preset configurations — Wan 2.2 I2V optimal ranges
# ============================================================
QUALITY_PRESETS = {
    "draft": {
        "num_inference_steps": 25,
        "guidance_scale": 3.5,
    },
    "standard": {
        "num_inference_steps": 40,
        "guidance_scale": 3.5,
    },
    "high": {
        "num_inference_steps": 50,
        "guidance_scale": 4.0,
    },
    "ultra": {
        "num_inference_steps": 80,
        "guidance_scale": 4.5,
    },
}


class VideoResponse(BaseModel):
    job_id: str | None = None
    video_url: str | None = None
    duration_seconds: float | None = None
    num_frames: int | None = None
    resolution: str | None = None
    fps: int | None = None
    post_processed: bool = False
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
