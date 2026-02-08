import os
import uuid
import gc
import torch
import numpy as np
from PIL import Image
from server import server_settings
from server.request_queue import CAMERA_MOTION_PROMPTS, QUALITY_PRESETS


class VideoService:
    """
    Wan 2.2 A14B MoE Video Generation Service.

    Key upgrades over Wan 2.1:
    - MoE architecture: Dual-expert system (high-noise for layout, low-noise for detail)
    - 65% more training images, 83% more training videos
    - Cinematic aesthetic control: lighting, color, composition
    - Complex motion: gestures, athletics, facial expressions
    - Precise semantic compliance: multi-object, spatial accuracy

    Models loaded at startup, reused across all requests.
    No safety filters or content restrictions.
    """

    def __init__(self):
        self.t2v_pipe = None
        self.i2v_pipe = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.bfloat16
        self._sage_attention_available = False
        self._check_sage_attention()
        self._load_pipelines()

    def _check_sage_attention(self):
        """Check if SageAttention is available for faster inference."""
        if not server_settings.ENABLE_SAGE_ATTENTION:
            print("SageAttention disabled via config.")
            return

        try:
            import sageattention
            self._sage_attention_available = True
            print("SageAttention available — will enable for ~40-50% faster inference.")
        except ImportError:
            print("SageAttention not installed — using default attention (slower).")
            self._sage_attention_available = False

    def _load_pipelines(self):
        """Load Wan 2.2 I2V pipeline. T2V disabled by default (uncomment to enable)."""

        # ----------------------------------------------------------
        # Uncomment to enable Text-to-Video pipeline
        # ----------------------------------------------------------
        # print("=" * 60)
        # print("Loading Wan 2.2 Text-to-Video A14B MoE pipeline...")
        # print("=" * 60)
        # self._load_t2v()

        print("=" * 60)
        print("Loading Wan 2.2 Image-to-Video A14B MoE pipeline...")
        print("=" * 60)
        self._load_i2v()

        print("=" * 60)
        print("All Wan 2.2 pipelines loaded successfully.")
        print("=" * 60)

    def _load_t2v(self):
        """Load the Wan 2.2 Text-to-Video MoE pipeline."""
        from diffusers import AutoencoderKLWan, WanPipeline

        model_id = server_settings.T2V_MODEL_ID
        print(f"  Model: {model_id}")

        vae = AutoencoderKLWan.from_pretrained(
            model_id,
            subfolder="vae",
            torch_dtype=torch.float32,
            cache_dir=server_settings.MODEL_CACHE_DIR,
        )

        self.t2v_pipe = WanPipeline.from_pretrained(
            model_id,
            vae=vae,
            torch_dtype=self.dtype,
            cache_dir=server_settings.MODEL_CACHE_DIR,
        )

        self._disable_safety(self.t2v_pipe)
        self._apply_optimizations(self.t2v_pipe)
        print("  T2V pipeline loaded.")

    def _load_i2v(self):
        """Load the Wan 2.2 Image-to-Video MoE pipeline."""
        from diffusers import AutoencoderKLWan, WanImageToVideoPipeline

        model_id = server_settings.I2V_MODEL_ID
        print(f"  Model: {model_id}")

        # VAE in float32 for quality — critical for reducing artifacts
        vae = AutoencoderKLWan.from_pretrained(
            model_id,
            subfolder="vae",
            torch_dtype=torch.float32,
            cache_dir=server_settings.MODEL_CACHE_DIR,
        )

        self.i2v_pipe = WanImageToVideoPipeline.from_pretrained(
            model_id,
            vae=vae,
            torch_dtype=self.dtype,
            cache_dir=server_settings.MODEL_CACHE_DIR,
        )

        self._disable_safety(self.i2v_pipe)
        self._apply_optimizations(self.i2v_pipe)
        print("  I2V pipeline loaded.")

    def _disable_safety(self, pipe):
        """Remove all safety checkers and content filters."""
        for attr in ["safety_checker", "feature_extractor", "watermarker"]:
            if hasattr(pipe, attr):
                setattr(pipe, attr, None)

    def _apply_optimizations(self, pipe):
        """Apply memory and speed optimizations based on environment config."""

        # Memory strategy
        if server_settings.ENABLE_SEQUENTIAL_CPU_OFFLOAD:
            print("  -> Enabling sequential CPU offload (lowest VRAM, slower)")
            pipe.enable_sequential_cpu_offload()
        elif server_settings.ENABLE_CPU_OFFLOAD:
            print("  -> Enabling model CPU offload")
            pipe.enable_model_cpu_offload()
        else:
            print("  -> Moving pipeline to CUDA (full VRAM mode)")
            pipe.to(self.device)

        # VAE tiling — reduces VRAM for high-res decoding
        if server_settings.ENABLE_VAE_TILING:
            print("  -> Enabling VAE tiling")
            pipe.enable_vae_tiling()

        # SageAttention — major speed boost
        if self._sage_attention_available:
            try:
                from sageattention import sageattn
                # Replace attention processors with SageAttention
                # This patches F.scaled_dot_product_attention globally
                import torch.nn.functional as F
                F._original_sdpa = F.scaled_dot_product_attention
                F.scaled_dot_product_attention = sageattn
                print("  -> SageAttention enabled (patched SDPA)")
            except Exception as e:
                print(f"  -> SageAttention patch failed: {e}")

        # torch.compile — JIT compilation for speed (slow first run)
        if server_settings.ENABLE_TORCH_COMPILE:
            try:
                pipe.transformer = torch.compile(
                    pipe.transformer, mode="reduce-overhead", fullgraph=True
                )
                print("  -> torch.compile enabled on transformer")
            except Exception as e:
                print(f"  -> torch.compile failed: {e}")

    def _determine_flow_shift(self, height, width, flow_shift=None):
        """
        Auto-determine flow_shift based on resolution.
        Wan 2.2 uses similar flow_shift ranges as 2.1:
        - 480p (low res): 3.0-5.0
        - 720p (high res): 7.0-9.0
        Higher flow_shift = more dynamic/diverse motion but can be less stable.
        """
        if flow_shift is not None:
            return flow_shift
        total_pixels = height * width
        if total_pixels <= 832 * 480:
            return 3.0
        elif total_pixels <= 1280 * 720:
            return 8.0
        else:
            return 9.0

    def _enhance_prompt(self, prompt, camera_motion=None):
        """
        Enhance prompt for Wan 2.2's improved semantic understanding.

        Wan 2.2 prompt structure (from official docs):
        Subject (description) + Scene (description) + Motion (description)
        + Aesthetic Control + Style

        Adds cinematic descriptors for better realism:
        - Temporal stability keywords
        - Lighting/aesthetic qualifiers
        - Motion smoothness descriptors
        """
        enhancements = []

        # Add camera motion to prompt
        if camera_motion and camera_motion in CAMERA_MOTION_PROMPTS:
            enhancements.append(CAMERA_MOTION_PROMPTS[camera_motion])

        # Add cinematic quality enhancers for realism
        quality_suffix = (
            "cinematic lighting, natural color grading, film grain, "
            "shallow depth of field, high detail, photorealistic, "
            "temporally consistent motion, smooth natural movement, "
            "8K quality, professional cinematography"
        )
        enhancements.append(quality_suffix)

        if enhancements:
            enhanced = prompt.rstrip(". ") + ". " + ", ".join(enhancements)
        else:
            enhanced = prompt

        return enhanced

    def _prepare_reference_image(self, image_path, width, height):
        """
        Load and prepare reference image for I2V.

        Uses high-quality LANCZOS resampling and ensures proper
        color space for best identity preservation.
        """
        img = Image.open(image_path).convert("RGB")

        # Calculate aspect-preserving resize then center crop
        img_ratio = img.width / img.height
        target_ratio = width / height

        if img_ratio > target_ratio:
            # Image is wider — fit height, crop width
            new_height = height
            new_width = int(height * img_ratio)
        else:
            # Image is taller — fit width, crop height
            new_width = width
            new_height = int(width / img_ratio)

        img = img.resize((new_width, new_height), Image.LANCZOS)

        # Center crop to exact dimensions
        left = (new_width - width) // 2
        top = (new_height - height) // 2
        img = img.crop((left, top, left + width, top + height))

        return img

    def _apply_quality_preset(self, params):
        """Apply quality preset overrides if specified."""
        preset_name = params.get("quality_preset")
        if not preset_name or preset_name not in QUALITY_PRESETS:
            return params

        preset = QUALITY_PRESETS[preset_name]
        # Preset values override defaults but not explicitly set values
        for key, value in preset.items():
            if key not in params or params[key] is None:
                params[key] = value

        print(f"  Applied quality preset: {preset_name} -> {preset}")
        return params

    def _export_video_high_quality(self, frames, output_path, fps=16):
        """
        Export frames to MP4 with high-quality encoding.
        Uses H.264 with CRF 18 for near-lossless quality.
        """
        import imageio

        # Try ffmpeg-based writer for best quality
        try:
            writer = imageio.get_writer(
                output_path,
                fps=fps,
                codec="libx264",
                quality=None,  # Use output_params instead
                output_params=[
                    "-crf", "18",           # High quality (lower = better, 0 = lossless)
                    "-preset", "slow",       # Better compression
                    "-pix_fmt", "yuv420p",   # Compatibility
                    "-movflags", "+faststart",  # Web streaming
                ],
            )
            for frame in frames:
                if isinstance(frame, Image.Image):
                    frame = np.array(frame)
                writer.append_data(frame)
            writer.close()
            return True
        except Exception as e:
            print(f"  High-quality export failed ({e}), falling back to default")

        # Fallback: basic export
        from diffusers.utils import export_to_video
        export_to_video(frames, output_path, fps=fps)
        return True

    def generate_video(
        self,
        prompt,
        negative_prompt="",
        height=480,
        width=832,
        num_frames=81,
        guidance_scale=5.0,
        num_inference_steps=50,
        seed=None,
        flow_shift=None,
        fps=16,
        reference_image_path=None,
        camera_motion=None,
        enhance_prompt=True,
        quality_preset=None,
        progress_callback=None,
    ):
        """
        Generate video using Wan 2.2 A14B MoE model.

        If reference_image_path is provided, uses I2V pipeline (identity preservation).
        Otherwise uses T2V pipeline.

        New Wan 2.2 features:
        - camera_motion: Predefined camera movement (pan, zoom, orbit, etc.)
        - enhance_prompt: Auto-enhance prompt for Wan 2.2 cinematic quality
        - quality_preset: Quick quality override (draft/standard/high/ultra)

        Returns:
            str: Absolute path to generated .mp4 file, or None on failure.
        """
        try:
            # Apply quality preset if specified
            if quality_preset and quality_preset in QUALITY_PRESETS:
                preset = QUALITY_PRESETS[quality_preset]
                num_inference_steps = preset.get("num_inference_steps", num_inference_steps)
                guidance_scale = preset.get("guidance_scale", guidance_scale)
                print(f"  Quality preset '{quality_preset}': steps={num_inference_steps}, guidance={guidance_scale}")

            # Decide which pipeline to use
            use_i2v = reference_image_path is not None and os.path.exists(reference_image_path)

            if not use_i2v and self.t2v_pipe is None:
                print("Error: T2V pipeline is disabled. Provide a reference_image_url for I2V.")
                return None

            pipe = self.i2v_pipe if use_i2v else self.t2v_pipe
            mode = "I2V" if use_i2v else "T2V"

            # Enhance prompt for Wan 2.2
            original_prompt = prompt
            if enhance_prompt:
                prompt = self._enhance_prompt(prompt, camera_motion)
                if prompt != original_prompt:
                    print(f"  [{mode}] Enhanced prompt: {prompt[:150]}...")

            print(
                f"  [{mode}] Generating: {width}x{height}, {num_frames} frames, "
                f"{num_inference_steps} steps, guidance={guidance_scale}, fps={fps}"
            )

            # Set up generator for reproducibility
            generator = None
            if seed is not None:
                generator = torch.Generator(device="cpu").manual_seed(seed)
                print(f"  [{mode}] Seed: {seed}")

            # Step callback for progress reporting
            total_steps = num_inference_steps

            def diffusion_step_callback(pipe_ref, step_index, timestep, callback_kwargs):
                progress_percent = int(((step_index + 1) / total_steps) * 90)
                if progress_callback:
                    progress_callback(progress_percent, "generating")
                return callback_kwargs

            # Build pipeline kwargs
            pipe_kwargs = dict(
                prompt=prompt,
                negative_prompt=negative_prompt if negative_prompt else None,
                height=height,
                width=width,
                num_frames=num_frames,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                generator=generator,
                callback_on_step_end=diffusion_step_callback,
            )

            # For I2V: load and prepare reference image
            if use_i2v:
                ref_image = self._prepare_reference_image(reference_image_path, width, height)
                pipe_kwargs["image"] = ref_image
                print(f"  [{mode}] Reference image loaded: {reference_image_path} -> {width}x{height}")

            # Run inference with autocast for optimal mixed precision
            with torch.inference_mode():
                output = pipe(**pipe_kwargs).frames[0]

            if progress_callback:
                progress_callback(95, "exporting")

            # Export to high-quality MP4
            os.makedirs(server_settings.TEMP_DIR, exist_ok=True)
            video_filename = f"{uuid.uuid4()}.mp4"
            video_path = os.path.join(server_settings.TEMP_DIR, video_filename)

            self._export_video_high_quality(output, video_path, fps=fps)
            print(f"  [{mode}] Video exported to {video_path}")

            if progress_callback:
                progress_callback(100, "generated")

            # Cleanup GPU memory
            del output
            if use_i2v:
                del ref_image
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            return video_path

        except Exception as e:
            print(f"Error generating video: {e}")
            import traceback
            traceback.print_exc()

            # Emergency VRAM cleanup
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            return None

    def get_gpu_info(self):
        """Return GPU info for diagnostics."""
        if not torch.cuda.is_available():
            return {"gpu": "none", "vram_total": 0, "vram_used": 0}

        return {
            "gpu": torch.cuda.get_device_name(0),
            "vram_total_gb": round(torch.cuda.get_device_properties(0).total_mem / 1e9, 1),
            "vram_used_gb": round(torch.cuda.memory_allocated(0) / 1e9, 1),
            "vram_cached_gb": round(torch.cuda.memory_reserved(0) / 1e9, 1),
            "dtype": str(self.dtype),
            "sage_attention": self._sage_attention_available,
            "model": server_settings.I2V_MODEL_ID,
        }
