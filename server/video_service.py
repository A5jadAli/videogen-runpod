import os
import uuid
import gc
import time
import threading
import torch
import numpy as np
from PIL import Image
from server import server_settings
from server.request_queue import CAMERA_MOTION_PROMPTS, QUALITY_PRESETS


# ============================================================
# Thread-Safe SageAttention Context Manager
#
# CRITICAL: Never monkey-patch F.scaled_dot_product_attention globally
# in a threaded environment. Use a locked context manager that
# patches ONLY during generation and restores immediately after.
# This prevents SageAttention from corrupting CodeFormer/RIFE/ESRGAN.
# ============================================================
_sdpa_lock = threading.Lock()
_original_sdpa = None
_sage_attn_fn = None


def _init_sage_attention():
    """Initialize SageAttention safely. Called once at startup."""
    global _original_sdpa, _sage_attn_fn
    import torch.nn.functional as F
    _original_sdpa = F.scaled_dot_product_attention

    if not server_settings.ENABLE_SAGE_ATTENTION:
        return False

    try:
        from sageattention import sageattn
        _sage_attn_fn = sageattn
        print("SageAttention available — will use locked context for generation only.")
        return True
    except ImportError:
        print("SageAttention not installed — using default attention.")
        return False


class SageAttentionContext:
    """Context manager that enables SageAttention ONLY within its scope."""

    def __enter__(self):
        if _sage_attn_fn is not None:
            _sdpa_lock.acquire()
            import torch.nn.functional as F
            F.scaled_dot_product_attention = _sage_attn_fn
        return self

    def __exit__(self, *args):
        if _sage_attn_fn is not None:
            import torch.nn.functional as F
            F.scaled_dot_product_attention = _original_sdpa
            _sdpa_lock.release()


# ============================================================
# Timeout via threading.Timer (replaces broken signal.SIGALRM)
#
# signal.SIGALRM only works in the main thread. RunPod serverless
# runs handlers in worker threads, so SIGALRM raises ValueError.
# threading.Timer + Event is thread-safe and works everywhere.
# ============================================================
class GenerationTimeout(Exception):
    pass


class TimeoutWatchdog:
    """Thread-safe timeout that works in any thread (unlike SIGALRM)."""

    def __init__(self, timeout_seconds):
        self._timeout = timeout_seconds
        self._deadline = None
        self._expired = threading.Event()
        self._timer = None

    def start(self):
        self._deadline = time.monotonic() + self._timeout
        self._expired.clear()
        self._timer = threading.Timer(self._timeout, self._on_timeout)
        self._timer.daemon = True
        self._timer.start()
        return self

    def _on_timeout(self):
        self._expired.set()

    def check(self):
        """Call periodically during generation to check timeout."""
        if self._expired.is_set():
            raise GenerationTimeout(
                f"Video generation timed out after {self._timeout}s"
            )

    def cancel(self):
        if self._timer:
            self._timer.cancel()
            self._timer = None

    @property
    def is_expired(self):
        return self._expired.is_set()


class VideoService:
    """
    Wan 2.2 A14B MoE Video Generation Service with Post-Processing Pipeline.

    Key fixes from research:
    - VAE always float32 (bfloat16/float16 VAE causes corruption, #12141)
    - VAE tiling DISABLED (broken for AutoencoderKLWan, #12529)
    - SageAttention via locked context manager (not global monkey-patch)
    - Threading.Timer timeout (signal.SIGALRM crashes in non-main threads)
    - Proper VRAM lifecycle: maybe_free_model_hooks() before del
    - OOM retry logic outside except block (stack frames pin tensors)
    - CacheDiT cleanup on all error paths
    """

    def __init__(self):
        self.t2v_pipe = None
        self.i2v_pipe = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.bfloat16  # MUST be bfloat16 for Wan 2.2 (float16 overflows)
        self._sage_attention_available = _init_sage_attention()
        self._post_processor = None
        self._load_pipelines()
        self._init_post_processor()

    # ==============================================================
    # Post-Processor Initialization
    # ==============================================================

    def _init_post_processor(self):
        """Initialize the post-processing pipeline if enabled."""
        if not server_settings.ENABLE_POST_PROCESSING:
            print("Post-processing pipeline: DISABLED")
            return

        try:
            from server.post_processor import PostProcessor
            self._post_processor = PostProcessor(
                model_dir=server_settings.POST_PROCESSING_MODEL_DIR
            )
            info = self._post_processor.get_info()
            print(f"Post-processing pipeline: ENABLED")
            print(f"  Face restore ({info['face_model']}): "
                  f"{'✓' if info['codeformer_available'] or info['gfpgan_available'] else '✗'}")
            print(f"  Real-ESRGAN (upscale): {'✓' if info['realesrgan_available'] else '✗'}")
            print(f"  RIFE (interpolation):  {'✓' if info['rife_available'] else '✗ (FFmpeg fallback)'}")
        except Exception as e:
            print(f"Post-processing pipeline: FAILED to initialize ({e})")
            self._post_processor = None

    # ==============================================================
    # Model Loading
    # ==============================================================

    def _load_pipelines(self):
        """Load Wan 2.2 I2V pipeline."""
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

        # CRITICAL: VAE MUST be float32 (diffusers #12141)
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

        # CRITICAL: VAE MUST be float32 (diffusers #12141)
        # bfloat16 or float16 VAE produces corrupted output
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
        """
        Apply memory and speed optimizations.

        Priority:
        1. sequential_cpu_offload (lowest VRAM, slowest — use for <48GB GPUs)
        2. model_cpu_offload (balanced — RECOMMENDED for A14B on 80GB)
        3. Full CUDA (only if you have 80GB+ AND no post-processing)
        """
        if server_settings.ENABLE_SEQUENTIAL_CPU_OFFLOAD:
            print("  -> Enabling sequential CPU offload (lowest VRAM, slower)")
            pipe.enable_sequential_cpu_offload()
        elif server_settings.ENABLE_CPU_OFFLOAD:
            print("  -> Enabling model CPU offload (recommended for A14B)")
            pipe.enable_model_cpu_offload()
        else:
            print("  -> Moving pipeline to CUDA (full VRAM mode — requires 80GB+)")
            pipe.to(self.device)

        # CRITICAL: VAE tiling is BROKEN for AutoencoderKLWan (#12529)
        # DO NOT enable even if the setting says true — it causes tensor size mismatch
        if server_settings.ENABLE_VAE_TILING:
            print("  -> WARNING: VAE tiling requested but DISABLED (broken for Wan 2.2, #12529)")

        if server_settings.ENABLE_VAE_SLICING:
            print("  -> Enabling VAE slicing")
            pipe.enable_vae_slicing()

        # SageAttention status (actual patching happens in context manager)
        if self._sage_attention_available:
            print("  -> SageAttention ready (applied via locked context during generation)")
        else:
            print("  -> Using default SDPA attention")

        # torch.compile is INCOMPATIBLE with Wan 2.2 fullgraph (#12728)
        if server_settings.ENABLE_TORCH_COMPILE:
            print("  -> WARNING: torch.compile SKIPPED (incompatible with Wan 2.2, #12728)")

    # ==============================================================
    # CacheDiT — TaylorSeer + DBCache (~2-3x speedup)
    #
    # Preferred over TeaCache for MoE architectures because it
    # supports separate caching configs per expert transformer.
    # ==============================================================

    def _enable_cache_acceleration(self, pipe):
        """Enable CacheDiT acceleration with conservative quality thresholds."""
        if not server_settings.ENABLE_TEACACHE:
            return False

        try:
            import cache_dit
            from cache_dit import BasicCacheConfig

            cache_dit.enable_cache(
                pipe,
                cache_config=BasicCacheConfig(
                    max_warmup_steps=5,            # Full fidelity for first 5 layout steps
                    max_cached_steps=-1,           # No limit on cached steps
                    max_continuous_cached_steps=3,  # Max 3 consecutive cached steps
                    Fn_compute_blocks=1,           # Compute 1 block per cached step
                    Bn_compute_blocks=0,           # TaylorSeer as calibrator (FnB0)
                    residual_diff_threshold=0.10,  # Conservative for max quality (research: 0.08-0.12)
                    enable_separate_cfg=True,      # Required for Wan models
                ),
            )
            print("  -> CacheDiT acceleration enabled (TaylorSeer + DBCache, ~2-3x speedup)")
            return True
        except ImportError:
            print("  -> CacheDiT not installed (pip install cache-dit for ~2-3x speedup)")
            return False
        except Exception as e:
            print(f"  -> CacheDiT setup failed: {e}")
            return False

    def _disable_cache_acceleration(self, pipe):
        """Disable CacheDiT acceleration after generation."""
        if pipe is None:
            return
        try:
            import cache_dit
            cache_dit.disable_cache(pipe)
        except Exception:
            pass

    # ==============================================================
    # Generation Helpers
    # ==============================================================

    def _determine_flow_shift(self, height, width, flow_shift=None):
        """
        Auto-determine flow_shift based on resolution.

        Official Wan 2.2 recommendations:
        - 480p (832x480): flow_shift=3.0 — stable motion
        - 720p (1280x720): flow_shift=5.0 — balanced dynamics
        """
        if flow_shift is not None:
            return flow_shift
        total_pixels = height * width
        if total_pixels <= 832 * 480:
            return 3.0
        elif total_pixels <= 1280 * 720:
            return 5.0
        else:
            return 5.0

    def _enhance_prompt(self, prompt, camera_motion=None, level="light"):
        """Enhance prompt for Wan 2.2 I2V. Focus on MOTION, not appearance."""
        parts = [prompt.rstrip(". ")]

        if camera_motion and camera_motion in CAMERA_MOTION_PROMPTS:
            parts.append(CAMERA_MOTION_PROMPTS[camera_motion])

        if level == "light":
            parts.append("high quality, smooth motion, cinematic")
        elif level == "full":
            parts.append(
                "cinematic lighting, natural color grading, high detail, "
                "photorealistic, smooth natural movement, film grain"
            )

        return ". ".join(parts)

    def _prepare_reference_image(self, image_path, width, height):
        """
        Load and prepare reference image for I2V.

        Research: resize to match target exactly using Lanczos.
        Dimensions must be divisible by vae_scale_factor * patch_size.
        """
        img = Image.open(image_path).convert("RGB")

        img_ratio = img.width / img.height
        target_ratio = width / height

        if img_ratio > target_ratio:
            new_height = height
            new_width = int(height * img_ratio)
        else:
            new_width = width
            new_height = int(width / img_ratio)

        img = img.resize((new_width, new_height), Image.LANCZOS)

        left = (new_width - width) // 2
        top = (new_height - height) // 2
        img = img.crop((left, top, left + width, top + height))

        return img

    def _export_video_high_quality(self, frames, output_path, fps=16):
        """Export frames to MP4 with high-quality encoding. Used only when PP is disabled."""
        import imageio

        try:
            writer = imageio.get_writer(
                output_path,
                fps=fps,
                codec="libx264",
                quality=None,
                output_params=[
                    "-crf", "18",
                    "-preset", "slow",
                    "-pix_fmt", "yuv420p",
                    "-movflags", "+faststart",
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

        from diffusers.utils import export_to_video
        export_to_video(frames, output_path, fps=fps)
        return True

    # ==============================================================
    # Main Generation + Post-Processing
    # ==============================================================

    def generate_video(
        self,
        prompt,
        negative_prompt="",
        height=480,
        width=832,
        num_frames=81,
        guidance_scale=3.5,
        num_inference_steps=40,
        seed=None,
        flow_shift=None,
        fps=16,
        reference_image_path=None,
        camera_motion=None,
        enhance_prompt="light",
        quality_preset=None,
        progress_callback=None,
        max_generation_time=None,
        # Post-processing params
        enable_post_processing=True,
        enable_face_restore=True,
        enable_interpolation=True,
        enable_upscale=True,
        enable_ffmpeg_enhance=True,
        upscale_factor=2.0,
        target_fps=24,
        face_fidelity=0.6,
    ):
        """
        Generate video using Wan 2.2 A14B MoE model + post-processing pipeline.

        Returns:
            str: Path to final enhanced .mp4, or None on failure.
        """
        if max_generation_time is None:
            max_generation_time = server_settings.MAX_GENERATION_TIME

        # Thread-safe timeout (replaces signal.SIGALRM)
        watchdog = TimeoutWatchdog(max_generation_time)

        pipe = None
        cache_active = False
        ref_image = None
        need_oom_retry = False

        try:
            watchdog.start()

            if quality_preset and quality_preset in QUALITY_PRESETS:
                preset = QUALITY_PRESETS[quality_preset]
                num_inference_steps = preset.get("num_inference_steps", num_inference_steps)
                guidance_scale = preset.get("guidance_scale", guidance_scale)
                print(f"  Quality preset '{quality_preset}': "
                      f"steps={num_inference_steps}, guidance={guidance_scale}")

            use_i2v = reference_image_path is not None and os.path.exists(
                reference_image_path
            )

            if not use_i2v and self.t2v_pipe is None:
                print("Error: T2V pipeline is disabled. Provide a reference_image_url for I2V.")
                watchdog.cancel()
                return None

            pipe = self.i2v_pipe if use_i2v else self.t2v_pipe
            mode = "I2V" if use_i2v else "T2V"

            flow_shift = self._determine_flow_shift(height, width, flow_shift)

            original_prompt = prompt
            if enhance_prompt and enhance_prompt != "none":
                prompt = self._enhance_prompt(prompt, camera_motion, level=enhance_prompt)
                if prompt != original_prompt:
                    print(f"  [{mode}] Enhanced prompt: {prompt[:150]}...")

            print(
                f"  [{mode}] Generating: {width}x{height}, {num_frames} frames, "
                f"{num_inference_steps} steps, guidance={guidance_scale}, "
                f"flow_shift={flow_shift}, fps={fps}"
            )

            generator = None
            if seed is not None:
                generator = torch.Generator(device="cpu").manual_seed(seed)
                print(f"  [{mode}] Seed: {seed}")

            # Enable CacheDiT acceleration
            cache_active = self._enable_cache_acceleration(pipe)

            # Progress tracking
            total_steps = num_inference_steps
            progress_step_interval = max(1, total_steps // 10)
            do_post = (
                enable_post_processing
                and self._post_processor is not None
                and server_settings.ENABLE_POST_PROCESSING
            )

            def diffusion_step_callback(pipe_ref, step_index, timestep, callback_kwargs):
                # Check timeout on each step (thread-safe)
                watchdog.check()

                if progress_callback and (
                    step_index % progress_step_interval == 0
                    or step_index == total_steps - 1
                ):
                    max_pct = 70 if do_post else 90
                    progress_percent = int(((step_index + 1) / total_steps) * max_pct)
                    progress_callback(progress_percent, "generating")
                return callback_kwargs

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

            if flow_shift is not None:
                pipe_kwargs["flow_shift"] = flow_shift

            if use_i2v:
                ref_image = self._prepare_reference_image(
                    reference_image_path, width, height
                )
                pipe_kwargs["image"] = ref_image
                print(f"  [{mode}] Reference image loaded: "
                      f"{reference_image_path} -> {width}x{height}")

            # ========== Run Wan 2.2 Generation ==========
            # SageAttention is ONLY active inside this context manager.
            # It is automatically restored before post-processing runs.
            gen_start = time.time()
            with SageAttentionContext():
                with torch.inference_mode():
                    output = pipe(**pipe_kwargs).frames[0]
            gen_time = time.time() - gen_start
            print(f"  [{mode}] Generation complete: {gen_time:.1f}s")

            # Disable CacheDiT after generation
            if cache_active:
                self._disable_cache_acceleration(pipe)
                cache_active = False

            os.makedirs(server_settings.TEMP_DIR, exist_ok=True)

            # ========== Post-Processing Pipeline ==========
            # SageAttention is already restored (exited context manager)
            # so CodeFormer/RIFE/ESRGAN use standard SDPA
            if do_post:
                print(f"\n{'='*60}")
                print(f"  Starting post-processing pipeline (DIRECT frame mode)")
                print(f"  Face restore: {enable_face_restore and server_settings.ENABLE_FACE_RESTORE}")
                print(f"  Interpolation: {enable_interpolation and server_settings.ENABLE_INTERPOLATION}")
                print(f"  Upscale ({upscale_factor}x): {enable_upscale and server_settings.ENABLE_UPSCALE}")
                print(f"  FFmpeg enhance: {enable_ffmpeg_enhance and server_settings.ENABLE_FFMPEG_ENHANCE}")
                print(f"{'='*60}")

                # Convert PIL Images to numpy arrays
                raw_frames = []
                for frame in output:
                    if isinstance(frame, Image.Image):
                        raw_frames.append(np.array(frame))
                    elif isinstance(frame, np.ndarray):
                        raw_frames.append(frame)
                    else:
                        raw_frames.append(np.array(frame))

                # Free generation outputs AND move pipeline to CPU to reclaim VRAM.
                # With enable_model_cpu_offload(), the VAE was the last component
                # to run (decoding) and may still be on GPU (~500MB+).
                # Post-processing models need that VRAM.
                del output
                ref_image = None
                self._reclaim_pipeline_vram(pipe)

                final_filename = f"{uuid.uuid4()}.mp4"
                final_path = os.path.join(server_settings.TEMP_DIR, final_filename)

                def post_progress(pct, status):
                    if progress_callback:
                        overall = 70 + int(pct * 0.28)  # 70-98
                        progress_callback(overall, status)

                self._post_processor.process_frames(
                    frames=raw_frames,
                    output_video_path=final_path,
                    enable_face_restore=enable_face_restore and server_settings.ENABLE_FACE_RESTORE,
                    enable_interpolation=enable_interpolation and server_settings.ENABLE_INTERPOLATION,
                    enable_upscale=enable_upscale and server_settings.ENABLE_UPSCALE,
                    enable_ffmpeg_enhance=enable_ffmpeg_enhance and server_settings.ENABLE_FFMPEG_ENHANCE,
                    face_fidelity=face_fidelity,
                    upscale_factor=upscale_factor,
                    target_fps=target_fps,
                    original_fps=fps,
                    progress_callback=post_progress,
                )

                del raw_frames
                video_path = final_path
                print(f"  [{mode}] Post-processed video: {video_path}")
            else:
                # No post-processing — encode directly
                video_filename = f"{uuid.uuid4()}.mp4"
                video_path = os.path.join(server_settings.TEMP_DIR, video_filename)
                self._export_video_high_quality(output, video_path, fps=fps)

                del output
                ref_image = None
                self._reclaim_pipeline_vram(pipe)

                print(f"  [{mode}] Raw video (no post-processing): {video_path}")

            if progress_callback:
                progress_callback(100, "generated")

            watchdog.cancel()

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            return video_path

        except GenerationTimeout:
            watchdog.cancel()
            if cache_active and pipe is not None:
                self._disable_cache_acceleration(pipe)
            print(f"ERROR: Generation timed out after {max_generation_time}s")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return None

        except torch.cuda.OutOfMemoryError:
            # CRITICAL: Set retry flag and exit except block FIRST.
            # The exception object holds stack frame references that pin
            # tensors in GPU memory. We must clear them before retrying.
            watchdog.cancel()
            if cache_active and pipe is not None:
                self._disable_cache_acceleration(pipe)
            need_oom_retry = True
            print(f"ERROR: CUDA OOM during generation at {width}x{height}")

        except Exception as e:
            watchdog.cancel()
            if cache_active and pipe is not None:
                self._disable_cache_acceleration(pipe)
            print(f"Error generating video: {e}")
            import traceback
            traceback.print_exc()

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return None

        # OOM retry logic OUTSIDE except block (stack frames are released)
        if need_oom_retry:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Fallback: reduce resolution to 480p if we were above it
            if height > 480 or width > 832:
                print("  Retrying at 480p (832x480) after OOM...")
                return self.generate_video(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    height=480,
                    width=832,
                    num_frames=num_frames,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps,
                    seed=seed,
                    flow_shift=None,  # Let it auto-determine for 480p
                    fps=fps,
                    reference_image_path=reference_image_path,
                    camera_motion=camera_motion,
                    enhance_prompt=enhance_prompt,
                    quality_preset=quality_preset,
                    progress_callback=progress_callback,
                    max_generation_time=max_generation_time,
                    enable_post_processing=enable_post_processing,
                    enable_face_restore=enable_face_restore,
                    enable_interpolation=enable_interpolation,
                    enable_upscale=enable_upscale,
                    enable_ffmpeg_enhance=enable_ffmpeg_enhance,
                    upscale_factor=upscale_factor,
                    target_fps=target_fps,
                    face_fidelity=face_fidelity,
                )
            else:
                print("  Already at minimum resolution, cannot retry.")
                return None

    def _reclaim_pipeline_vram(self, pipe):
        """
        Explicitly move pipeline components to CPU before post-processing.

        With enable_model_cpu_offload(), the last-used component (usually VAE)
        may still be on GPU. Accelerate hooks will re-load components
        automatically on the next request, so this is safe.
        """
        if pipe is None:
            return
        try:
            for attr_name in ["vae", "transformer", "text_encoder", "image_encoder"]:
                component = getattr(pipe, attr_name, None)
                if component is not None and hasattr(component, "to"):
                    try:
                        component.to("cpu")
                    except Exception:
                        pass
        except Exception as e:
            print(f"  Warning: partial VRAM reclaim: {e}")

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if torch.cuda.is_available():
            vram_mb = torch.cuda.memory_allocated(0) / 1e6
            print(f"  VRAM after pipeline offload: {vram_mb:.0f}MB (ready for post-processing)")

    def cleanup_vram(self):
        """Aggressive VRAM cleanup between jobs."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

        vram_used = torch.cuda.memory_allocated(0) / 1e9 if torch.cuda.is_available() else 0
        print(f"  VRAM after cleanup: {vram_used:.2f}GB")

    def get_gpu_info(self):
        """Return GPU info for diagnostics."""
        if not torch.cuda.is_available():
            return {"gpu": "none", "vram_total": 0, "vram_used": 0}

        info = {
            "gpu": torch.cuda.get_device_name(0),
            "vram_total_gb": round(torch.cuda.get_device_properties(0).total_mem / 1e9, 1),
            "vram_used_gb": round(torch.cuda.memory_allocated(0) / 1e9, 1),
            "vram_cached_gb": round(torch.cuda.memory_reserved(0) / 1e9, 1),
            "dtype": str(self.dtype),
            "sage_attention": self._sage_attention_available,
            "teacache": server_settings.ENABLE_TEACACHE,
            "model": server_settings.I2V_MODEL_ID,
            "post_processing": self._post_processor is not None,
        }

        if self._post_processor:
            info["post_processing_info"] = self._post_processor.get_info()

        return info
