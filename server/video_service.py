import os
import uuid
import gc
import torch
from PIL import Image
from diffusers import AutoencoderKLWan, WanPipeline, WanImageToVideoPipeline
from diffusers.utils import export_to_video
from server import server_settings


class VideoService:
    """
    Manages Wan 2.1 pipelines for text-to-video and image-to-video generation.
    Instantiated ONCE globally in main.py. Both pipelines are loaded at startup
    and reused across all requests.

    Models:
    - Text-to-Video: Wan-AI/Wan2.1-T2V-14B-Diffusers (pure text prompt to video)
    - Image-to-Video: Wan-AI/Wan2.1-I2V-14B-720P-Diffusers (reference image + prompt to video, preserves identity)

    No safety filters or content restrictions are applied.
    """

    def __init__(self):
        self.t2v_pipe = None
        self.i2v_pipe = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._load_pipelines()

    def _load_pipelines(self):
        """Load both T2V and I2V pipelines with no safety checkers."""
        print("=" * 60)
        print("Loading Wan 2.1 Text-to-Video pipeline...")
        print("=" * 60)
        self._load_t2v()

        print("=" * 60)
        print("Loading Wan 2.1 Image-to-Video pipeline...")
        print("=" * 60)
        self._load_i2v()

        print("=" * 60)
        print("All pipelines loaded successfully.")
        print("=" * 60)

    def _load_t2v(self):
        """Load the Text-to-Video pipeline."""
        model_id = server_settings.T2V_MODEL_ID

        # Load VAE in float32 for quality
        vae = AutoencoderKLWan.from_pretrained(
            model_id,
            subfolder="vae",
            torch_dtype=torch.float32,
            cache_dir=server_settings.MODEL_CACHE_DIR,
        )

        self.t2v_pipe = WanPipeline.from_pretrained(
            model_id,
            vae=vae,
            torch_dtype=torch.bfloat16,
            cache_dir=server_settings.MODEL_CACHE_DIR,
        )

        # Disable safety checker if it exists
        if hasattr(self.t2v_pipe, "safety_checker"):
            self.t2v_pipe.safety_checker = None
        if hasattr(self.t2v_pipe, "feature_extractor"):
            self.t2v_pipe.feature_extractor = None

        self._apply_memory_optimizations(self.t2v_pipe)
        print("T2V pipeline loaded.")

    def _load_i2v(self):
        """Load the Image-to-Video pipeline for identity preservation."""
        model_id = server_settings.I2V_MODEL_ID

        # Load VAE in float32 for quality
        vae = AutoencoderKLWan.from_pretrained(
            model_id,
            subfolder="vae",
            torch_dtype=torch.float32,
            cache_dir=server_settings.MODEL_CACHE_DIR,
        )

        self.i2v_pipe = WanImageToVideoPipeline.from_pretrained(
            model_id,
            vae=vae,
            torch_dtype=torch.bfloat16,
            cache_dir=server_settings.MODEL_CACHE_DIR,
        )

        # Disable safety checker if it exists
        if hasattr(self.i2v_pipe, "safety_checker"):
            self.i2v_pipe.safety_checker = None
        if hasattr(self.i2v_pipe, "feature_extractor"):
            self.i2v_pipe.feature_extractor = None

        self._apply_memory_optimizations(self.i2v_pipe)
        print("I2V pipeline loaded.")

    def _apply_memory_optimizations(self, pipe):
        """Apply memory optimizations based on environment config."""
        if server_settings.ENABLE_SEQUENTIAL_CPU_OFFLOAD:
            print("  -> Enabling sequential CPU offload (lowest VRAM, slower)")
            pipe.enable_sequential_cpu_offload()
        elif server_settings.ENABLE_CPU_OFFLOAD:
            print("  -> Enabling model CPU offload")
            pipe.enable_model_cpu_offload()
        else:
            print("  -> Moving pipeline to CUDA (full VRAM mode)")
            pipe.to(self.device)

        if server_settings.ENABLE_VAE_TILING:
            print("  -> Enabling VAE tiling")
            pipe.enable_vae_tiling()

    def _determine_flow_shift(self, height, width, flow_shift=None):
        """
        Auto-determine flow_shift based on resolution.
        Lower values (2-5) for 480p, higher values (7-12) for 720p+.
        """
        if flow_shift is not None:
            return flow_shift
        total_pixels = height * width
        if total_pixels <= 832 * 480:
            return 3.0
        else:
            return 9.0

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
        fps=15,
        reference_image_path=None,
        progress_callback=None,
    ):
        """
        Generate a video from text prompt (T2V) or from reference image + prompt (I2V).

        If reference_image_path is provided, uses I2V pipeline (identity preservation).
        Otherwise uses T2V pipeline.

        Returns:
            str: Absolute path to generated .mp4 file, or None on failure.
        """
        try:
            # Decide which pipeline to use
            use_i2v = reference_image_path is not None and os.path.exists(reference_image_path)
            pipe = self.i2v_pipe if use_i2v else self.t2v_pipe
            mode = "I2V" if use_i2v else "T2V"

            print(f"[{mode}] Starting generation: {width}x{height}, {num_frames} frames, "
                  f"{num_inference_steps} steps, guidance={guidance_scale}")

            # Set up generator for reproducibility
            generator = None
            if seed is not None:
                generator = torch.Generator(device="cpu").manual_seed(seed)

            # Build step callback for progress
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

            # For I2V, load and pass reference image
            if use_i2v:
                ref_image = Image.open(reference_image_path).convert("RGB")
                ref_image = ref_image.resize((width, height), Image.LANCZOS)
                pipe_kwargs["image"] = ref_image
                print(f"[I2V] Reference image loaded: {reference_image_path} -> {width}x{height}")

            # Run the pipeline
            output = pipe(**pipe_kwargs).frames[0]

            if progress_callback:
                progress_callback(95, "exporting")

            # Export to MP4
            os.makedirs(server_settings.TEMP_DIR, exist_ok=True)
            video_filename = f"{uuid.uuid4()}.mp4"
            video_path = os.path.join(server_settings.TEMP_DIR, video_filename)

            export_to_video(output, video_path, fps=fps)
            print(f"[{mode}] Video exported to {video_path}")

            if progress_callback:
                progress_callback(100, "generated")

            # Cleanup
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
            return None
