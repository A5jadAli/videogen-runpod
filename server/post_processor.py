"""
Post-Processing Pipeline for Wan 2.2 A14B Video Generation.

Four-stage enhancement pipeline:

  Stage 1: Face Restoration    — CodeFormer FP16 (w=0.6, identity-preserving codebook lookup)
  Stage 2: Frame Interpolation — RIFE v4.22+ (multi-timestep for exact target fps)
  Stage 3: Video Upscaling     — Real-ESRGAN x4plus (480p → exact 1080p, FP16 tiled)
  Stage 4: FFmpeg Finishing    — deflicker, deband, denoise, sharpen, grain, encode

Research-backed ordering: Face → Interpolate → Upscale → FFmpeg
- Interpolate BEFORE upscale: optical flow is more accurate at lower resolution
- Upscale cleans up minor RIFE artifacts
- FFmpeg is always last (operates on final resolution)

CRITICAL DESIGN: Zero intermediate encoding. Raw numpy arrays flow through
all 4 stages with ZERO disk I/O until the final FFmpeg pipe encode.
"""

import os
import gc
import time
import uuid
import subprocess
import tempfile
import shutil
import numpy as np
import torch
from PIL import Image
from typing import List, Optional, Callable, Tuple


# Standard output resolutions for exact-pixel delivery
STANDARD_RESOLUTIONS = {
    (832, 480): (1920, 1080),
    (848, 480): (1920, 1080),
    (864, 480): (1920, 1080),
    (816, 480): (1920, 1080),
}


class PostProcessor:
    """
    Multi-stage video post-processing pipeline.

    Models are loaded lazily on first use and cached for subsequent calls.
    All frame processing happens in-memory — no intermediate disk encoding.

    Primary entry point: process_frames() — accepts raw numpy arrays directly
    from the diffusion pipeline, avoiding any intermediate encode/decode.
    """

    def __init__(self, model_dir="/app/models"):
        self.model_dir = model_dir
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Lazy-loaded models
        self._face_restorer = None
        self._face_helper = None
        self._face_mode = None
        self._realesrgan_upsampler = None
        self._rife_model = None
        self._rife_available = None

        print("[PostProcessor] Initialized (models load on first use)")

    # ==============================================================
    # Primary Entry Point — Direct Frame Input
    # ==============================================================

    def process_frames(
        self,
        frames: List[np.ndarray],
        output_video_path: str,
        enable_face_restore: bool = True,
        enable_interpolation: bool = True,
        enable_upscale: bool = True,
        enable_ffmpeg_enhance: bool = True,
        face_fidelity: float = 0.6,
        upscale_factor: float = 2.0,
        target_fps: int = 24,
        original_fps: int = 16,
        progress_callback: Optional[Callable] = None,
    ) -> str:
        """
        Process raw numpy frames directly — NO intermediate encode/decode.

        Order: Face Restore → Interpolate → Upscale → FFmpeg
        Research confirms interpolation before upscale produces better results.
        """
        pipeline_start = time.time()

        try:
            self._report(progress_callback, 0, "post_processing")

            if not frames:
                print("[PostProcessor] ERROR: No frames provided")
                return output_video_path

            # Convert RGB → BGR for OpenCV-based processing
            h, w = frames[0].shape[:2]
            print(f"[PostProcessor] Processing {len(frames)} frames ({w}x{h}) — DIRECT mode")
            bgr_frames = [f[:, :, ::-1].copy() for f in frames]
            del frames
            gc.collect()

            current_fps = original_fps

            # Stage 1: Face Restoration (at native resolution — most efficient)
            if enable_face_restore:
                self._report(progress_callback, 5, "restoring_faces")
                print(f"[PostProcessor] Stage 1: Face restoration (CodeFormer w={face_fidelity})")
                stage_start = time.time()
                bgr_frames = self._restore_faces(bgr_frames, fidelity_weight=face_fidelity)
                print(f"[PostProcessor] Face restoration: {time.time() - stage_start:.1f}s")
                self._unload_face_models()

            # Stage 2: Frame Interpolation (at native resolution — BEFORE upscale)
            # Research: optical flow estimation degrades at higher resolution
            if enable_interpolation and target_fps > original_fps:
                self._report(progress_callback, 25, "interpolating_frames")
                print(f"[PostProcessor] Stage 2: Frame interpolation ({original_fps}→{target_fps}fps)")
                stage_start = time.time()
                bgr_frames, current_fps = self._interpolate_frames(bgr_frames, original_fps, target_fps)
                print(f"[PostProcessor] Interpolation: {time.time() - stage_start:.1f}s → {len(bgr_frames)} frames at {current_fps}fps")
                self._unload_rife()

            # Stage 3: Upscaling (after interpolation for cleaner results)
            if enable_upscale:
                self._report(progress_callback, 40, "upscaling")
                print(f"[PostProcessor] Stage 3: Upscaling ({upscale_factor}x via Real-ESRGAN)")
                stage_start = time.time()
                bgr_frames = self._upscale_frames(bgr_frames, upscale_factor, progress_callback)
                print(f"[PostProcessor] Upscaling: {time.time() - stage_start:.1f}s")
                self._unload_upscaler()

            # Stage 4: Single FFmpeg encode (THE ONLY ENCODE in the entire pipeline)
            self._report(progress_callback, 88, "finishing")
            if enable_ffmpeg_enhance:
                print(f"[PostProcessor] Stage 4: FFmpeg finishing (deflicker, hqdn3d, deband, cas, grain)")
            else:
                print(f"[PostProcessor] Stage 4: FFmpeg clean encode (no filters)")
            stage_start = time.time()
            self._ffmpeg_pipe_encode(bgr_frames, output_video_path, current_fps, apply_filters=enable_ffmpeg_enhance)
            print(f"[PostProcessor] FFmpeg encode: {time.time() - stage_start:.1f}s")

            del bgr_frames
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            total_time = time.time() - pipeline_start
            print(f"[PostProcessor] Pipeline complete: {total_time:.1f}s total")
            self._report(progress_callback, 95, "post_processing_complete")
            return output_video_path

        except Exception as e:
            print(f"[PostProcessor] ERROR: {e}")
            import traceback
            traceback.print_exc()
            return output_video_path

    # ==============================================================
    # Fallback Entry Point — File-based Input
    # ==============================================================

    def process(
        self,
        input_video_path: str,
        output_video_path: str,
        enable_face_restore: bool = True,
        enable_interpolation: bool = True,
        enable_upscale: bool = True,
        enable_ffmpeg_enhance: bool = True,
        face_fidelity: float = 0.6,
        upscale_factor: float = 2.0,
        target_fps: int = 24,
        original_fps: int = 16,
        progress_callback: Optional[Callable] = None,
    ) -> str:
        """Fallback: extract frames from file, then process."""
        try:
            print(f"[PostProcessor] Extracting frames from {input_video_path}")
            frames_bgr = self._extract_frames(input_video_path)

            if not frames_bgr:
                print("[PostProcessor] ERROR: No frames extracted")
                shutil.copy2(input_video_path, output_video_path)
                return output_video_path

            # Convert BGR → RGB for process_frames input
            frames_rgb = [f[:, :, ::-1].copy() for f in frames_bgr]
            del frames_bgr

            return self.process_frames(
                frames=frames_rgb,
                output_video_path=output_video_path,
                enable_face_restore=enable_face_restore,
                enable_interpolation=enable_interpolation,
                enable_upscale=enable_upscale,
                enable_ffmpeg_enhance=enable_ffmpeg_enhance,
                face_fidelity=face_fidelity,
                upscale_factor=upscale_factor,
                target_fps=target_fps,
                original_fps=original_fps,
                progress_callback=progress_callback,
            )

        except Exception as e:
            print(f"[PostProcessor] ERROR: {e}")
            import traceback
            traceback.print_exc()
            if os.path.exists(input_video_path):
                shutil.copy2(input_video_path, output_video_path)
            return output_video_path

    # ==============================================================
    # Frame Extraction (fallback path only)
    # ==============================================================

    def _extract_frames(self, video_path: str) -> List[np.ndarray]:
        import cv2
        frames = []
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[PostProcessor] Failed to open video: {video_path}")
            return frames
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        return frames

    # ==============================================================
    # Stage 1: Face Restoration (CodeFormer FP16 primary, GFPGAN fallback)
    # ==============================================================

    def _load_face_restorer(self):
        if self._face_restorer is not None:
            return self._face_restorer

        # Try CodeFormer first
        try:
            from basicsr.archs.codeformer_arch import CodeFormer as CodeFormerArch
            from facexlib.utils.face_restoration_helper import FaceRestoreHelper

            codeformer_path = os.path.join(self.model_dir, "codeformer", "codeformer.pth")
            if not os.path.exists(codeformer_path):
                for fb in ["/app/models/codeformer/codeformer.pth"]:
                    if os.path.exists(fb):
                        codeformer_path = fb
                        break

            if os.path.exists(codeformer_path):
                net = CodeFormerArch(
                    dim_embd=512, codebook_size=1024, n_head=8, n_layers=9,
                    connect_list=["32", "64", "128", "256"],
                ).to(self.device)

                ckpt = torch.load(codeformer_path, map_location=self.device, weights_only=False)
                net.load_state_dict(ckpt["params_ema"])
                net.eval()

                if self.device == "cuda":
                    net = net.half()
                    print("[PostProcessor] CodeFormer loaded in FP16")

                self._face_restorer = net

                det_model_dir = os.path.join(self.model_dir, "facelib")
                self._face_helper = FaceRestoreHelper(
                    upscale_factor=1,
                    face_size=512,
                    crop_ratio=(1, 1),
                    det_model="retinaface_resnet50",
                    save_ext="png",
                    use_parse=True,
                    device=self.device,
                    model_rootpath=det_model_dir if os.path.isdir(det_model_dir) else None,
                )
                self._face_mode = "codeformer"
                print(f"[PostProcessor] CodeFormer loaded from {codeformer_path}")
                return self._face_restorer
        except Exception as e:
            print(f"[PostProcessor] CodeFormer load failed: {e}")

        # Fallback to GFPGAN
        try:
            from gfpgan import GFPGANer
            model_path = os.path.join(self.model_dir, "gfpgan", "GFPGANv1.4.pth")
            if not os.path.exists(model_path):
                for fb in ["/app/models/gfpgan/GFPGANv1.4.pth", "GFPGANv1.4.pth"]:
                    if os.path.exists(fb):
                        model_path = fb
                        break

            self._face_restorer = GFPGANer(
                model_path=model_path, upscale=1, arch="clean",
                channel_multiplier=2, bg_upsampler=None,
                device=torch.device(self.device),
            )
            self._face_mode = "gfpgan"
            print(f"[PostProcessor] GFPGAN fallback loaded from {model_path}")
            return self._face_restorer
        except Exception as e:
            print(f"[PostProcessor] GFPGAN also failed: {e}")
            return None

    def _restore_faces(self, frames: List[np.ndarray], fidelity_weight: float = 0.6) -> List[np.ndarray]:
        restorer = self._load_face_restorer()
        if restorer is None:
            print("[PostProcessor] Skipping face restoration (no model available)")
            return frames

        if self._face_mode == "codeformer":
            return self._restore_codeformer(frames, fidelity_weight)
        else:
            return self._restore_gfpgan(frames)

    def _restore_codeformer(self, frames: List[np.ndarray], w: float) -> List[np.ndarray]:
        """CodeFormer FP16 restoration. w=0.6 optimal for AI video."""
        from basicsr.utils import img2tensor, tensor2img
        from torchvision.transforms.functional import normalize

        use_fp16 = self.device == "cuda" and next(self._face_restorer.parameters()).dtype == torch.float16
        restored = []
        faces_found = 0

        for i, frame in enumerate(frames):
            try:
                self._face_helper.clean_all()
                self._face_helper.read_image(frame)
                self._face_helper.get_face_landmarks_5(
                    only_center_face=False, eye_dist_threshold=3,
                )
                self._face_helper.align_warp_face()

                num_faces = len(self._face_helper.cropped_faces)
                faces_found += num_faces

                if num_faces == 0:
                    restored.append(frame)
                    continue

                for cropped_face in self._face_helper.cropped_faces:
                    face_t = img2tensor(cropped_face / 255.0, bgr2rgb=True, float32=True)
                    normalize(face_t, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5], inplace=True)
                    face_t = face_t.unsqueeze(0).to(self.device)

                    if use_fp16:
                        face_t = face_t.half()

                    with torch.inference_mode():
                        output = self._face_restorer(face_t, w=w, adain=True)[0]

                    restored_face = tensor2img(output.squeeze(0).float(), rgb2bgr=True, min_max=(-1, 1))
                    restored_face = restored_face.astype("uint8")
                    self._face_helper.add_restored_face(restored_face)

                self._face_helper.get_inverse_affine(None)
                output_frame = self._face_helper.paste_faces_to_input_image()
                restored.append(output_frame)

            except Exception as e:
                if i == 0:
                    print(f"[PostProcessor] CodeFormer error frame {i}: {e}")
                restored.append(frame)

        print(f"[PostProcessor] CodeFormer FP16: {faces_found} faces across {len(frames)} frames (w={w})")
        return restored

    def _restore_gfpgan(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        restored = []
        faces_found = 0
        for i, frame in enumerate(frames):
            try:
                _, rf, output = self._face_restorer.enhance(
                    frame, has_aligned=False, only_center_face=False, paste_back=True, weight=0.5,
                )
                if rf:
                    faces_found += len(rf)
                restored.append(output if output is not None else frame)
            except Exception as e:
                if i == 0:
                    print(f"[PostProcessor] GFPGAN error frame {i}: {e}")
                restored.append(frame)
        print(f"[PostProcessor] GFPGAN: {faces_found} faces across {len(frames)} frames")
        return restored

    # ==============================================================
    # Stage 2: Frame Interpolation (RIFE / FFmpeg fallback)
    # ==============================================================

    def _check_rife_available(self) -> bool:
        if self._rife_available is not None:
            return self._rife_available
        try:
            import sys
            rife_path = "/app/rife"
            if os.path.isdir(rife_path) and rife_path not in sys.path:
                sys.path.insert(0, rife_path)
            from model.RIFE import Model as RIFEModel
            if os.path.isdir("/app/rife/train_log"):
                self._rife_available = True
            else:
                self._rife_available = False
        except ImportError:
            self._rife_available = False
        return self._rife_available

    def _get_rife(self):
        if self._rife_model is not None:
            return self._rife_model
        if not self._check_rife_available():
            return None
        try:
            import sys
            if "/app/rife" not in sys.path:
                sys.path.insert(0, "/app/rife")
            from model.RIFE import Model as RIFEModel
            model = RIFEModel()
            model.load_model("/app/rife/train_log", -1)
            model.eval()
            model.device()
            self._rife_model = model
            print("[PostProcessor] RIFE model loaded")
            return self._rife_model
        except Exception as e:
            print(f"[PostProcessor] RIFE load failed: {e}")
            self._rife_available = False
            return None

    def _rife_inference(self, f0: np.ndarray, f1: np.ndarray, t: float) -> np.ndarray:
        """Single RIFE inference at arbitrary timestep t in (0, 1). Input/output: BGR."""
        img0 = f0[:, :, ::-1].copy()
        img1 = f1[:, :, ::-1].copy()
        img0_t = torch.from_numpy(img0.transpose(2, 0, 1)).float().unsqueeze(0).to(self.device) / 255.0
        img1_t = torch.from_numpy(img1.transpose(2, 0, 1)).float().unsqueeze(0).to(self.device) / 255.0

        h, w = img0_t.shape[2], img0_t.shape[3]
        ph = ((h - 1) // 64 + 1) * 64
        pw = ((w - 1) // 64 + 1) * 64
        img0_p = torch.nn.functional.pad(img0_t, (0, pw - w, 0, ph - h))
        img1_p = torch.nn.functional.pad(img1_t, (0, pw - w, 0, ph - h))

        with torch.inference_mode():
            mid = self._rife_model.inference(img0_p, img1_p, timestep=t)

        mid = mid[:, :, :h, :w]
        mid_np = (mid[0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8).transpose(1, 2, 0)
        return mid_np[:, :, ::-1].copy()

    def _interpolate_frames(self, frames: List[np.ndarray], input_fps: int, target_fps: int) -> Tuple[List[np.ndarray], int]:
        rife = self._get_rife()
        if rife is not None:
            return self._rife_exact(frames, input_fps, target_fps)
        return self._ffmpeg_interpolate(frames, input_fps, target_fps)

    def _rife_exact(self, frames: List[np.ndarray], in_fps: int, out_fps: int) -> Tuple[List[np.ndarray], int]:
        """RIFE interpolation targeting exact output FPS with arbitrary timestep."""
        n = len(frames)
        duration = (n - 1) / in_fps
        n_out = round(duration * out_fps) + 1
        result = []

        for out_i in range(n_out):
            t_out = out_i / out_fps
            src_pos = t_out * in_fps
            src_idx = int(src_pos)

            if src_idx >= n - 1:
                result.append(frames[-1])
                continue

            frac = src_pos - src_idx
            if frac < 0.01:
                result.append(frames[src_idx])
            elif frac > 0.99:
                result.append(frames[min(src_idx + 1, n - 1)])
            else:
                result.append(self._rife_inference(frames[src_idx], frames[src_idx + 1], frac))

        print(f"[PostProcessor] RIFE: {n}@{in_fps}fps -> {len(result)}@{out_fps}fps (exact)")
        return result, out_fps

    def _ffmpeg_interpolate(self, frames: List[np.ndarray], in_fps: int, out_fps: int) -> Tuple[List[np.ndarray], int]:
        """FFmpeg minterpolate fallback."""
        work_dir = tempfile.mkdtemp(prefix="interp_")
        try:
            inp = os.path.join(work_dir, "in.mp4")
            out = os.path.join(work_dir, "out.mp4")
            self._ffmpeg_pipe_encode(frames, inp, in_fps, apply_filters=False, crf=10)

            cmd = [
                "ffmpeg", "-y", "-i", inp,
                "-vf", f"minterpolate=fps={out_fps}:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1",
                "-c:v", "libx264", "-crf", "10", "-preset", "fast", "-pix_fmt", "yuv420p", out,
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            if result.returncode != 0:
                return frames, in_fps

            import cv2
            interp = []
            cap = cv2.VideoCapture(out)
            while True:
                ret, f = cap.read()
                if not ret:
                    break
                interp.append(f)
            cap.release()
            return (interp, out_fps) if interp else (frames, in_fps)
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)

    # ==============================================================
    # Stage 3: Video Upscaling (Real-ESRGAN x4plus → exact 1080p)
    # ==============================================================

    def _determine_tile_size(self) -> int:
        """Auto-determine Real-ESRGAN tile size based on available VRAM."""
        if not torch.cuda.is_available():
            return 256

        try:
            free_vram_gb = (
                torch.cuda.get_device_properties(0).total_mem
                - torch.cuda.memory_allocated(0)
            ) / 1e9
            if free_vram_gb >= 40:
                return 0     # No tiling — best quality (no seam artifacts)
            elif free_vram_gb >= 20:
                return 512
            elif free_vram_gb >= 12:
                return 400
            elif free_vram_gb >= 8:
                return 320
            elif free_vram_gb >= 5:
                return 256
            else:
                return 192
        except Exception:
            return 320

    def _get_realesrgan(self):
        if self._realesrgan_upsampler is not None:
            return self._realesrgan_upsampler
        try:
            from basicsr.archs.rrdbnet_arch import RRDBNet
            from realesrgan import RealESRGANer

            model_path = os.path.join(self.model_dir, "realesrgan", "RealESRGAN_x4plus.pth")
            if not os.path.exists(model_path):
                for fb in ["/app/models/realesrgan/RealESRGAN_x4plus.pth", "RealESRGAN_x4plus.pth"]:
                    if os.path.exists(fb):
                        model_path = fb
                        break

            tile_size = self._determine_tile_size()
            # Research: tile_pad=16 reduces seam artifacts vs default 10
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
            self._realesrgan_upsampler = RealESRGANer(
                scale=4, model_path=model_path, dni_weight=None, model=model,
                tile=tile_size, tile_pad=16, pre_pad=0, half=True,
                device=torch.device(self.device),
            )
            print(f"[PostProcessor] Real-ESRGAN loaded (tile={tile_size}, tile_pad=16, half=True)")
            return self._realesrgan_upsampler
        except Exception as e:
            print(f"[PostProcessor] Real-ESRGAN load failed: {e}")
            return None

    def _resize_to_standard(self, frames: List[np.ndarray], input_w: int, input_h: int) -> List[np.ndarray]:
        """Resize to exact standard resolution (e.g., 1920x1080) after upscaling."""
        import cv2

        target = STANDARD_RESOLUTIONS.get((input_w, input_h))
        if target is None:
            return frames

        target_w, target_h = target
        actual_h, actual_w = frames[0].shape[:2]

        if actual_w == target_w and actual_h == target_h:
            return frames

        if abs(actual_w - target_w) / target_w > 0.25 or abs(actual_h - target_h) / target_h > 0.25:
            print(f"[PostProcessor] Skipping resize: {actual_w}x{actual_h} too far from {target_w}x{target_h}")
            return frames

        print(f"[PostProcessor] Resizing {actual_w}x{actual_h} -> {target_w}x{target_h} (exact 1080p)")
        resized = []
        for frame in frames:
            resized.append(cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4))
        return resized

    def _upscale_frames(self, frames: List[np.ndarray], scale: float = 2.0, progress_callback: Optional[Callable] = None) -> List[np.ndarray]:
        upsampler = self._get_realesrgan()
        if upsampler is None:
            return frames

        input_h, input_w = frames[0].shape[:2]

        # Progressive tile size fallback chain for OOM recovery
        tile_fallback_chain = sorted(set(filter(None, [upsampler.tile, 512, 320, 256, 192, 128])), reverse=True)

        upscaled = []
        total = len(frames)
        current_tile_idx = 0

        for i, frame in enumerate(frames):
            success = False
            while current_tile_idx < len(tile_fallback_chain):
                try:
                    upsampler.tile = tile_fallback_chain[current_tile_idx]
                    output, _ = upsampler.enhance(frame, outscale=scale)
                    upscaled.append(output)
                    success = True
                    break
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        torch.cuda.empty_cache()
                        current_tile_idx += 1
                        if current_tile_idx < len(tile_fallback_chain):
                            print(f"[PostProcessor] OOM at tile={tile_fallback_chain[current_tile_idx-1]}, "
                                  f"falling back to tile={tile_fallback_chain[current_tile_idx]}")
                        continue
                    else:
                        break
                except Exception:
                    break

            if not success:
                if i == 0:
                    print(f"[PostProcessor] Upscale failed on frame {i}, using original")
                upscaled.append(frame)

            if progress_callback and total > 0 and i % max(1, total // 10) == 0:
                progress_callback(40 + int((i / total) * 45), "upscaling")

        # Resize to exact 1080p if applicable
        if upscaled and upscaled[0].shape[:2] != (input_h, input_w):
            upscaled = self._resize_to_standard(upscaled, input_w, input_h)

        if upscaled and frames:
            print(f"[PostProcessor] Upscaled {len(upscaled)} frames: "
                  f"{input_w}x{input_h} -> {upscaled[0].shape[1]}x{upscaled[0].shape[0]}")
        return upscaled

    # ==============================================================
    # Stage 4: FFmpeg — Direct stdin pipe encoding
    #
    # Research-optimized filter chain for Wan 2.2 + Real-ESRGAN:
    # 1. deflicker (7-frame) — fix Real-ESRGAN temporal flicker
    # 2. hqdn3d — light spatial, moderate temporal denoise
    # 3. deband — remove AI gradient banding (highest ROI filter)
    # 4. cas — contrast-adaptive sharpen (fewer halos than unsharp)
    # 5. noise — film grain to mask AI artifacts
    #
    # Encoding: -tune grain preserves the added grain through encode
    # ==============================================================

    def _ffmpeg_pipe_encode(self, frames: List[np.ndarray], output_path: str, fps: int,
                            apply_filters: bool = True, crf: int = 16):
        """Encode frames to H.264 via FFmpeg stdin pipe. ZERO intermediate files."""
        if not frames:
            return
        h, w = frames[0].shape[:2]

        if apply_filters:
            # Research-backed filter chain:
            # - deflicker size=7 (wider window = smoother temporal, mode=am for averaging)
            # - hqdn3d 1.5:1.5:6:6 (light spatial, moderate temporal — AI flicker fix)
            # - deband 0.03 threshold (AI content has frequent color banding)
            # - cas 0.4 (contrast-adaptive sharpen — cleaner than unsharp)
            # - noise c0s=6 (subtle grain masks AI temporal artifacts)
            vf = ["-vf", ",".join([
                "deflicker=size=7:mode=am",
                "hqdn3d=1.5:1.5:6:6",
                "deband=1thr=0.03:2thr=0.03:3thr=0.03:range=16:blur=1",
                "cas=strength=0.4",
                "eq=contrast=1.03:saturation=1.08:brightness=0.01",
                "noise=c0s=6:c0f=t+u",
            ])]
        else:
            vf = []

        cmd = [
            "ffmpeg", "-y",
            "-f", "rawvideo", "-vcodec", "rawvideo",
            "-s", f"{w}x{h}", "-pix_fmt", "bgr24", "-r", str(fps),
            "-i", "pipe:0",
            *vf,
            "-c:v", "libx264", "-crf", str(crf),
            "-preset", "slow", "-tune", "grain",  # -tune grain preserves film grain
            "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            output_path,
        ]

        try:
            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            for frame in frames:
                if not frame.flags['C_CONTIGUOUS']:
                    frame = np.ascontiguousarray(frame)
                proc.stdin.write(frame.tobytes())
            proc.stdin.close()
            _, stderr = proc.communicate(timeout=300)
            if proc.returncode != 0:
                err = stderr.decode()[-500:] if stderr else "unknown"
                print(f"[PostProcessor] FFmpeg pipe failed: {err}")
                # Fallback: try without CAS filter (not available in all ffmpeg builds)
                if apply_filters and ("cas" in err.lower() or "no such filter" in err.lower()
                                      or "unknown filter" in err.lower()):
                    print("[PostProcessor] Filter error detected, retrying with fallback filters")
                    self._ffmpeg_pipe_encode_fallback(frames, output_path, fps)
                elif apply_filters:
                    self._ffmpeg_pipe_encode(frames, output_path, fps, apply_filters=False)
        except Exception as e:
            print(f"[PostProcessor] FFmpeg pipe error: {e}")
            # Last-resort fallback via OpenCV
            import cv2
            fourcc = cv2.VideoWriter_fourcc(*"avc1")
            writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
            for frame in frames:
                writer.write(frame)
            writer.release()

    def _ffmpeg_pipe_encode_fallback(self, frames, output_path, fps):
        """Fallback filter chain without CAS (uses unsharp instead)."""
        if not frames:
            return
        h, w = frames[0].shape[:2]

        vf = ["-vf", ",".join([
            "deflicker=size=7:mode=am",
            "hqdn3d=1.5:1.5:6:6",
            "deband=1thr=0.03:2thr=0.03:3thr=0.03:range=16:blur=1",
            "unsharp=5:5:0.6:5:5:0.3",
            "eq=contrast=1.03:saturation=1.08:brightness=0.01",
            "noise=c0s=6:c0f=t+u",
        ])]

        cmd = [
            "ffmpeg", "-y",
            "-f", "rawvideo", "-vcodec", "rawvideo",
            "-s", f"{w}x{h}", "-pix_fmt", "bgr24", "-r", str(fps),
            "-i", "pipe:0",
            *vf,
            "-c:v", "libx264", "-crf", "16",
            "-preset", "slow", "-tune", "grain",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            output_path,
        ]

        try:
            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            for frame in frames:
                if not frame.flags['C_CONTIGUOUS']:
                    frame = np.ascontiguousarray(frame)
                proc.stdin.write(frame.tobytes())
            proc.stdin.close()
            _, stderr = proc.communicate(timeout=300)
            if proc.returncode != 0:
                # Give up on filters entirely
                self._ffmpeg_pipe_encode(frames, output_path, fps, apply_filters=False)
        except Exception as e:
            print(f"[PostProcessor] Fallback FFmpeg error: {e}")

    # ==============================================================
    # VRAM Management — unload between stages
    # ==============================================================

    def _unload_face_models(self):
        if self._face_restorer is not None:
            if hasattr(self._face_restorer, 'to'):
                try:
                    self._face_restorer.to('cpu')
                except Exception:
                    pass
        self._face_restorer = None
        self._face_helper = None
        self._face_mode = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _unload_rife(self):
        self._rife_model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _unload_upscaler(self):
        self._realesrgan_upsampler = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def cleanup(self):
        self._unload_face_models()
        self._unload_rife()
        self._unload_upscaler()
        print("[PostProcessor] All models unloaded")

    def _report(self, callback, pct, status):
        if callback:
            callback(pct, status)

    def get_info(self) -> dict:
        cf = os.path.exists(os.path.join(self.model_dir, "codeformer", "codeformer.pth"))
        gf = os.path.exists(os.path.join(self.model_dir, "gfpgan", "GFPGANv1.4.pth"))
        return {
            "codeformer_available": cf,
            "gfpgan_available": gf,
            "realesrgan_available": os.path.exists(
                os.path.join(self.model_dir, "realesrgan", "RealESRGAN_x4plus.pth")
            ),
            "rife_available": self._check_rife_available(),
            "face_model": "codeformer" if cf else ("gfpgan" if gf else "none"),
            "device": self.device,
        }
