import os
from typing import Optional

import torch

from app.config import settings
from app.logger import log
from ltx_core.loader import LTXV_LORA_COMFY_RENAMING_MAP, LoraPathStrengthAndSDOps
from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
from ltx_pipelines.ti2vid_two_stages import TI2VidTwoStagesPipeline
from ltx_pipelines.utils.constants import AUDIO_SAMPLE_RATE
from ltx_pipelines.utils.media_io import encode_video


class PipelineManager:
    """Manages the LTX video generation pipeline."""

    _instance = None
    _pipeline = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def initialize(self):  # noqa: ANN201
        """Initialize the pipeline with models."""
        if self._pipeline is not None:
            log.info("Pipeline already initialized")
            return

        try:
            log.info("Initializing LTX pipeline...")

            distilled_lora = [
                LoraPathStrengthAndSDOps(
                    settings.distilled_lora,
                    0.6,
                    LTXV_LORA_COMFY_RENAMING_MAP
                ),
            ]

            self._pipeline = TI2VidTwoStagesPipeline(
                checkpoint_path=settings.model_checkpoint,
                distilled_lora=distilled_lora,
                spatial_upsampler_path=settings.spatial_upsampler,
                gemma_root=settings.gemma_root,
                loras=[],
                fp8transformer=True,
                device=settings.device
            )

            log.info("Pipeline initialized successfully")

        except Exception as e:
            log.error(f"Failed to initialize pipeline: {e}")
            raise

    def generate_video(  # noqa: PLR0913
        self,
        prompt: str,
        negative_prompt: Optional[str],
        seed: Optional[int],
        width: int,
        height: int,
        fps: int,
        duration: float,
        num_inference_steps: int,
        cfg_guidance_scale: float,
        output_path: str,
        image_path: Optional[str] = None,
    ) -> str:
        """
        Generate video using the pipeline.

        Args:
            prompt: Text prompt for generation
            negative_prompt: Negative prompt
            seed: Random seed
            width: Video width
            height: Video height
            fps: Frames per second
            duration: Duration in seconds
            num_inference_steps: Number of inference steps
            cfg_guidance_scale: CFG scale
            output_path: Path to save output video
            image_path: Optional path to input image

        Returns:
            Path to generated video
        """
        if self._pipeline is None:
            raise RuntimeError("Pipeline not initialized")

        try:
            log.info(f"Starting video generation with prompt: {prompt[:50]}...")

            # Calculate num_frames from duration and fps
            num_frames = int(duration * fps)

            # Prepare images parameter
            images = []
            if image_path and os.path.exists(image_path):  # noqa: PTH110
                images = [(image_path, 0, 1.0)]
                log.info(f"Using input image: {image_path}")
            else:
                log.info("No input image provided, generating from text only")

            # Get tiling config and video chunks
            tiling_config = TilingConfig.default()
            video_chunks_number = get_video_chunks_number(
                num_frames=num_frames,
                tiling_config=tiling_config
            )

            # Generate video
            video, audio = self._pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt or "Text in video, Bad hand shapes, bad motion, incorrect physics.",
                seed=seed,
                width=width,
                height=height,
                frame_rate=fps,
                num_frames=num_frames,
                tiling_config=tiling_config,
                enhance_prompt=True,
                num_inference_steps=num_inference_steps,
                cfg_guidance_scale=cfg_guidance_scale,
                images=images,  # Empty list [] or [(path, 0, 1.0)]
            )

            # Encode and save video
            with torch.no_grad():
                encode_video(
                    video=video,
                    fps=fps,
                    audio=audio,
                    audio_sample_rate=AUDIO_SAMPLE_RATE,
                    output_path=output_path,
                    video_chunks_number=video_chunks_number
                )

            log.info(f"Video generated successfully: {output_path}")
            return output_path

        except Exception as e:
            log.error(f"Video generation failed: {e}")
            raise

    def cleanup(self):  # noqa: ANN201
        """Cleanup pipeline resources."""
        if self._pipeline is not None:
            log.info("Cleaning up pipeline resources...")
            del self._pipeline
            self._pipeline = None
            torch.cuda.empty_cache()
            log.info("Pipeline cleanup complete")


# Global instance
pipeline_manager = PipelineManager()
