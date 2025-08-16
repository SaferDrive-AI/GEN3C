# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
import torch
import numpy as np
from cosmos_predict1.diffusion.inference.inference_utils import (
    add_common_arguments,
)
from cosmos_predict1.diffusion.inference.gen3c_pipeline import Gen3cPipeline
from cosmos_predict1.utils import log, misc
from cosmos_predict1.utils.io import read_prompts_from_file, save_video
from cosmos_predict1.diffusion.inference.cache_3d import Cache4D
from cosmos_predict1.diffusion.inference.camera_utils import generate_frame_trajectory
from cosmos_predict1.diffusion.inference.data_loader_utils import load_data_auto_detect
import torch.nn.functional as F
torch.enable_grad(False)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Video to video viewpoint transformation")
    # Add common arguments
    add_common_arguments(parser)

    parser.add_argument(
        "--prompt_upsampler_dir",
        type=str,
        default="Pixtral-12B",
        help="Prompt upsampler weights directory relative to checkpoint_dir",
    )
    parser.add_argument(
        "--input_video_path",
        type=str,
        required=True,
        help="Input video path (ViPE processed data directory or .pt file)",
    )
    parser.add_argument(
        "--trajectory",
        type=str,
        choices=[
            "left",
            "right", 
            "up",
            "down",
            "zoom_in",
            "zoom_out",
            "clockwise",
            "counterclockwise",
        ],
        default="left",
        help="Select a trajectory type from the available options",
    )
    parser.add_argument(
        "--camera_rotation",
        type=str,
        choices=["center_facing", "no_rotation", "trajectory_aligned"],
        default="no_rotation",
        help="Controls camera rotation during movement",
    )
    parser.add_argument(
        "--movement_distance",
        type=float,
        default=0.3,
        help="Distance of the camera movement",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=120,
        help="Maximum number of frames to process",
    )
    parser.add_argument(
        "--save_buffer",
        action="store_true",
        help="If set, save the warped images (buffer) side by side with the output video.",
    )
    parser.add_argument(
        "--filter_points_threshold",
        type=float,
        default=0.05,
        help="If set, filter the points continuity of the warped images.",
    )
    parser.add_argument(
        "--foreground_masking",
        action="store_true",
        help="If set, use foreground masking for the warped images.",
    )
    return parser.parse_args()


def validate_args(args):
    assert args.num_video_frames is not None, "num_video_frames must be provided"
    assert (args.num_video_frames - 1) % 120 == 0, "num_video_frames must be 121, 241, 361, ... (N*120+1)"
    assert args.max_frames <= 120, "max_frames should not exceed 120 for this implementation"




def demo(args):
    """Run video-to-video viewpoint transformation."""
    misc.set_random_seed(args.seed)
    inference_type = "video2world"
    validate_args(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    log.info(f"Processing video with viewpoint transformation: {args.trajectory}")
    log.info(f"Input video path: {args.input_video_path}")

    if args.num_gpus > 1:
        from megatron.core import parallel_state
        from cosmos_predict1.utils import distributed

        distributed.init()
        parallel_state.initialize_model_parallel(context_parallel_size=args.num_gpus)
        process_group = parallel_state.get_context_parallel_group()

    # Initialize video2world generation model pipeline
    pipeline = Gen3cPipeline(
        inference_type=inference_type,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_name="Gen3C-Cosmos-7B",
        prompt_upsampler_dir=args.prompt_upsampler_dir,
        enable_prompt_upsampler=not args.disable_prompt_upsampler,
        offload_network=args.offload_diffusion_transformer,
        offload_tokenizer=args.offload_tokenizer,
        offload_text_encoder_model=args.offload_text_encoder_model,
        offload_prompt_upsampler=args.offload_prompt_upsampler,
        offload_guardrail_models=args.offload_guardrail_models,
        disable_guardrail=args.disable_guardrail,
        disable_prompt_encoder=args.disable_prompt_encoder,
        guidance=args.guidance,
        num_steps=args.num_steps,
        height=args.height,
        width=args.width,
        fps=args.fps,
        num_video_frames=121,
        seed=args.seed,
    )

    if args.num_gpus > 1:
        pipeline.model.net.enable_context_parallel(process_group)

    # Handle multiple prompts if prompt file is provided
    if args.batch_input_path:
        log.info(f"Reading batch inputs from path: {args.batch_input_path}")
        prompts = read_prompts_from_file(args.batch_input_path)
    else:
        # Single prompt case
        prompts = [{"prompt": args.prompt, "visual_input": args.input_video_path}]

    os.makedirs(os.path.dirname(args.video_save_folder), exist_ok=True)
    
    for i, input_dict in enumerate(prompts):
        current_prompt = input_dict.get("prompt", None)
        if current_prompt is None and args.disable_prompt_upsampler:
            log.critical("Prompt is missing, skipping world generation.")
            continue
        current_video_path = input_dict.get("visual_input", None)
        if current_video_path is None:
            log.critical("Visual input is missing, skipping world generation.")
            continue

        # Load data using the auto-detect loader
        try:
            (
                image_fchw_float,     # [F, C, H, W]
                depth_f1hw,           # [F, 1, H, W] 
                mask_f1hw,            # [F, 1, H, W]
                w2c_f44,              # [F, 4, 4]
                intrinsics_f33,       # [F, 3, 3]
            ) = load_data_auto_detect(current_video_path)
        except Exception as e:
            log.critical(f"Failed to load visual input from {current_video_path}: {e}")
            continue

        # Limit to max_frames
        num_frames = min(image_fchw_float.shape[0], args.max_frames)
        log.info(f"Processing {num_frames} frames from video")
        
        image_fchw_float = image_fchw_float[:num_frames].to(device)
        depth_f1hw = depth_f1hw[:num_frames].to(device)
        mask_f1hw = mask_f1hw[:num_frames].to(device)
        w2c_f44 = w2c_f44[:num_frames].to(device)
        intrinsics_f33 = intrinsics_f33[:num_frames].to(device)

        # Initialize Cache4D with complete video data
        cache = Cache4D(
            input_image=image_fchw_float,        # [F, C, H, W]
            input_depth=depth_f1hw,              # [F, 1, H, W]
            input_mask=mask_f1hw,                # [F, 1, H, W]
            input_w2c=w2c_f44,                   # [F, 4, 4]
            input_intrinsics=intrinsics_f33,     # [F, 3, 3]
            filter_points_threshold=args.filter_points_threshold,
            input_format=["F", "C", "H", "W"],
            foreground_masking=args.foreground_masking,
        )

        # Generate new camera trajectory for each frame
        log.info(f"Generating new camera trajectory: {args.trajectory}")
        generated_w2cs_list = []
        generated_intrinsics_list = []

        for frame_idx in range(num_frames):
            frame_new_w2c, frame_new_intrinsics = generate_frame_trajectory(
                original_w2c=w2c_f44[frame_idx],
                original_intrinsics=intrinsics_f33[frame_idx],
                trajectory_type=args.trajectory,
                movement_distance=args.movement_distance,
                camera_rotation=args.camera_rotation,
                device=device,
            )
            generated_w2cs_list.append(frame_new_w2c)
            generated_intrinsics_list.append(frame_new_intrinsics)

        # Stack into batch format
        generated_w2cs = torch.stack(generated_w2cs_list).unsqueeze(0)  # [1, F, 4, 4]
        generated_intrinsics = torch.stack(generated_intrinsics_list).unsqueeze(0)  # [1, F, 3, 3]

        log.info(f"Rendering warped images for {num_frames} frames")
        
        # Render warped images for all frames at once
        rendered_warp_images, rendered_warp_masks = cache.render_cache(
            generated_w2cs,
            generated_intrinsics,
            start_frame_idx=0,
        )

        all_rendered_warps = []
        if args.save_buffer:
            all_rendered_warps.append(rendered_warp_images.clone().cpu())

        log.info("Generating new viewpoint video...")
        
        # Generate video using complete frame sequence
        # Note: We need to ensure the input format matches what pipeline.generate expects
        input_video_batch = image_fchw_float.unsqueeze(0).unsqueeze(2)  # [1, F, 1, C, H, W] -> [1, C, F, H, W]
        input_video_batch = input_video_batch.squeeze(2).permute(0, 2, 1, 3, 4)  # [1, F, C, H, W]
        
        generated_output = pipeline.generate(
            prompt=current_prompt,
            image_path=input_video_batch,
            negative_prompt=args.negative_prompt,
            rendered_warp_images=rendered_warp_images,
            rendered_warp_masks=rendered_warp_masks,
        )
        
        if generated_output is None:
            log.critical("Guardrail blocked video2world generation.")
            continue
        video, prompt = generated_output

        # Final video processing
        final_video_to_save = video
        final_width = args.width

        if args.save_buffer and all_rendered_warps:
            # Process and concatenate buffer visualization
            # This code is adapted from the original implementation
            squeezed_warps = [t.squeeze(0) for t in all_rendered_warps]
            
            if squeezed_warps:
                n_max = max(t.shape[1] for t in squeezed_warps)
                
                padded_t_list = []
                for sq_t in squeezed_warps:
                    current_n_i = sq_t.shape[1]
                    padding_needed_dim1 = n_max - current_n_i
                    
                    pad_spec = (0,0, 0,0, 0,0, 0,padding_needed_dim1, 0,0)
                    padded_t = F.pad(sq_t, pad_spec, mode='constant', value=-1.0)
                    padded_t_list.append(padded_t)
                
                full_rendered_warp_tensor = torch.cat(padded_t_list, dim=0)
                
                T_total, _, C_dim, H_dim, W_dim = full_rendered_warp_tensor.shape
                buffer_video_TCHnW = full_rendered_warp_tensor.permute(0, 2, 3, 1, 4)
                buffer_video_TCHWstacked = buffer_video_TCHnW.contiguous().view(T_total, C_dim, H_dim, n_max * W_dim)
                buffer_video_TCHWstacked = (buffer_video_TCHWstacked * 0.5 + 0.5) * 255.0
                buffer_numpy_TCHWstacked = buffer_video_TCHWstacked.cpu().numpy().astype(np.uint8)
                buffer_numpy_THWC = np.transpose(buffer_numpy_TCHWstacked, (0, 2, 3, 1))
                
                final_video_to_save = np.concatenate([buffer_numpy_THWC, final_video_to_save], axis=2)
                final_width = args.width * (1 + n_max)
                log.info(f"Concatenating video with {n_max} warp buffers. Final video width will be {final_width}")

        # Save video
        if args.batch_input_path:
            video_save_path = os.path.join(args.video_save_folder, f"{i}.mp4")
        else:
            video_save_path = os.path.join(args.video_save_folder, f"{args.video_save_name}.mp4")

        os.makedirs(os.path.dirname(video_save_path), exist_ok=True)

        save_video(
            video=final_video_to_save,
            fps=args.fps,
            H=args.height,
            W=final_width,
            video_save_quality=5,
            video_save_path=video_save_path,
        )
        log.info(f"Saved video to {video_save_path}")

    # Clean up properly
    if args.num_gpus > 1:
        parallel_state.destroy_model_parallel()
        import torch.distributed as dist
        dist.destroy_process_group()


if __name__ == "__main__":
    args = parse_arguments()
    if args.prompt is None:
        args.prompt = ""
    args.disable_guardrail = True
    args.disable_prompt_upsampler = True
    demo(args)