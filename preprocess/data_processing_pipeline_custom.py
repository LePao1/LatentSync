import argparse
import os
# from preprocess.affine_transform import affine_transform_multi_gpus
from preprocess.remove_broken_videos import remove_broken_videos_multiprocessing
from preprocess.detect_shot import detect_shot_multiprocessing
# from preprocess.filter_high_resolution import filter_high_resolution_multiprocessing
from preprocess.resample_fps_hz import resample_fps_hz_multiprocessing
# from preprocess.segment_videos import segment_videos_multiprocessing
from preprocess.sync_av import sync_av_multi_gpus
from preprocess.filter_visual_quality import filter_visual_quality_multi_gpus
# from preprocess.remove_incorrect_affined import remove_incorrect_affined_multiprocessing
from preprocess.face_pose_filter import filter_and_copy_videos_multiprocessing
from preprocess.short_filter import short_filter_multiprocessing
import subprocess
from pathlib import Path

def check_model_and_download(ckpt_path: str, huggingface_model_id: str = "ByteDance/LatentSync-1.5"):
    if not os.path.exists(ckpt_path):
        ckpt_path_obj = Path(ckpt_path)
        download_cmd = f"huggingface-cli download {huggingface_model_id} {Path(*ckpt_path_obj.parts[1:])} --local-dir {Path(ckpt_path_obj.parts[0])}"
        subprocess.run(download_cmd, shell=True)


def data_processing_pipeline(
    total_num_workers, per_gpu_num_workers, resolution, sync_conf_threshold, temp_dir, input_dir
):
    print("Checking models are downloaded...")
    check_model_and_download("checkpoints/auxiliary/syncnet_v2.model")
    check_model_and_download("checkpoints/auxiliary/sfd_face.pth")
    check_model_and_download("checkpoints/auxiliary/koniq_pretrained.pkl")

    print("Removing broken videos...")
    remove_broken_videos_multiprocessing(input_dir, total_num_workers)

    print("Resampling FPS hz...")
    resampled_dir = os.path.join(os.path.dirname(input_dir), "resampled")
    resample_fps_hz_multiprocessing(input_dir, resampled_dir, total_num_workers)

    print("Detecting shot...")
    shot_dir = os.path.join(os.path.dirname(input_dir), "shot")
    detect_shot_multiprocessing(resampled_dir, shot_dir, total_num_workers)

    print("Filtering short...")
    short_filter_multiprocessing(shot_dir, shot_dir, total_num_workers)

    print("Filtering pose...")
    pose_dir = os.path.join(os.path.dirname(input_dir), "pose")
    filter_and_copy_videos_multiprocessing(shot_dir, pose_dir, total_num_workers)

    print("Syncing audio and video...")
    av_synced_dir = os.path.join(os.path.dirname(input_dir), f"av_synced_{sync_conf_threshold}")
    print('av_synced_dir',av_synced_dir)
    sync_av_multi_gpus(pose_dir, av_synced_dir, temp_dir, per_gpu_num_workers, sync_conf_threshold)

    print("Filtering visual quality...")
    high_visual_quality_dir = os.path.join(os.path.dirname(input_dir), "high_visual_quality")
    filter_visual_quality_multi_gpus(av_synced_dir, high_visual_quality_dir, per_gpu_num_workers)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--total_num_workers", type=int, default=100)
    parser.add_argument("--per_gpu_num_workers", type=int, default=20)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--sync_conf_threshold", type=int, default=3)
    parser.add_argument("--temp_dir", type=str, default="temp")
    parser.add_argument("--input_dir", type=str, required=True)
    args = parser.parse_args()

    data_processing_pipeline(
        args.total_num_workers,
        args.per_gpu_num_workers,
        args.resolution,
        args.sync_conf_threshold,
        args.temp_dir,
        args.input_dir,
    )
