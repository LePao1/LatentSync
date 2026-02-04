import os
import ffmpeg
import multiprocessing

def get_video_duration(video_path):
    """使用ffmpeg获取视频的时长"""
    try:
        # 使用ffmpeg probe命令获取视频信息
        probe = ffmpeg.probe(video_path, v='error', select_streams='v:0', show_entries='stream=duration')
        duration = float(probe['streams'][0]['duration'])  # 获取视频时长（秒）
        return duration
    except ffmpeg.Error as e:
        print(f"Error reading {video_path}: {e}")
        return 0

def short_filter(video_path, short_dir):
    """过滤短时长的视频，保留3秒以上的"""
    duration = get_video_duration(video_path)
    if duration < 3:
        try:
            os.remove(video_path)  # 删除小于3秒的视频
            print(f"Removed short video: {video_path}")
        except Exception as e:
            print(f"Error removing {video_path}: {e}")

def short_filter_multiprocessing(shot_dir, short_dir, total_num_workers):
    """使用多进程过滤短视频"""
    if not os.path.exists(short_dir):
        os.makedirs(short_dir)

    # 获取所有视频文件
    video_files = [os.path.join(shot_dir, f) for f in os.listdir(shot_dir) if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))]

    # 使用 multiprocessing.Pool 来进行多进程处理
    with multiprocessing.Pool(total_num_workers) as pool:
        pool.starmap(short_filter, [(video_path, short_dir) for video_path in video_files])

    print("Short video filtering complete.")