import cv2
import mediapipe as mp
import numpy as np
import os
import tqdm
from multiprocessing import Pool
import shutil  # 用于复制文件
import glob

# FacePoseFilter 类保持不变，因为它只负责判断逻辑
class FacePoseFilter:
    def __init__(self, 
                 min_face_size=128,      
                 max_yaw=45,             
                 max_pitch=20,           
                 max_roll=30,            
                 detection_rate=0.90,    
                 check_stride=5          
                 ):
        self.min_face_size = min_face_size
        self.max_yaw = max_yaw
        self.max_pitch = max_pitch
        self.max_roll = max_roll
        self.detection_rate = detection_rate
        self.check_stride = check_stride
        
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def _calculate_pose(self, face_landmarks, img_w, img_h):
        face_2d = []
        key_indices = [1, 152, 33, 263, 61, 291]
        for idx in key_indices:
            lm = face_landmarks.landmark[idx]
            x, y = int(lm.x * img_w), int(lm.y * img_h)
            face_2d.append([x, y])
        face_2d = np.array(face_2d, dtype=np.float64)

        model_points = np.array([
            (0.0, 0.0, 0.0), (0.0, -330.0, -65.0), (-225.0, 170.0, -135.0),
            (225.0, 170.0, -135.0), (-150.0, -150.0, -125.0), (150.0, -150.0, -125.0)
        ])

        focal_length = img_w
        cam_matrix = np.array([[focal_length, 0, img_w / 2], [0, focal_length, img_h / 2], [0, 0, 1]])
        dist_matrix = np.zeros((4, 1), dtype=np.float64)

        success, rot_vec, trans_vec = cv2.solvePnP(model_points, face_2d, cam_matrix, dist_matrix, flags=cv2.SOLVEPNP_ITERATIVE)
        if not success:
            return None

        rmat, jac = cv2.Rodrigues(rot_vec)
        angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)
        pitch, yaw, roll = angles[0], angles[1], angles[2]

        # 规范化角度
        if pitch > 90: pitch -= 180
        elif pitch < -90: pitch += 180
        if roll > 90: roll -= 180
        elif roll < -90: roll += 180
        
        # 注意：这里我们不对yaw进行规范化，因为[-180, 180]的范围对判断侧脸更有效
        # 但如果你发现yaw也有跳变问题，可以加上同样的规范化逻辑
        if yaw > 90: yaw -= 180
        elif yaw < -90: yaw += 180

        return pitch, yaw, roll

    def process_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return False, {"reason": "Read Error"}

        valid_frames = 0
        checked_frames = 0
        current_frame = 0
        fail_reasons = {"no_face": 0, "small_face": 0, "bad_pose": 0}
        try:
            while True:
                ret, frame = cap.read()
                if not ret: break
                
                if current_frame % self.check_stride != 0:
                    current_frame += 1
                    continue
                
                checked_frames += 1
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.face_mesh.process(frame_rgb)
                
                is_frame_valid = False
                if results.multi_face_landmarks:
                    face_landmarks = results.multi_face_landmarks[0]
                    h, w, c = frame.shape
                    x_coords = [lm.x for lm in face_landmarks.landmark]
                    y_coords = [lm.y for lm in face_landmarks.landmark]
                    min_x, max_x = min(x_coords) * w, max(x_coords) * w
                    min_y, max_y = min(y_coords) * h, max(y_coords) * h
                    face_w, face_h = max_x - min_x, max_y - min_y
                    
                    if face_w < self.min_face_size or face_h < self.min_face_size:
                        fail_reasons["small_face"] += 1
                    else:
                        pose = self._calculate_pose(face_landmarks, w, h)
                        if pose:
                            pitch, yaw, roll = pose
                            if (abs(pitch) > self.max_pitch or 
                                abs(yaw) > self.max_yaw or 
                                abs(roll) > self.max_roll):
                                fail_reasons["bad_pose"] += 1
                                print(f"Frame {current_frame} Bad Pose: P={pitch:.1f}, Y={yaw:.1f}, R={roll:.1f}")
                            else:
                                is_frame_valid = True
                else:
                    fail_reasons["no_face"] += 1

                if is_frame_valid: valid_frames += 1
                current_frame += 1
                
        except Exception as e:
            return False, {"reason": str(e)}
        finally:
            cap.release()

        if checked_frames == 0:
            return False, {"reason": "Empty video"}

        ratio = valid_frames / checked_frames
        is_pass = ratio >= self.detection_rate
        
        return is_pass, {"ratio": ratio, "stats": fail_reasons}



def gather_paths(input_dir, output_dir):
    """
    递归收集输入目录中的视频文件，并构建对应的输出路径
    """
    paths = []
    for item in sorted(os.listdir(input_dir)):
        item_path = os.path.join(input_dir, item)
        if os.path.isfile(item_path) and item.endswith(".mp4"):
            video_input = item_path
            # 保持相同的文件名和目录结构
            relative_path = os.path.relpath(video_input, input_dir)
            video_output = os.path.join(output_dir, relative_path)
            # 如果输出文件已存在，则跳过
            if not os.path.isfile(video_output):
                paths.append([video_input, video_output])
        elif os.path.isdir(item_path):
            # 递归处理子目录
            sub_input = item_path
            sub_output = os.path.join(output_dir, item)
            paths.extend(gather_paths(sub_input, sub_output))
    return paths


def process_and_copy_video(args):
    """
    多进程包装函数：处理一个视频，如果合格则复制到输出目录
    """
    video_input_path, video_output_path = args
    filter_params = {
    "min_face_size": 120,
    "max_yaw": 20,         # 左右转头不超过40度
    "max_pitch": 20,       # 上下点头不超过30度
    "max_roll": 20,        # 歪头不超过30度
    "detection_rate": 0.9  # 70%的帧合格才通过
    }
    try:
        # 在每个进程中创建新的FacePoseFilter实例
        filter_instance = FacePoseFilter(min_face_size=filter_params["min_face_size"],
                                         max_yaw=filter_params["max_yaw"],
                                         max_pitch=filter_params["max_pitch"],
                                         max_roll=filter_params["max_roll"],
                                         detection_rate=filter_params["detection_rate"])

        is_valid, info = filter_instance.process_video(video_input_path)

        if is_valid:
            # 确保输出目录存在
            os.makedirs(os.path.dirname(video_output_path), exist_ok=True)
            # 复制文件 (使用copy2保留元数据)
            shutil.copy2(video_input_path, video_output_path)
            return video_input_path, True, "Copied"
        else:
            return video_input_path, False, info.get("reason", "Filtered")
            
    except Exception as e:
        return video_input_path, False, f"Process Error: {str(e)}"


def filter_and_copy_videos_multiprocessing(input_dir, output_dir, num_workers):
    """
    使用多进程过滤视频，并将合格的文件复制到新目录，保持原有结构
    """
    print(f"Recursively gathering video paths from {input_dir}...")
    paths = gather_paths(input_dir, output_dir)
    
    if not paths:
        print("No new videos to process (all may already exist in output directory).")
        return

    print(f"Found {len(paths)} videos to process.")
    
    # 准备任务参数
    tasks = [(p_in, p_out) for p_in, p_out in paths]
    
    results = []
    print(f"Filtering and copying videos with {num_workers} workers...")
    with Pool(num_workers) as pool:
        for result in tqdm.tqdm(pool.imap_unordered(process_and_copy_video, tasks), total=len(tasks)):
            results.append(result)
            
    # 统计并打印结果
    passed = sum(1 for _, is_valid, _ in results if is_valid)
    failed = len(results) - passed
    
    print("\nProcessing complete:")
    print(f"Total videos processed: {len(results)}")
    print(f"Passed and copied: {passed}")
    print(f"Failed and filtered: {failed}")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    # --- 配置参数 ---
    input_dir = "/path/to/your/raw_videos"  # 原始视频目录
    output_dir = "/path/to/your/filtered_videos" # 过滤后视频的输出目录
    num_workers = os.cpu_count() # 建议使用CPU核心数
    

    # --- 执行 ---
    filter_and_copy_videos_multiprocessing(input_dir, output_dir, num_workers)