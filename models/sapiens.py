import os
import cv2
import subprocess
import tempfile
import torch
from tqdm import tqdm
import numpy as np
file_dir = os.path.dirname(os.path.realpath(__file__))
file_dir = os.path.join(file_dir, os.pardir)  # Go one directory back
from tools.globals import SAPIENS_DIR, SEG_DIR

class SapiensSegmentation:
    def __init__(self):
        # Constants
        self.SAPIENS_ROOT = os.path.join(file_dir,'third_party','sapiens')
        self.SAPIENS_CHECKPOINT_ROOT = os.path.join(self.SAPIENS_ROOT,'sapiens_lite_host')
        self.MODE = 'torchscript'  # original. no optimizations (slow). full precision inference.
        # self.MODE = 'bfloat16'  # A100 gpus. faster inference at bfloat16
        self.SAPIENS_CHECKPOINT_ROOT = os.path.join(self.SAPIENS_CHECKPOINT_ROOT, self.MODE)
        self.MODEL_NAME = 'sapiens_1b'
        # Set your input and output directories
        self.OUTPUT = os.path.join(SEG_DIR, self.MODEL_NAME)

        self.CHECKPOINT = os.path.join(self.SAPIENS_CHECKPOINT_ROOT,'seg','checkpoints',self.MODEL_NAME,f'{self.MODEL_NAME}_goliath_best_goliath_mIoU_7994_epoch_151_{self.MODE}.pt2')
        # self.CHECKPOINT = os.path.join(self.SAPIENS_CHECKPOINT_ROOT,'seg','checkpoints',self.MODEL_NAME,f'{self.MODEL_NAME}_goliath_best_goliath_mIoU_7673_epoch_194_{self.MODE}.pt2')


    # def load_video_segmentation(self, video_name, rgb_frames, segment_required=False):
    #     depth_frames = []
    #     for i in tqdm(range(len(rgb_frames))):
    #         frame_name = f"{video_name}_frame_{i}"
    #         if segment_required:
    #             self.run_segmentation(rgb_frames[i], video_name, frame_name)
    #         depth = np.load(f"{self.OUTPUT}/{video_name}/seg/{frame_name}_seg.npy") == 21
    #         depth_frames.append(depth)
    #     return depth_frames

    # def run_segmentation(self, image_array, video_name, frame_name):
    #     # Save the input image temporarily
    #     with tempfile.TemporaryDirectory() as temp_dir:
    #         temp_input_path = os.path.join(temp_dir, frame_name + ".jpg")
    #         cv2.imwrite(temp_input_path, image_array)

    #         # Create a temporary text file listing the image
    #         temp_list_path = os.path.join(temp_dir, "input_list.txt")
    #         with open(temp_list_path, 'w') as f:
    #             f.write(temp_input_path + "\n")

    #         # Run the vis_seg.py script
    #         command = [
    #             'python',
    #             f'{self.SAPIENS_ROOT}/lite/demo/vis_seg.py',
    #             self.CHECKPOINT,
    #             '--input', temp_list_path,
    #             '--output-root', os.path.join(self.OUTPUT, str(video_name), "seg"),
    #             '--device', 'cuda:0' if torch.cuda.is_available() else 'cpu',
    #             '--batch-size', '1'
    #         ]
    #         subprocess.run(command)

    #     print("Processing complete.")
    #     print(f"Results saved to {self.OUTPUT}")

    def load_video_segmentation(self, video_name, rgb_frames, segment_required=False):
        if segment_required:
            self.run_segmentation(rgb_frames, video_name)
        # LOAD FRAMES
        depth_frames = []
        for frame_num in range(len(rgb_frames)):
            depth = np.load(f"{self.OUTPUT}/{video_name}/seg/{video_name}_frame_{frame_num}_seg.npy") == 21
            depth_frames.append(depth)
        return depth_frames
        
    def run_segmentation(self, img_arrs, video_name):
        # Save the input image temporarily
        with tempfile.TemporaryDirectory() as temp_dir:
            # Saving all images temporarily
            temp_input_text = ""
            for i, img in enumerate(img_arrs):
                frame_name = f"{video_name}_frame_{i}" + ".jpg"
                temp_input_path = os.path.join(temp_dir, frame_name)
                cv2.imwrite(temp_input_path, img)
                temp_input_text += temp_input_path + "\n"
            print("Done writing video frames to temporary folder")
            # Create a temporary text file listing all the video frames
            temp_list_path = os.path.join(temp_dir, "input_list.txt")
            with open(temp_list_path, 'w') as f:
                f.write(temp_input_text)

            # Run the vis_seg.py script
            command = [
                'python',
                f'{self.SAPIENS_ROOT}/lite/demo/vis_seg.py',
                self.CHECKPOINT,
                '--input', temp_list_path,
                '--output-root', os.path.join(self.OUTPUT, str(video_name), "seg"),
                '--device', 'cuda:0' if torch.cuda.is_available() else 'cpu',
                '--batch-size', '4'
            ]
            subprocess.run(command)

        print("Processing complete.")
        print(f"Results saved to {self.OUTPUT}")

    
class SapiensDepthEstimator:
    def __init__(self):
        # Constants
        self.SAPIENS_ROOT = os.path.join(file_dir,'third_party','sapiens')
        self.SAPIENS_CHECKPOINT_ROOT = os.path.join(self.SAPIENS_ROOT,'sapiens_lite_host')
        self.MODE = 'torchscript'  # original. no optimizations (slow). full precision inference.
        # self.MODE = 'bfloat16'  # A100 gpus. faster inference at bfloat16
        self.SAPIENS_CHECKPOINT_ROOT = os.path.join(self.SAPIENS_CHECKPOINT_ROOT, self.MODE)

        # Set your input and output directories
        self.OUTPUT = os.path.join(self.SAPIENS_ROOT, 'Outputs')
        self.SEG_DIR = os.path.join(self.SAPIENS_ROOT,'Outputs','sapiens_1b')

        # Model card
        self.MODEL_NAME = 'sapiens_1b'
        self.CHECKPOINT = os.path.join(self.SAPIENS_CHECKPOINT_ROOT,'seg','checkpoints',self.MODEL_NAME,f'{self.MODEL_NAME}_goliath_best_goliath_mIoU_7994_epoch_151_{self.MODE}.pt2')
        self.OUTPUT = os.path.join(self.OUTPUT, self.MODEL_NAME)

    def load_video_depth(self, video_name, rgb_frames, segment_required=False):
        depth_frames = []
        for i in tqdm(range(len(rgb_frames))):
            frame_name = f"{video_name}_frame_{i}"
            if segment_required:
                self.run_depth(rgb_frames[i], video_name, frame_name)
            depth = np.load(f"{self.OUTPUT}/{video_name}/depth/{frame_name}.npy")
            depth_frames.append(depth)
        return depth_frames

    def run_depth(self, image_array, video_name, img_name):
        # Save the input image temporarily
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_input_path = os.path.join(temp_dir, img_name + ".jpg")
            cv2.imwrite(temp_input_path, image_array)

            # Create a temporary text file listing the image
            temp_list_path = os.path.join(temp_dir, "input_list.txt")
            with open(temp_list_path, 'w') as f:
                f.write(temp_input_path + "\n")

            command = [
                'python',
                f'{self.SAPIENS_ROOT}/lite/demo/vis_depth.py',
                self.CHECKPOINT,
                '--input', temp_list_path,
                '--output-root', os.path.join(self.OUTPUT, str(video_name), "depth"),
                '--device', 'cuda:0' if torch.cuda.is_available() else 'cpu',
                '--batch-size', '1',
                '--seg_dir', os.path.join(self.SEG_DIR, video_name, "seg")
            ]
            subprocess.run(command)

        print("Processing complete.")
        print(f"Results saved to {self.OUTPUT}")
