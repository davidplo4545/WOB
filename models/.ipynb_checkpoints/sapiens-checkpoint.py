import os
import cv2
import subprocess
import tempfile
import torch

file_dir = os.path.dirname(os.path.realpath(__file__))
file_dir = os.path.join(file_dir, os.pardir)  # Go one directory back


class SapiensSegmentation:
    def __init__(self):
        # Constants
        self.SAPIENS_ROOT = os.path.join(file_dir,'third_party','sapiens')
        self.SAPIENS_CHECKPOINT_ROOT = os.path.join(self.SAPIENS_ROOT,'sapiens_lite_host')
        self.MODE = 'torchscript'  # original. no optimizations (slow). full precision inference.
        # self.MODE = 'bfloat16'  # A100 gpus. faster inference at bfloat16
        self.SAPIENS_CHECKPOINT_ROOT = os.path.join(self.SAPIENS_CHECKPOINT_ROOT, self.MODE)

        # Set your input and output directories
        self.OUTPUT = os.path.join(self.SAPIENS_ROOT,'seg','Outputs','vis','itw_videos','reel1_seg')

        # Model card
        self.MODEL_NAME = 'sapiens_1b'
        self.CHECKPOINT = os.path.join(self.SAPIENS_CHECKPOINT_ROOT,'seg','checkpoints',self.MODEL_NAME,f'{self.MODEL_NAME}_goliath_best_goliath_mIoU_7994_epoch_151_{self.MODE}.pt2')
        self.OUTPUT = os.path.join(self.OUTPUT, self.MODEL_NAME)

    def run_segmentation(self, image_array):
        # Save the input image temporarily
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_input_path = os.path.join(temp_dir, "temp_image.jpg")
            cv2.imwrite(temp_input_path, image_array)

            # Create a temporary text file listing the image
            temp_list_path = os.path.join(temp_dir, "input_list.txt")
            with open(temp_list_path, 'w') as f:
                f.write(temp_input_path + "\n")

            # Run the vis_seg.py script
            command = [
                'python',
                f'{self.SAPIENS_ROOT}/lite/demo/vis_seg.py',
                self.CHECKPOINT,
                '--input', temp_list_path,
                '--output-root', self.OUTPUT,
                '--device', 'cuda:0' if torch.cuda.is_available() else 'cpu',
                '--batch-size', '1'
            ]
            subprocess.run(command)

        print("Processing complete.")
        print(f"Results saved to {self.OUTPUT}")
