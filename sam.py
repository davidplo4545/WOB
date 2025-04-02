# =============================================================================
#  Imports
# =============================================================================


import numpy as np
import torch
from torchvision.ops import box_convert
import os
from tqdm import tqdm
import cv2
from PIL import Image
import matplotlib.pyplot as plt
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'third_party', 'segment-anything-2')))
from sam2.sam2_video_predictor import SAM2VideoPredictor
from sam2.build_sam import build_sam2_video_predictor
import argparse
import sys
import glob
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'third_party', 'GroundingDINO')))
import groundingdino.util.inference as GD

def extract_video(video_number, video_file, images_dir):
    # Extract frames from the video
    cap = cv2.VideoCapture(video_file)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    for i in tqdm(range(frame_count), desc='Extracting frames from video'):
        ret, frame = cap.read()
        if not ret:
            break
        frame_filename = os.path.join(images_dir, f'{i:04}.jpg')
        cv2.imwrite(frame_filename, frame)

    cap.release()

    return fps

def init_models(images_dir, use_local_sam2_weights, device='cuda'):
    # Initialize models
    SAM2_predictor = None
    if use_local_sam2_weights:
        SAM2_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'third_party', 'segment-anything-2'))
        checkpoint = os.path.join(SAM2_dir, 'checkpoints', 'sam2_hiera_large.pt')
        model_cfg = 'sam2_hiera_l.yaml'
        cfg_path = os.path.join(SAM2_dir, 'sam2_configs')
        SAM2_predictor = build_sam2_video_predictor(model_cfg, config_path=cfg_path, ckpt_path=checkpoint, device=device)
    else:
        SAM2_predictor = SAM2VideoPredictor.from_pretrained("facebook/sam2-hiera-large", device='cuda')
    inference_state = SAM2_predictor.init_state(video_path=images_dir)

    GD_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'third_party', 'GroundingDINO'))
    GD_model = GD.load_model(
        os.path.join(GD_dir, 'groundingdino', 'config', 'GroundingDINO_SwinT_OGC.py'),
        os.path.join(GD_dir, 'weights', 'groundingdino_swint_ogc.pth')
    )

    return GD_model, SAM2_predictor, inference_state

class Masker:
    def __init__(self, GD_model, predictor, inference_state, images_dir, output_dir, image_number=0, prompt='torso'):
        """
        Initialize the Masker class.

        Parameters:
        GD_model (GroundingDINO): GroundingDINO model class.
        predictor (SAM2VideoPredictor): SAM2 predictor.
        predictor (dict): SAM2 inference state.
        images_dir (str): Directory of jpg images for SAM2.
        image_number (int): Base image number from which to start masking.
        """

        self.GD_model = GD_model

        self.predictor = predictor
        self.inference_state = inference_state
        self.predictor.reset_state(self.inference_state)
        self.output_dir = output_dir
        self.images_dir = images_dir

        self.index = image_number

        self.masks = {}
        self.masked_images = {}

        self.image_filenames = [
            p for p in os.listdir(images_dir)
            if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
        ]
        self.image_filenames.sort(key=lambda p: int(os.path.splitext(p)[0]))
        image_filename = os.path.join(images_dir, self.image_filenames[self.index])

        self.points = []
        self.bboxes = self.get_GroundingDINO_bbox(image_filename, prompt)

        self.predictor.reset_state(self.inference_state)
        points = np.array([point[0] for point in self.points]) if len(self.points) > 0 else None
        point_labels = np.array([point[1] for point in self.points]) if len(self.points) > 0 else None
        bbox = self.bboxes

        _, out_obj_ids, out_mask_logits = self.predictor.add_new_points_or_box(
            inference_state=self.inference_state,
            frame_idx=self.index,
            obj_id=1,
            points=points,
            labels=point_labels,
            box=bbox
        )
        self.masks[self.index] = (out_mask_logits[0] > 0.0).cpu().numpy().squeeze(0)
        self.save_mask()

    def get_GroundingDINO_bbox(self, image_path, prompt):
        """
        use GroundingDINO model to get a bounding box from an image and a prompt.

        Parameters:
        image_path (str): the path to the image file.
        prompt (str): the prompt of the target object

        Returns:
        np.ndarray: an array containing the bounding box
        """
        IMAGE_PATH = image_path
        TEXT_PROMPT = prompt
        BOX_TRESHOLD = 0.35
        TEXT_TRESHOLD = 0.25
        
        image_source, image = GD.load_image(IMAGE_PATH)
        boxes, logits, phrases = GD.predict(
            model=self.GD_model,
            image=image,
            caption=TEXT_PROMPT,
            box_threshold=BOX_TRESHOLD,
            text_threshold=TEXT_TRESHOLD
        )
        
        h, w, _ = image_source.shape
        boxes = boxes * torch.Tensor([w, h, w, h])

        return box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()[0]
    
    def save_mask(self):
        image_filename = os.path.join(self.images_dir, self.image_filenames[self.index])
        image = cv2.imread(image_filename)
        mask = self.masks[self.index]
        masked_image_filename = os.path.join(self.output_dir, f'{self.index:03}_masked_image.png')

        alpha = 0.5  # Transparency factor
        mask_colored = np.zeros_like(image)
        mask_colored[mask == 1] = [0, 0, 255]  # Color the mask red
        masked_image = cv2.addWeighted(image, 1, mask_colored, alpha, 0)
        self.masked_images[self.index] = masked_image
        cv2.imwrite(masked_image_filename, masked_image)
        np.save(os.path.join(self.output_dir, f'{self.index:03}_mask.npy'), mask)


    def segment(self):
        """
        Propagate the mask throughout the video using SAM2 and save the resulting masks.

        Returns:
        None
        """
        for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(self.inference_state):
            self.index = out_frame_idx
            mask = (out_mask_logits[0] > 0.0).cpu().numpy().squeeze(0)
            self.masks[self.index] = mask
            self.save_mask()

        plt.close('all')
        
    def create_video_from_masks(self, video_number, fps=30.0):
        """
        Create a video from the masked images.

        Returns:
        None
        """
        height, width, _ = self.masked_images[0].shape
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(os.path.join(self.output_dir, f'WOBI_{video_number:03}_masked_video.avi'), fourcc, fps, (width, height))

        for i in range(len(self.masked_images)):
            out.write(self.masked_images[i].astype(np.uint8))

        out.release()

def get_video_segmentation(video_number, video_file, images_dir, output_dir, prompt="torso", device="cuda"):

    # Parameters
    # video_number = 17
    # prompt = 'torso'
    use_local_sam2_weights = True

    # Create directories
    # video_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), f'WOBI_{video_number:03}'))
    # video_dir = os.path.abspath(os.path.dirname(__file__))
    # images_dir = os.path.join(video_dir, 'images_jpg')
    # output_dir = os.path.join(video_dir, 'masked_images')
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # Extract frames from the video
    print(video_number, video_file, images_dir)
    video_fps = extract_video(video_number, video_file, images_dir)

    # Initialize models
    GD_model, SAM2_predictor, inference_state = init_models(images_dir, use_local_sam2_weights=use_local_sam2_weights, device=device) 

    # Initialize Masker
    masker = Masker(GD_model=GD_model, predictor=SAM2_predictor, inference_state=inference_state, images_dir=images_dir, output_dir=output_dir, image_number=0, prompt=prompt)

    # Propagate masks
    masker.segment()

    # Create video from masked images
    # masker.create_video_from_masks(video_number, fps=video_fps)

def load_video_segmentation(video_name, rgb_frames):
    # LOAD FRAMES
    seg_frames = []
    for frame_num in range(len(rgb_frames)):
        # depth = np.load(f"{self.OUTPUT}/{video_name}/seg/{video_name}_frame_{frame_num}_seg.npy") == 21
        seg = np.load(f"masked_images/{frame_num:03}_mask.npy")
        print(seg)
        seg_frames.append(seg)
    return seg_frames

