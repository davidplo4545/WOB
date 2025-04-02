from argparse import Namespace
import cv2
import torch.nn.functional as F
import torch

import os
import sys

file_dir = os.path.dirname(os.path.realpath(__file__))
file_dir = os.path.join(file_dir, os.pardir)  # Go one directory back

sys.path.append(os.path.join(file_dir, 'third_party', 'SEA-RAFT'))
sys.path.append(os.path.join(file_dir, 'third_party', 'SEA-RAFT', 'core'))

from raft import RAFT
from utils.flow_viz import flow_to_image
from utils.utils import load_ckpt



class FlowEstimator:
    def __init__(self, cfg, checkpoint_path=None, checkpoint_url=None, device='cpu'):
        """
        Initialize the FlowEstimator with a RAFT model.

        :param cfg: Path to the experiment configuration file.
        :param checkpoint_path: Local path to the model checkpoint.
        :param checkpoint_url: URL to the model checkpoint (if local path not provided).
        :param device: Device for inference ('cpu' or 'cuda').
        """
        # parser = argparse.ArgumentParser()
        # parser.add_argument('--cfg', type=str, default=cfg)
        args = Namespace(name='spring-M', dataset='spring', gpus=[0, 1, 2, 3, 4, 5, 6, 7], use_var=True, var_min=0, var_max=10, pretrain='resnet34', initial_dim=64, block_dims=[64, 128, 256], radius=4, dim=128, num_blocks=2, iters=4, image_size=[540, 960], scale=-1, batch_size=32, epsilon=1e-08, lr=0.0004, wdecay=1e-05, dropout=0, clip=1.0, gamma=0.85, num_steps=120000, restore_ckpt=None, coarse_config=None, cfg='config/eval/spring-M.json', path=None, url='MemorySlices/Tartan-C-T-TSKH-spring540x960-M', device='cpu', corr_levels=4, corr_radius=4, corr_channel=324)
        if args.path is None and args.url is None:
            raise ValueError("Either checkpoint_path or checkpoint_url must be provided")
        
        if args.path is not None:
            self.model = RAFT(args)
            load_ckpt(self.model, args.path)
        else:
            self.model = RAFT.from_pretrained(args.url, args=args)
        
        self.device = torch.device(device)
        self.model = self.model.to(self.device)
        self.model.eval()
        self.args = args

    def _calc_flow(self, image1, image2):
        """
        Calculate the flow and info between two images.

        :param image1: First image as a torch tensor.
        :param image2: Second image as a torch tensor.
        :return: Downscaled flow and info tensors.
        """
        img1 = F.interpolate(image1, scale_factor=2 ** self.args.scale if self.args.scale > 0 else 1, mode='bilinear', align_corners=False)
        img2 = F.interpolate(image2, scale_factor=2 ** self.args.scale if self.args.scale > 0 else 1, mode='bilinear', align_corners=False)
        flow, info = self._forward_flow(img1, img2)
        flow_down = F.interpolate(flow, scale_factor=0.5 ** self.args.scale if self.args.scale > 0 else 1, mode='bilinear', align_corners=False) * (0.5 ** self.args.scale)
        info_down = F.interpolate(info, scale_factor=0.5 ** self.args.scale if self.args.scale > 0 else 1, mode='area')
        return flow_down, info_down

    def _forward_flow(self, image1, image2):
        """
        Forward pass through the model to calculate optical flow.

        :param image1: First image as a torch tensor.
        :param image2: Second image as a torch tensor.
        :return: Flow and info tensors from the model.
        """
        output = self.model(image1, image2, iters=self.args.iters, test_mode=True)
        flow_final = output['flow'][-1]
        info_final = output['info'][-1]
        return flow_final, info_final

    @torch.no_grad()
    def estimate_flow(self, image1, image2):
        """
        Estimate the optical flow between two images.

        :param image1: First image as a numpy array (H, W, C).
        :param image2: Second image as a numpy array (H, W, C).
        :return: Flow visualization image as a numpy array.
        """
        image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
        image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
        image1 = torch.tensor(image1, dtype=torch.float32).permute(2, 0, 1)[None].to(self.device)
        image2 = torch.tensor(image2, dtype=torch.float32).permute(2, 0, 1)[None].to(self.device)
        
        flow, _ = self._calc_flow(image1, image2)
        return flow[0].permute(1, 2, 0).cpu().numpy()
        # flow_vis = flow_to_image(flow[0].permute(1, 2, 0).cpu().numpy(), convert_to_bgr=True)
        # return flow_vis.astype(np.uint8)
