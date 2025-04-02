<p align="center">
  <h1 align="center">WOB: Work of Breathing project</h1>
  <p align="center"> David Plotkin · Daniel Yitzhak </p>
  <div align="center"></div>
</p>

<p align="center">
  Early detection of respiratory distress is critical for timely medical intervention, particularly in
young children. This study presents a non-invasive, video-based approach for assessing respiratory
effort by analyzing depth variations of the torso segmentation. Using a combination of depth
estimation models, segmentation models, and signal processing techniques, we extract chest and
abdomen respiratory rates (Breaths per minute) & phase angle between the signals. The data
undergoes noise reduction, Gaussian-based depth calculations, and signal processing to enhance
signal clarity. This framework demonstrates the potential for automated, contact-free respiratory
assessment, offering a promising tool for early detection and continuous monitoring in clinical and
home environments.
</p>

<p align="center">
  <img src="doll_video.gif" width="400" />
</p>

## Installation

**Environment**

First, clone the repository:
```bash
git clone https://github.com/davidplo4545/WOB.git
```
Then, create the environment, activate it and install dependencies:
```bash
# create the environment
conda wob_env create -f environment.yaml
# activate the environment
conda activate wob_env
# install conda dependencies
pip install -r requirements.txt
```

**Download SAM2 weights**

```bash
# navigate to the folder for the pretrained model
cd third_party/segment-anything-2/checkpoints
# download pretrained model
wget https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt
```


**Download GroundingDINO weights**

```bash
# create the folder for the pretrained model
cd third_party/GroundingDINO
mkdir weights
cd weights
# download pretrained model
wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
```

**Deplying RollingDepth model**  
For installation/running instructions, check [the official repository](https://github.com/prs-eth/rollingdepth).  

Feel free to check out our attached notebook, WOB.ipynb, which demonstrates the stages of our proposed pipeline.


