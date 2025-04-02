import os
ROOT_DIR = os.path.join(os.path.abspath(os.sep), "mnt", "gipnetapp_public", "DnD")

# Output Directories
OUTPUT_DIR = os.path.join(ROOT_DIR, "Outputs")
VIDEOS_DIR = os.path.join(ROOT_DIR, "Videos")
DEPTHS_DIR = os.path.join(OUTPUT_DIR, "Depths")
GRAPHS_DIR = os.path.join(OUTPUT_DIR, "Graphs")
SEG_DIR = os.path.join(OUTPUT_DIR, "Seg")
SAM_SEG_DIR = os.path.join(OUTPUT_DIR, "SAM_Seg")
IMG_DIR = os.path.join(OUTPUT_DIR, "Images")
COMBINED_VIDEOS_DIR = os.path.join(OUTPUT_DIR, "Combined_videos")
ANNOTATIONS_DIR = os.path.join(ROOT_DIR, "Annotations")
RESULTS_DIR = os.path.join(OUTPUT_DIR, "Results")
# Sapiens Roots
SAPIENS_DIR = os.path.join(ROOT_DIR, "Models", "Sapiens")