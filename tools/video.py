import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from moviepy.editor import ImageSequenceClip, clips_array, concatenate_videoclips
import io
from PIL import Image

def get_folder_name(vid_name):
    folder_name = vid_name[:8]
    vid_num = int(folder_name[5:])
    return vid_num, folder_name

def create_segmented_frames(vid_frames, chest_depths_with_seg, abdomen_depths_with_seg, vid_name, chest_RR, abdomen_RR, PA, annot_chest_RR=None, annot_abdomen_RR=None, annot_PA=None):    
    frames = []
    for i, _ in tqdm(enumerate(vid_frames)):     
        frame = cv2.cvtColor(vid_frames[i],cv2.COLOR_BGR2RGB)

        curr_chest_depth = chest_depths_with_seg[i]
        curr_abdomen_depth = abdomen_depths_with_seg[i]
        curr_chest_depth = np.ma.masked_where(curr_chest_depth == 0, curr_chest_depth)
        curr_abdomen_depth = np.ma.masked_where(curr_abdomen_depth == 0, curr_abdomen_depth)        
        plt.figure(figsize=(10, 6))
        plt.imshow(frame)  # Show image
        plt.imshow(curr_abdomen_depth, cmap="jet", alpha=0.3)  # Overlay depth with colormap
        plt.imshow(curr_chest_depth, cmap="jet", alpha=0.3)  # Overlay depth with colormap
        if annot_chest_RR and annot_abdomen_RR and annot_PA:
            plt.title(f"{vid_name}\nChest RR: {chest_RR:.3f}, Abdomen RR: {abdomen_RR:.3f}, PA: {PA:.3f} \nAnnot.Chest RR: {annot_chest_RR:.3f}, Annot.Abdomen RR: {annot_abdomen_RR:.3f}, Annot.PA: {annot_PA:.3f} ")
        else:
            plt.title(f"{vid_name}\nChest RR: {chest_RR:.3f}, Abdomen RR: {abdomen_RR:.3f}, PA: {PA:.3f}")
        # Remove axes and show the plot
        plt.axis("off")
        # plt.show()
        buf = io.BytesIO()
        plt.savefig(buf, format="jpg")

        buf.seek(0)
        frames.append(Image.open(buf))
        plt.close()
    return frames
    
def load_video_frames(left_video_path, right_video_path, rgb_video_path):
    left_frames = []
    right_frames = []
    rgb_frames = []

    right_fps = 0
    left_fps = 0
    # # Load the left video and get all frames
    # left_cap = cv2.VideoCapture(left_video_path)
    # if not left_cap.isOpened():
    #     print(f"Error: Could not open left video {left_video_path}")
    #     return None, None, None, None
    # left_fps = left_cap.get(cv2.CAP_PROP_FPS)
    # while True:
    #     ret, frame = left_cap.read()
    #     if not ret:
    #         break
    #     left_frames.append(frame)
    # left_cap.release()

    # # Load the right video and get all frames
    # right_cap = cv2.VideoCapture(right_video_path)
    # if not right_cap.isOpened():
    #     print(f"Error: Could not open right video {right_video_path}")
    #     return None, None, None, None
    # right_fps = right_cap.get(cv2.CAP_PROP_FPS)
    # while True:
    #     ret, frame = right_cap.read()
    #     if not ret:
    #         break
    #     right_frames.append(frame)
    # right_cap.release()

    # Load the RGB video and get all frames
    print(rgb_video_path)
    rgb_cap = cv2.VideoCapture(rgb_video_path)
    if not rgb_cap.isOpened():
        print(f"Error: Could not open RGB video {rgb_video_path}")
        return None, None, None, None
    rgb_fps = rgb_cap.get(cv2.CAP_PROP_FPS)
    while True:
        ret, frame = rgb_cap.read()
        if not ret:
            break
        rgb_frames.append(frame)
    rgb_cap.release()

    return left_frames, right_frames, rgb_frames, (left_fps, right_fps, rgb_fps)

def get_video_stats(input_path):
    rgb_cap = cv2.VideoCapture(input_path)
    if not rgb_cap.isOpened():
        print(f"Error: Could not open RGB video {rgb_video_path}")
        exit()
    
    # Get video properties
    rgb_fps = rgb_cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(rgb_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(rgb_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(rgb_cap.get(cv2.CAP_PROP_FRAME_COUNT))

    return rgb_fps, frame_width, frame_height, frame_count
    
def create_video_with_depth(fps, frame_width, frame_height, frames, chest_depths=[], abdomen_depths=[], body_depths=[]):
    # Set up the figure
    fig, ax = plt.subplots()

    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    output_dir = os.path.dirname(os.path.realpath(__file__))
    output_dir = os.path.join(output_dir, os.pardir, "segment_first_output.avi")  # Go one directory back
    print(output_dir)
    out = cv2.VideoWriter(output_dir, fourcc, fps, (frame_width, frame_height))
    
    
    for i in range(len(frames) - 1): 
        frame = cv2.cvtColor(frames[i],cv2.COLOR_BGR2RGB)
        ax.clear()
        ax.imshow(frame)

        if body_depths:
            ax.imshow(body_depths[i], alpha=0.3)
        if chest_depths:
            ax.imshow(chest_depths[i], alpha=0.5)
        if abdomen_depths:
            ax.imshow(abdomen_depths[i], alpha=0.5)
        ax.set_title(f"Frame No.:{i}")
    
        # Save the current frame as an image
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    
        # Resize to match OpenCV output dimensions
        img = cv2.resize(img, (frame_width, frame_height))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR for OpenCV
        out.write(img)
    
    out.release()
    plt.close(fig)
    print("Video saved as output.avi")

def seperate_masks_to_abdomen_and_chest(masks):
    bottom_masks = []
    upper_masks = []
    invalid_indicies = {-1, -2}
    for i, mask in enumerate(masks):
        try:
            # Find the bounding box of the non-zero region
            y_indices, x_indices = np.where(mask > 0)
            # Bounding box coordinates
            y_min, y_max = y_indices.min(), y_indices.max()
            x_min, x_max = x_indices.min(), x_indices.max()
            # Height of the bounding box
            bbox_height = y_max - y_min + 1
            # Calculate the height for the thirds
            # third_height = bbox_height // 5 * 2
            third_height = bbox_height // 3
            # Upper third bounding box coordinates
            upper_y_min = y_min
            upper_y_max = y_min + third_height - 1
            # Bottom third bounding box coordinates
            bottom_y_min = y_max - third_height + 1
            bottom_y_max = y_max
            # Create the upper and bottom third masks
            upper_mask = np.zeros_like(mask)
            bottom_mask = np.zeros_like(mask)
            # Fill in the upper third in the new mask
            upper_mask[upper_y_min:upper_y_max + 1, x_min:x_max + 1] = mask[upper_y_min:upper_y_max + 1, x_min:x_max + 1]
            # Fill in the bottom third in the new mask
            bottom_mask[bottom_y_min:bottom_y_max + 1, x_min:x_max + 1] = mask[bottom_y_min:bottom_y_max + 1, x_min:x_max + 1]
            bottom_masks.append(bottom_mask)
            upper_masks.append(upper_mask)
        except:
            invalid_indicies.add(i)
    return bottom_masks, upper_masks, invalid_indicies

def gaussian_weighted_average_for_abdomen(mask, depth, sigma_x=None, sigma_y=None):
    h, w = mask.shape  # Get mask dimensions
    
    # Get indices where mask is 1
    y_indices, x_indices = np.where(mask == 1)

    if len(x_indices) == 0 or len(y_indices) == 0:
        return 0, np.zeros_like(mask)  # No foreground pixels

    # Compute bounding box of mask pixels
    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices)

    # Compute Gaussian mean
    mu_x = (x_min + x_max) / 2
    mu_y = y_min + (y_max - y_min) * 2 / 3   # 1/3 from the top of available region


    # Define Gaussian standard deviations (σ)
    if sigma_x is None:
        sigma_x = w / 6  # Default: 1/6th of the width
    if sigma_y is None:
        sigma_y = h / 6  # Default: 1/6th of the height

    # Create coordinate grids
    x = np.arange(w)
    y = np.arange(h)
    X, Y = np.meshgrid(x, y)

    # Compute the 2D Gaussian weights
    gaussian_weights = np.exp(-(((X - mu_x) ** 2) / (2 * sigma_x ** 2) +
                                 ((Y - mu_y) ** 2) / (2 * sigma_y ** 2)))
    # Compute the weighted average
    weighted_sum = np.sum(depth * gaussian_weights)
    normalization = np.sum(gaussian_weights)
    
    weighted_avg = weighted_sum / normalization if normalization != 0 else 0
    return weighted_avg, gaussian_weights


def gaussian_weighted_average_for_chest(mask, depth, sigma_x=None, sigma_y=None):
    h, w = mask.shape  # Get mask dimensions
    
    # Get indices where mask is 1
    y_indices, x_indices = np.where(mask == 1)

    if len(x_indices) == 0 or len(y_indices) == 0:
        return 0, np.zeros_like(mask)  # No foreground pixels

    # Compute bounding box of mask pixels
    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices)
    # Compute Gaussian mean
    mu_x_left = x_min + (x_max - x_min) / 8 
    mu_x_right = x_min + (x_max - x_min) * 7 / 8
    mu_y = y_min + (y_max - y_min) / 2   # 1/2 from the top of available region


    # Define Gaussian standard deviations (σ)
    if sigma_x is None:
        sigma_x = h / 6  # Default: 1/6th of the height
    if sigma_y is None:
        sigma_y = h / 6  # Default: 1/6th of the height

    # Create coordinate grids
    x = np.arange(w)
    y = np.arange(h)
    X, Y = np.meshgrid(x, y)

    # Compute the 2D Gaussian weights
    gaussian_weights_left = np.exp(-(((X - mu_x_left) ** 2) / (2 * sigma_x ** 2) +
                                 ((Y - mu_y) ** 2) / (2 * sigma_y ** 2)))

    gaussian_weights_right = np.exp(-(((X - mu_x_right) ** 2) / (2 * sigma_x ** 2) +
                                 ((Y - mu_y) ** 2) / (2 * sigma_y ** 2)))


    gaussian_weights = gaussian_weights_left + gaussian_weights_right
    # Compute the weighted average
    weighted_sum = np.sum(gaussian_weights * depth)
    normalization = np.sum(gaussian_weights)
    
    weighted_avg = weighted_sum / normalization if normalization != 0 else 0
    return weighted_avg, gaussian_weights

def apply_masks_erosion(abdomen_masks, chest_masks):
    kernel = np.ones((3, 3), np.uint8)  # 3x3 kernel
    for i, mask in enumerate(abdomen_masks):
        abdomen_masks[i] = cv2.erode(mask, kernel, iterations=3)
    for i, mask in enumerate(chest_masks):
        chest_masks[i] = cv2.erode(mask, kernel, iterations=3)
    return abdomen_masks, chest_masks

def get_masked_depths(erosion_masks, no_erosion_masks, depths, is_chest):
    # TODO: REDUCE PARAMETERS LATER FOR FASTER PROCESSING (AFTER CREATING VIDEOS)
    result_depths = []
    result_depths_avg = []
    results_depths_no_erosion = []
    result_plot_depths = []
    
    mean_without_zero = lambda x: np.nanmedian(np.where(x==0,np.nan,x))
    for i, depth in tqdm(enumerate(depths)):
        abs_depth = depths[i]
        # NOTE: While using Optical Flow model
        # abs_depth = np.linalg.norm(depth,axis=2)
        
        curr_depth_erosion = abs_depth*erosion_masks[i]
        curr_depth_no_erosion = abs_depth*no_erosion_masks[i]

        if is_chest:
            result_depths.append(gaussian_weighted_average_for_chest(erosion_masks[i], curr_depth_erosion)[0])
            results_depths_no_erosion.append(gaussian_weighted_average_for_chest(no_erosion_masks[i], curr_depth_erosion)[0])
        else:
            result_depths.append(gaussian_weighted_average_for_abdomen(erosion_masks[i], curr_depth_erosion)[0])
            results_depths_no_erosion.append(gaussian_weighted_average_for_abdomen(no_erosion_masks[i], curr_depth_erosion)[0])
        
        result_depths_avg.append(mean_without_zero(curr_depth_erosion))
        result_plot_depths.append(np.where(curr_depth_erosion==0,np.nan,curr_depth_erosion))
        
    return result_depths, result_depths_avg, results_depths_no_erosion, result_plot_depths

def resize_frames(frames, target_height=720, target_width=1280):
    frames = frames.astype(np.float32)

    resized_frames = []
    for i, depth in tqdm(enumerate(frames)):

        resized_frame = cv2.resize(frames[i], (target_width, target_height), interpolation=cv2.INTER_LINEAR)
        resized_frames.append(resized_frame)
    result_frames = np.stack(resized_frames)
    return result_frames

def create_side_by_side_video(img_folder, buffer_frames, output_path, frame_start, frame_end, fps=24, seg_frames = []):
    # Get sorted lists of PNG files from both folders
    images = []
    if not seg_frames:
        images = sorted([os.path.join(img_folder, f) for f in os.listdir(img_folder) if f.endswith('.png') or f.endswith('.jpg')] )
        images = images[frame_start:frame_end + 1]
    else:
        images = seg_frames
        images = [np.array(frame) for frame in seg_frames]
    # images3 = sorted([os.path.join(folder3, f) for f in os.listdir(folder3) if f.endswith('.png') or f.endswith('.jpg')])
    buffer_frames = [np.array(frame) for frame in buffer_frames]

    
    # Check if both folders have the same number of images
    if len(images) != len(buffer_frames):
        raise ValueError(f"Both folders must contain the same number of PNG files. Image Frames:{len(images)}, Buffer Frames:{len(buffer_frames)}")

    top_clip = ImageSequenceClip(images, fps=fps)
    bottom_clip = ImageSequenceClip(buffer_frames, fps=fps)
    
    # Combine clips with the blank clip in place of None
    combined_clip = clips_array([
        [top_clip],
        [bottom_clip]
    ])

    # Write the video to the output file
    combined_clip.write_videofile(output_path, codec="libx264", fps=fps)



        

    