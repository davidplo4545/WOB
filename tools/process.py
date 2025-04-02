import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import io
from PIL import Image
import csv
import json
from .VideoData import VideoData
from .globals import GRAPHS_DIR, VIDEOS_DIR, SAM_SEG_DIR, DEPTHS_DIR, COMBINED_VIDEOS_DIR, IMG_DIR, RESULTS_DIR, ANNOTATIONS_DIR, ROOT_DIR
from .video import load_video_frames, resize_frames, get_folder_name, seperate_masks_to_abdomen_and_chest, get_masked_depths, get_video_stats, create_side_by_side_video, apply_masks_erosion, create_segmented_frames
from tools.signals import SignalSmoother, calc_bpm_and_phase_shift_fft, fit_sinus, generate_wave_data

class Pipeline():
    def __init__(self, file_name, start_time=0, end_time=None, has_annot = False):
        self.file_name = file_name
        # _, self.folder_name = get_folder_name(self.file_name) # Uncomment later - not for dolls
        self.folder_name = self.file_name.split('.avi')[0]
        self.full_video_path = os.path.join(VIDEOS_DIR,"doll", self.file_name) # DOLLS PATH
        # self.full_video_path = os.path.join(VIDEOS_DIR, self.file_name) # Regular PATH

        # self.fps, self.frame_width, self.frame_height, self.frames_count = get_video_stats(self.full_video_path)
        self.depths = None
        
        _, _, self.rgb_frames, (_, _, self.fps) = load_video_frames("", "", self.full_video_path)
        self.start_frame = int(start_time * self.fps)
        
        if end_time:
            self.end_frame = int(end_time * self.fps)
        else:
            self.end_frame = len(self.rgb_frames)

        self.frames_count = len(self.rgb_frames)

        self.rgb_frames = self.rgb_frames[self.start_frame: self.end_frame + 1]
        self.segs = []
        
        self.annot_json_data = None
        if has_annot:
            self.get_annotation_json_data()
            
    def get_annotation_json_data(self):
        annot_dir = os.path.join(ANNOTATIONS_DIR, f"{self.file_name.split('.avi')[0]}_annotations.json")

        with open(annot_dir) as f:
            self.annot_json_data = json.load(f)

        # Sort frames by ascending order
        self.annot_json_data["chest_peaks"] = sorted(self.annot_json_data["chest_peaks"], key=lambda x: x["frame_number"])
        self.annot_json_data["abdomen_peaks"] = sorted(self.annot_json_data["abdomen_peaks"], key=lambda x: x["frame_number"])
        
        # Filter out by start and end frames
        self.annot_json_data["chest_peaks"] = [peak for peak in self.annot_json_data['chest_peaks'] if peak['frame_number'] <= self.end_frame and peak['frame_number'] >= self.start_frame]
        self.annot_json_data["abdomen_peaks"] = [peak for peak in self.annot_json_data['abdomen_peaks'] if peak['frame_number'] <= self.end_frame and peak['frame_number'] >= self.start_frame]
                    
    def get_npy_file_name(self):
        # return self.file_name.split('.')[0]+'_pred.npy'
        return self.file_name.split('.avi')[0]+'_pred.npy' # DOLL
        
    def load_depths(self):
        depths = np.load(os.path.join(DEPTHS_DIR, "doll",self.get_npy_file_name())) # DOLLS PATH
        # depths = np.load(os.path.join(DEPTHS_DIR,self.get_npy_file_name())) # REGULAR PATH
        print("Loading depths complete")
        self.depths = resize_frames(depths)[self.start_frame: self.end_frame + 1]
        
        print("Resizing depths complete")
        return self.depths
        
    def load_video_segmentation(self):
        print("Loading Segmentation...")
        seg_frames = []

        for frame_num in range(self.frames_count):
            seg = np.load(f"{SAM_SEG_DIR}/{self.folder_name}/{frame_num:03}_mask.npy")
            self.segs.append(seg)
        self.segs = self.segs[self.start_frame: self.end_frame + 1]
        self.segs = np.array(self.segs)
        self.segs = resize_frames(self.segs)
        print("Loading Segmentation complete.")
        return self.segs


    def seperate_depth_masks(self):
        #TODO : REMOVE THE AVG & NO EROSION MASKS LATER TO SAVE TIME
        abdomen_masks, chest_masks, invalid_indicies = seperate_masks_to_abdomen_and_chest(self.segs)

        self.depths = [item for i, item in enumerate(self.depths) if i not in invalid_indicies]

        chest_no_erosion_masks = [mask.copy() for mask in chest_masks]
        abdomen_no_erosion_masks = [mask.copy() for mask in abdomen_masks]
        
        abdomen_masks, chest_masks = apply_masks_erosion(abdomen_masks, chest_masks)
        
        abdomen_depths, abdomen_depths_avg, abdomen_depths_no_erosion, abdomen_depths_for_plot = get_masked_depths(abdomen_masks, abdomen_no_erosion_masks, self.depths,  is_chest = False) 
        chest_depths, chest_depths_avg, chest_depths_no_erosion, chest_depths_for_plot = get_masked_depths(chest_masks, chest_no_erosion_masks, self.depths, is_chest = True)
        
        return chest_depths, chest_depths_avg, chest_depths_no_erosion, chest_depths_for_plot, abdomen_depths, abdomen_depths_avg, abdomen_depths_no_erosion, abdomen_depths_for_plot
        
    def save_video(self, plot_frames, seg_frames=[]):
        output_dir = os.path.join(RESULTS_DIR, self.folder_name)
        os.makedirs(output_dir, exist_ok = True)
        output_path = os.path.join(output_dir, f"{self.folder_name}_combined_v2.mp4")

        imgs_dir = os.path.join(IMG_DIR, self.folder_name)
        create_side_by_side_video(imgs_dir,
                          plot_frames,
                          output_path,
                          self.start_frame,
                          self.end_frame,
                          fps = self.fps,
                          seg_frames = seg_frames)
        print(f"Saved video: {self.file_name}")

    def save_json_data(self, video_data):
        json_path = os.path.join(RESULTS_DIR, video_data.folder_name, "data_v2.json")
        # Dump data to json file
        with open(json_path, "w") as f:
            json.dump(video_data, f, default=lambda obj: obj.to_dict())
        print("Json data has been saved")

    def get_peaks(self):
        chest_peaks = [peak['frame_number'] - self.start_frame for peak in self.annot_json_data['chest_peaks'] if peak['frame_number'] <= self.end_frame and peak['frame_number'] >= self.start_frame]
        abdomen_peaks = [peak['frame_number']- self.start_frame for peak  in self.annot_json_data['abdomen_peaks'] if peak['frame_number'] <= self.end_frame and peak['frame_number'] >= self.start_frame]
        return chest_peaks, abdomen_peaks
        
    def process(self, add_peaks=False, save_video=False):
        # TODO: FUNCTION NEEDS TO BE UPDATED
        self.load_depths()
        self.load_video_segmentation()
        
        print("Creating abdomen & chest depth masks")
        chest_depths, chest_avg, chest_no_erosion, chest_for_vid, abdomen_depths, abdomen_avg, abdomen_no_erosion, abdomen_for_vid =self.seperate_depth_masks()
        print("Creating abdomen & chest depth masks done.")
        
        chest_signal = np.array(chest_depths)
        chest_avg_signal = np.array(chest_avg)
        chest_no_erosion_signal = np.array(chest_no_erosion)
        
        abdomen_signal = np.array(abdomen_depths)
        abdomen_avg_signal = np.array(abdomen_avg)
        abdomen_no_erosion_signal = np.array(abdomen_no_erosion)

        smoother = SignalSmoother(self.fps)
        smoothed_chest_signal = smoother.process(chest_signal)
        smoothed_chest_avg_signal = smoother.process(chest_avg_signal)
        smoothed_chest_no_erosion_signal = smoother.process(chest_no_erosion_signal)
        
        smoothed_abdomen_signal = smoother.process(abdomen_signal)
        smoothed_abdomen_avg_signal = smoother.process(abdomen_avg_signal)
        smoothed_abdomen_no_erosion_signal = smoother.process(abdomen_no_erosion_signal)

        # Extract peaks + RR & PA from annotations
        chest_peaks = []
        abdomen_peaks = []
        annot_chest_RR, annot_abdomen_RR, annot_PA = None, None, None
        if add_peaks:
            chest_peaks, abdomen_peaks = self.get_peaks()
            # Create annotation sinusoidal
            # abdomen_x, abdomen_y, abdomen_peak_x, abdomen_peak_y = fit_sinus(self.annot_json_data["abdomen_peaks"], self.end_frame)
            # chest_x, chest_y, chest_peak_x, chest_peak_y = fit_sinus(self.annot_json_data["chest_peaks"], self.end_frame)

            abdomen_x, abdomen_y = generate_wave_data(self.annot_json_data["abdomen_peaks"])
            chest_x, chest_y = generate_wave_data(self.annot_json_data["chest_peaks"])
            annot_chest_RR, annot_abdomen_RR, annot_PA = calc_bpm_and_phase_shift_fft(chest_y, abdomen_y, self.fps)    
        #######################################
        
        # Extracting RR/PA of all types including annotations
        chest_RR, abdomen_RR, PA = calc_bpm_and_phase_shift_fft(smoothed_chest_signal, smoothed_abdomen_signal, self.fps)
        chest_RR_avg, abdomen_RR_avg, PA_avg = calc_bpm_and_phase_shift_fft(smoothed_chest_avg_signal, smoothed_abdomen_avg_signal, self.fps)
        chest_RR_no_erosion, abdomen_RR_no_erosion, PA_no_erosion = calc_bpm_and_phase_shift_fft(smoothed_chest_no_erosion_signal, smoothed_abdomen_no_erosion_signal, self.fps)

        # Extracting Signal Frames for video
        print("Extracting Signal Frames")
        saved_signal_frame = []
        saved_signal_frames = self.save_signals(smoothed_abdomen_signal, smoothed_chest_signal, abdomen_peaks, chest_peaks)
        print("Done.")
        #######################################

        # Extracting Image + Segmentation Frames for video + Saving Video
        if save_video:
            print("Extracting Image + Segmentation Frames")
            seg_frames = create_segmented_frames(self.rgb_frames, chest_for_vid, abdomen_for_vid, self.folder_name, chest_RR, abdomen_RR, PA, annot_chest_RR, annot_abdomen_RR, annot_PA)
            print("Saving Video...")
            self.save_video(saved_signal_frames, seg_frames) # add segmentation frames
        #######################################
            
        # Save json file
        print("Saving json data")
        v_data = VideoData(self.file_name, self.folder_name, chest_signal = smoothed_chest_signal, abdomen_signal = smoothed_abdomen_signal)
        self.save_json_data(v_data)
        print("Done saving json data")

        # Add data to csv file
        csv_path = os.path.join(ROOT_DIR, "annot_results_v2.csv")
        results = [self.folder_name, self.file_name, chest_RR, abdomen_RR, PA, annot_chest_RR, annot_abdomen_RR, annot_PA, chest_RR_avg, abdomen_RR_avg, PA_avg, chest_RR_no_erosion, abdomen_RR_no_erosion, PA_no_erosion]
        self.add_to_csv(csv_path, results)
        #######################################


    def add_to_csv(self, csv_path, new_row):
        with open(csv_path, "a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(new_row)  
    def save_signals(self, abdomen_signal, chest_signal, abdomen_peaks = [], chest_peaks = []):
        frames = []
        for i, _ in tqdm(enumerate(abdomen_signal)):
            fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))  # 1 row, 2 columns
        
            axes[0].set_xlim(0, len(abdomen_signal))  # x-axis limit
            axes[0].set_ylim(min(abdomen_signal), max(abdomen_signal))  # x-axis limit
            axes[0].set_title("Abdomen Signal")

            for peak in chest_peaks:
                axes[0].axvline(x=peak, color='blue', linestyle='--', alpha=0.7, label='Chest Peak' if peak == chest_peaks[0] else "")

            axes[1].set_xlim(0, len(chest_signal))  # x-axis limit
            axes[1].set_ylim(min(chest_signal), max(chest_signal))  # x-axis limit
            axes[1].set_title("Chest Signal")
            for peak in abdomen_peaks:
                axes[1].axvline(x=peak, color='orange', linestyle='--', alpha=0.7, label='Abdomen Peak' if peak == abdomen_peaks[0] else "")

            line, = axes[0].plot(abdomen_signal[:i], color="blue")
            line, = axes[1].plot(chest_signal[:i], color="orange")
            buf = io.BytesIO()
            fig.savefig(buf, format="jpg")
            buf.seek(0)
            frames.append(Image.open(buf))
            # plt.savefig(os.path.join("Outputs", f"{i:03}.jpg"), dpi=300, bbox_inches="tight")  # Adjust dpi for resolution, bbox_inches for padding
            plt.close(fig)
        return frames
