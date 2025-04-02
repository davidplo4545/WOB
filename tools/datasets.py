import json
import torch, os
import numpy as np
from torch.utils.data import Dataset, DataLoader
import random
from scipy import signal

# Define the Dataset class
class BreathingPatternDataset(Dataset):
    def __init__(self, json_file, blacklist=[], is_test = False, num_segments = 10, sample_length_sec=15, fps=15, transform=None):
        """
        Args:
            json_file (str): Path to the JSON file with data.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        with open(json_file, 'r') as f:
            self.data = json.load(f)
        self.sample_length = sample_length_sec * fps  # Convert sample length to number of samples
        self.fps = fps
        self.transform = transform
        self.num_segments = num_segments
        self.segments = []
        for key, sample in self.data.items():

            abdomen_signal = np.array(sample['abdomen_signal'])
            chest_signal = np.array(sample['chest_signal'])
            doc_class = sample['doc_class']
            if doc_class == None:
                continue
            doc_class = int(doc_class) - 1
            signal_length = len(abdomen_signal)
            
            # Skip samples that are too short
            if signal_length < self.sample_length:
                continue

            if (key in blacklist and is_test == False) or (key not in blacklist and is_test==True):
                continue
            # Sequential Segments
            chest_segments = [np.array(chest_signal[i:i+self.sample_length]) for i in range(0, len(chest_signal) - self.sample_length + 1, self.sample_length)]
            abdomen_segments = [np.array(abdomen_signal[i:i+self.sample_length]) for i in range(0, len(abdomen_signal) - self.sample_length + 1, self.sample_length)]
            for i in range(len(chest_segments)):
                self.segments.append((abdomen_segments[i], chest_segments[i], doc_class))
    
            # Random Segements
            # max_start_idx = signal_length - self.sample_length
            # start_indices = random.sample(range(0, max_start_idx), min(self.num_segments, max_start_idx))

            
            # for start in start_indices:
            #     self.segments.append((abdomen_signal[start:start + self.sample_length],
            #                           chest_signal[start:start + self.sample_length], 
            #                           doc_class))
    
    def __len__(self):
        return len(self.segments)
    
    def __getitem__(self, idx):
        abdomen_segment, chest_segment, doc_class = self.segments[idx]
        
        if self.transform:
            abdomen_segment, chest_segment = self.transform(abdomen_segment, chest_segment)
        
        # Convert to PyTorch tensors
        abdomen_segment = torch.tensor(abdomen_segment, dtype=torch.float32)  # Shape: (sample_length,)
        chest_segment = torch.tensor(chest_segment, dtype=torch.float32)      # Shape: (sample_length,)
        doc_class = torch.tensor(doc_class, dtype=torch.long)  # Scalar doc_class
        
        return abdomen_segment, chest_segment, doc_class
class Augmentation:
    def __init__(self, jitter_strength=0.01, time_warp_strength=0.05):
        self.jitter_strength = jitter_strength
        self.time_warp_strength = time_warp_strength
    
    def jitter(self, signal):
        """Add random noise to the signal"""
        noise = np.random.normal(0, self.jitter_strength, size=signal.shape)
        return signal + noise
    
    def time_warp(self, signal):
        """Perform time-warping augmentation"""
        return signal + np.sin(np.linspace(0, np.pi, len(signal))) * self.time_warp_strength
    
    def __call__(self, abdomen_signal, chest_signal):
        """Apply augmentations to both abdomen and chest signals"""
        abdomen_signal = self.jitter(abdomen_signal)
        chest_signal = self.jitter(chest_signal)
        
        abdomen_signal = self.time_warp(abdomen_signal)
        chest_signal = self.time_warp(chest_signal)
        
        return abdomen_signal, chest_signal


