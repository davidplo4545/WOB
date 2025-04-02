class VideoData:
    def __init__(self, file_name, folder_name, chest_signal, abdomen_signal):
        self.file_name = file_name
        self.folder_name = folder_name
        self.chest_signal = chest_signal.tolist()
        self.abdomen_signal = abdomen_signal.tolist()
        self.phase_angle = 0
        self.breathing_rate = 0
        self.frames_range = []

    def to_dict(self):
        return { 
            "file_name": self.file_name,
            "folder_name": self.folder_name,
            "chest_signal": self.chest_signal,
            "abdomen_signal": self.abdomen_signal,
            "phase_angle": self.phase_angle,
            "breathing_rate": self.breathing_rate,
            "frames_range": self.frames_range
       }
        