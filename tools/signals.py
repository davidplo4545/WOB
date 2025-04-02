import numpy as np
import json
from scipy.signal import butter, filtfilt
from scipy.ndimage import median_filter
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

def RR_and_PA_from_json(json_path):
    with open(json_path) as f:
        gt = json.load(f)

    # Sort frames by ascending order
    gt["chest_peaks"] = sorted(gt["chest_peaks"], key=lambda x: x["frame_number"])
    gt["abdomen_peaks"] = sorted(gt["abdomen_peaks"], key=lambda x: x["frame_number"])

    max_start = max(gt["chest_peaks"][0]["frame_number"], gt["abdomen_peaks"][0]["frame_number"])
    min_end = min(gt["chest_peaks"][-1]["frame_number"], gt["abdomen_peaks"][-1]["frame_number"])

    end_frame = max(gt["chest_peaks"][-1]["frame_number"], gt["abdomen_peaks"][-1]["frame_number"])
    fps = gt["chest_peaks"][-1]["frame_number"] / float(gt["chest_peaks"][-1]["timestamp"].replace(':','.'))
    # print(f"End frame is:{end_frame}, FPS: {fps}")
    
    abdomen_x, abdomen_y = generate_wave_data(gt["abdomen_peaks"], max_start, min_end)
    chest_x, chest_y = generate_wave_data(gt["chest_peaks"], max_start, min_end)
    # print(max_start, min_end)
    # print(len(chest_y), len(abdomen_y))
    RR_chest, RR_abdomen, PA = calc_bpm_and_phase_shift_fft(chest_y, abdomen_y, fps)    

    return RR_chest, RR_abdomen, PA


def calc_bpm_and_phase_shift_fft(signal1, signal2, rgb_fps):
    signal1 = np.array(signal1)
    signal2 = np.array(signal2)
    signal1_BPMs = []
    signal2_BPMs = []
    PAs = []
    window_length_seconds = 15
    crop = lambda x: x[start:start+int(window_length_seconds*rgb_fps)]
    for start in range(0,len(signal1)-int(window_length_seconds*rgb_fps),2):
        # try:
        output = get_bpm_and_phase_shift_fft(crop(signal1), crop(signal2), rgb_fps)
        signal1_BPMs.append(output[0])
        signal2_BPMs.append(output[1])
        PAs.append(output[2])
        # except:
            # print(f"Last window = not enough values to process {start}")
            # pass
    signal1_BPMs = np.array(signal1_BPMs)
    signal2_BPMs = np.array(signal2_BPMs)
    PAs = np.array(PAs)
    
    signal1_RR = mean_without_outliers(signal1_BPMs)
    signal2_RR = mean_without_outliers(signal2_BPMs)
    PA = mean_without_outliers(PAs)
    return signal1_RR, signal2_RR, PA
    
def get_bpm_and_phase_shift_fft(signal1, signal2, sampling_rate):
    # Compute the FFT of both signals
    min_len = min(len(signal1), len(signal2))
    signal1 = signal1[:min_len]
    signal2 = signal2[:min_len]
    
    fft1 = np.fft.fft(signal1-signal1.mean())
    fft2 = np.fft.fft(signal2-signal2.mean())
    # Compute the frequency bins
    freq_bins = np.fft.fftfreq(len(signal1), 1/sampling_rate)
    
    # Find the index of the peak frequency component
    # Focus on positive frequencies and ignore very low frequencies to avoid DC component
    # positive_frequencies = freq_bins > 0.2
    positive_frequencies = freq_bins > 0.2

    # Compute the magnitude of the FFT for both signals
    magnitude_fft1 = np.abs(fft1)
    magnitude_fft2 = np.abs(fft2)
    # print(magnitude_fft1.reshape(-1,1).shape)
    # Find the index of the peak frequency component for each signal
    peak_index_1 = np.argmax(magnitude_fft1[positive_frequencies])
    peak_index_2 = np.argmax(magnitude_fft2[positive_frequencies])

    if np.max(magnitude_fft1[positive_frequencies]) > np.max(magnitude_fft2[positive_frequencies]):
        chosen_index = peak_index_1
    else:
        chosen_index = peak_index_2
        
    # Peak frequencies for each signal
    peak_frequency_1 = freq_bins[positive_frequencies][peak_index_1]
    peak_frequency_2 = freq_bins[positive_frequencies][peak_index_2]

    # Calculate phase angles at the peak frequency
    phase_angle_1 = np.angle(fft1[positive_frequencies][chosen_index])
    phase_angle_2 = np.angle(fft2[positive_frequencies][chosen_index])

    # Compute the phase shift
    # print(np.rad2deg(phase_angle_2),np.rad2deg(phase_angle_1))
    phase_shift_calculated = phase_angle_2 - phase_angle_1
    # print(np.abs(np.rad2deg(phase_shift_calculated)))
    # Convert phase shift to degrees and adjust it to be within 0 to 180 degrees
    phase_shift_calculated_deg = np.abs((np.rad2deg(phase_shift_calculated) + 180) % 360 - 180)
    # phase_shift_calculated_deg = min(360 - phase_shift_calculated, phase_shift_calculated)

    return 60*peak_frequency_1, 60*peak_frequency_2, phase_shift_calculated_deg

def mean_without_outliers(data):
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    filtered_data = [x for x in data if lower_bound <= x <= upper_bound]

    if not filtered_data:
        raise ValueError("All values are outliers")

    return np.mean(filtered_data)

def mean_without_outliers_using_std(data, num_std_dev=2):
    mean = np.mean(data)
    std_dev = np.std(data)

    lower_bound = mean - num_std_dev * std_dev
    upper_bound = mean + num_std_dev * std_dev

    filtered_data = [x for x in data if lower_bound <= x <= upper_bound]

    if not filtered_data:
        raise ValueError("No data left after removing outliers")

    return np.mean(filtered_data)

class SignalSmoother():
    def __init__(self, sampling_rate):
        self.sampling_rate = sampling_rate

    def cubic_spline_interpolation(self, signal, target_frequency=40):
        return signal

    def butter_bandpass(self, lowcut, highcut, order):
        nyq = 0.5 * self.sampling_rate
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return b, a
    
    def butter_bandpass_filter(self, data, lowcut, highcut, order):
        b, a = self.butter_bandpass(lowcut, highcut, order)
        y = filtfilt(b, a, data)
        return y
    
    def moving_median_filter(self, signal, window_size):
        median_filtered_signal = median_filter(signal, size=int(self.sampling_rate*window_size))
        return median_filtered_signal
    
    def nonlinear_compression(self, signal):
        compressed_signal = np.arctan(signal/(np.sqrt(2*np.sum((signal-signal.mean())**2)/(signal.shape[0]-1))))
        return compressed_signal
    
    def process(self, signal, bandpass_min=0.05, bandpass_max=1, bandpass_order=2, median_window=3):
    
        raw_respiratory_signal = signal
        
        # Step 1: Cubic Spline Interpolation
        interpolated_signal = self.cubic_spline_interpolation(raw_respiratory_signal, self.sampling_rate)
        
        # Step 2: Bandpass Filter
        filtered_signal = self.butter_bandpass_filter(interpolated_signal, bandpass_min, bandpass_max, bandpass_order)
        
        # Step 3: Moving Median Filter
        median_filtered_signal = self.moving_median_filter(filtered_signal,median_window)
        
        # Step 4: Nonlinear Compression
        compressed_signal = self.nonlinear_compression(filtered_signal)
    
        return compressed_signal


def generate_wave_data(peaks, max_start, min_end):
    x_values = []
    y_values = []
    
    for i in range(len(peaks) - 1):
        start_frame = peaks[i]["frame_number"]
        end_frame = peaks[i + 1]["frame_number"]
        # Generate a smooth sine wave between two peaks, covering full cycle (-1 to 1)
        x_segment = np.linspace(start_frame, end_frame, end_frame - start_frame + 1)
        y_segment = np.cos(np.linspace(0, 2 * np.pi, end_frame - start_frame + 1))  # Cosine wave for full rounded effect
        x_values.extend(x_segment)
        y_values.extend(y_segment)

    result_x = []
    result_y = []
    for i in range(len(x_values)):
        if min_end >=x_values[i] >= max_start:
            result_x.append(x_values[i])
            result_y.append(y_values[i])
    return result_x, result_y
    
def sinusoidal(x, A, B, C, D):
    return A * np.sin(B * x + C) + D

def fit_sinus(peaks, end_frame):
    peak_values = []
    lows_values = []
    for i in range(len(peaks)-1):
        prev_peak_frame = peaks[i]["frame_number"]
        peak_values.append((prev_peak_frame, 1))
        next_peak_frame = peaks[i+1]["frame_number"]
        mid_frame = (prev_peak_frame + next_peak_frame) // 2
        lows_values.append((mid_frame, -1))
    all_values = peak_values + lows_values
    all_values = sorted(all_values, key=lambda x: x[0])

    x_values = [item[0] for item in all_values]
    y_values = [item[1] for item in all_values]
    x_values = np.array(x_values)
    peaks, _ = find_peaks(y_values, height=0.5)  # Find peaks at y ≈ 1
    peak_x = x_values[peaks]  # x-values of peaks
    
    # Estimate period and frequency
    if len(peak_x) > 1:
        estimated_period = np.mean(np.diff(peak_x))  # Average distance between peaks
        estimated_B = 2 * np.pi / estimated_period  # Convert period to angular frequency
    else:
        estimated_B = 2 * np.pi / (max(x_values) - min(x_values))  # Fallback estimate

    A_init = (max(y_values) - min(y_values)) / 2  # Amplitude
    B_init = estimated_B  
    C_init = 0  # Phase shift
    D_init = np.mean(y_values)  # Vertical shift
    initial_guess = [A_init, B_init, C_init, D_init]
    
    # Fit the function to the data
    params, _ = curve_fit(sinusoidal, x_values, y_values, p0=initial_guess)
    
    # Extract the fitted parameters
    A_fit, B_fit, C_fit, D_fit = params
    
    # Generate fitted curve
    x_fit = np.linspace(min(x_values), max(x_values), end_frame)  # Smooth curve
    A_fit = 1
    y_fit = sinusoidal(x_fit, A_fit, B_fit, C_fit, D_fit)

    return x_fit, y_fit, x_values, y_values




