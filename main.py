import numpy as np 
import sys
import matplotlib
import matplotlib.pyplot as plt
# import plotly.express as px
import seaborn as sns
import os
import cv2
from collections import defaultdict
import pandas as pd 
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
# matplotlib.use('TkAgg') 
from matplotlib.animation import FuncAnimation
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
import itertools
import struct

class FrameConfiguration:
    def __init__(self):
        self.numtx=3
        self.numrx=4
        self.chirps=182
        self.range_bins=256
        self.iq=2
        self.frameSize=3*4*182*256
        self.enable_static_removal=False


class RawDataReader:
    def __init__(self, path):
        self.path = path
        self.ADCBinFile = open(path, 'rb')

    def getNextFrame(self, frameconfig):
        timestamp = self.ADCBinFile.read(8)
        timestamp = struct.unpack('d', timestamp)[0]

        frame = np.frombuffer(self.ADCBinFile.read(frameconfig.frameSize * 4), dtype=np.int16)

        r_omega = self.ADCBinFile.read(8)
        r_omega = struct.unpack('d', r_omega)[0]

        l_omega = self.ADCBinFile.read(8)
        l_omega = struct.unpack('d', l_omega)[0]

        angle = self.ADCBinFile.read(8)
        angle = struct.unpack('d', angle)[0]

        return timestamp, frame, r_omega, l_omega, angle

    def close(self):
        self.ADCBinFile.close()

def i_qvalues(frame):#returns the frma in iq value
    np_frame = np.zeros(shape=(len(frame) // 2), dtype=np.complex_)
    np_frame[0::2] = frame[0::4] + 1j * frame[2::4]
    np_frame[1::2] = frame[1::4] + 1j * frame[3::4]
    return np_frame

def reshape_frame(frame):
    #the size of the frame has to be made to 3*4*182*256
     frame=np.reshape(frame,(182,3,4,-1))
     return frame.transpose(1,2,0,3)
     

def range_FFT(reshape_frame,window=-1):
    rangeFFT=None
    if window==-1:#by default rectangular window
        rangeFFT=np.fft.fft(reshape_frame)
        return rangeFFT
    if window=='hamming':
        window = np.hamming(256)
        windowed_frame=reshape_frame*window
        return np.fft.fft(windowed_frame,axis=-1)
    
def dopplerFFT(range_result,frameconfig):
    #rangeresult ka shape hai 3*4*128*256
    #frameConfig.numLoopsPerFrame hai 128
    windowedBins2D = range_result * np.reshape(np.ones(frameconfig.chirps), (1, 1, -1, 1))
    #taking a hamming window for FFT.
    dopplerFFTResult = np.fft.fft(windowedBins2D, axis=2) #array of size 3*4*128*256
    dopplerFFTResult = np.fft.fftshift(dopplerFFTResult, axes=2)#shift the zero frequency to the center of the array
    return dopplerFFTResult

def clutter_removal(input_val, axis=0):  #axis =2 passed from main function
    # Reorder the axes
    #input val is rangeFFT of dimension 3*4*128*256
    reordering = np.arange(len(input_val.shape))
    #input_val.shape ka length 4 hai as there are 4 dimensions
    #so reordering is the array [0,1,2,3]
    reordering[0] = axis
    #reordering=[2,1,2,3]
    reordering[axis] = 0
    #reordering=[2,1,0,3]
    input_val = input_val.transpose(reordering)
    #abhi input value ka shape hai 182*4*3*256

    # Apply static clutter removal
    mean = input_val.mean(0) #caluclate mean across the first axis(across the 128 wala axis)
    #mean ka shape is 4*3*256
    output_val = input_val - mean
    """
    This operation essentially removes the static background or "clutter" from 
    the radar data, leaving behind only the dynamic components (like moving objects). 
    This is a common preprocessing step in radar processing to enhance the detection 
    capability of moving targets.
    """
    #basically this will give a black line in the doppler range heatmap at velocity =zero. This code removes the zero velocity onject
    return output_val.transpose(reordering)

def get_coordinates(dopplerResult):
    #First 30cm make it very negative so the first 3 bins
    cfar_result=np.zeros(dopplerResult.shape,bool)
    top_128=128
    energy_threshold = np.partition(dopplerResult.ravel(), 182 * 256 - top_128 - 1)[182 * 256 - top_128 - 1]
        #So energy Thre128 is the 128th most energetic point
    # print(energy_threshold)
    cfar_result[dopplerResult>energy_threshold]=True
    det_peaks_indices = np.argwhere(cfar_result == True)
    # print(det_peaks_indices.shape)
    object_energy_coordinates=np.zeros((top_128,3))
    object_energy_coordinates[:,0]=det_peaks_indices[:,0]
    object_energy_coordinates[:,1]=det_peaks_indices[:,1]
    for i in range(top_128):
        x_cor=object_energy_coordinates[i][0]
        y_cor=object_energy_coordinates[i][1]
        object_energy_coordinates[i][2]=dopplerResult[int(x_cor)][int(y_cor)]
    
    return object_energy_coordinates,cfar_result
        
def get_azimuthal_angle(dopplerResult,cfar_result):
    az_angle_map={}
    for i in range(cfar_result.shape[0]):
        for j in range(cfar_result.shape[1]):
            if cfar_result[i][j]==True:
                key=(i,j)
                az_angle_map[key]=dopplerResult[:,:,i,j].reshape(12,-1).flatten()[0:8]
    for key,value in az_angle_map.items():
        azimuth_fft_padded=np.zeros(64,dtype=np.complex_)
        azimuth_fft_padded[0:8]=az_angle_map[key]
        azimuth_fft_padded=np.fft.fft(azimuth_fft_padded)
        azimuth_fft_padded = np.fft.fftshift(azimuth_fft_padded)
        az_angle_map[key]=np.abs(azimuth_fft_padded)
    return az_angle_map


def get_scores_and_labels(combinations, X):
    scores = []
    all_labels_list = []

    for i, (eps, num_samples) in enumerate(combinations):
        dbscan_cluster_model = DBSCAN(eps=eps, min_samples=num_samples).fit(X)
        labels = dbscan_cluster_model.labels_
        labels_set = set(labels)
        num_clusters = len(labels_set)
        if -1 in labels_set:
            num_clusters -= 1
    
        if (num_clusters < 2) or (num_clusters > 50):
            scores.append(-10)
            all_labels_list.append('bad')
            c = (eps, num_samples)
            print(f"Combination {c} on iteration {i+1} of {N} has {num_clusters} clusters. Moving on")
            continue
    
        scores.append(ss(X, labels))
        all_labels_list.append(labels)
        print(f"Index: {i}, Score: {scores[-1]}, Labels: {all_labels_list[-1]}, NumClusters: {num_clusters}")

    best_index = np.argmax(scores)
    best_parameters = combinations[best_index]
    best_labels = all_labels_list[best_index]
    best_score = scores[best_index]

    return {'best_epsilon': best_parameters[0],
            'best_min_samples': best_parameters[1], 
            'best_labels': best_labels,
            'best_score': best_score}


def main():
    total_frame_number=0
    file_path=sys.argv[1]
    total_frame_number=int(sys.argv[2])
    count=1
    frameconfig=FrameConfiguration()
    bin_reader = RawDataReader(file_path)
    range_vals = []
    dop_vals = []
    angle_vals = []
    gt_vals = []
    for frame_no in range(total_frame_number):
        count+=1
        timestamp, np_frame, l_omega, r_omega, angle = bin_reader.getNextFrame(frameconfig)
        print(timestamp, l_omega, r_omega, angle)

        np_frame=i_qvalues(np_frame)
        reshaped_np_frame=reshape_frame(np_frame)
        range_result=range_FFT(reshaped_np_frame)
        range_vals.append(np.abs(range_result).sum(axis=0).sum(axis=0).sum(axis=0)[0])
        # range_result=clutter_removal(range_result,axis=2)
        dopplerResult=dopplerFFT(range_result,frameconfig)
        dopplerResultabs=np.absolute(dopplerResult)
        dopplerResultabs=np.sum(dopplerResultabs,axis=(0,1))
        dop_vals.append(dopplerResultabs)

        energy_coordinates,cfar_result=get_coordinates(dopplerResultabs)
        energy_coordinates=energy_coordinates[energy_coordinates[:,2].argsort()[::-1]]
        az_angle_map=get_azimuthal_angle(dopplerResult,cfar_result)

        range_angle=np.zeros((256,64),dtype=np.complex_)
        for key,value in az_angle_map.items():      #key = (vel, range)
            range_angle[key[1]]+=np.abs(value)           #unique range

        range_angle_abs = np.abs(range_angle)
        angle_vals.append(range_angle_abs)
        avg_speed = ((l_omega+r_omega)/2)
        avg_speed_x = avg_speed*np.cos(angle)
        avg_speed_y = avg_speed*np.sin(angle)
        gt_vals.append((avg_speed_x, avg_speed_y))


if __name__ == "__main__":
    main()