import pandas as pd
import numpy as np
from utils.util import *
from utils.parser import *
from sklearn.ensemble import ExtraTreesClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

def ana_pipe(pipe):
    folder, subfolder = create_folders("./output/", folder="smartg_plate", subfolder=["peak_feature", "window_feature"])
    d_spectra = parse_data("./MLP-VAE-WGAN/data/Raw_samples")

    if pipe == "window_feature":
        df = pd.DataFrame()
        intervals = [(400, 2000)] 
        for_mean = {}
        for file, raman_spectrum in d_spectra.items():
            preprocessed_spectrum = preprocessor.apply_pipeline(raman_spectrum, file, 10000, folder, subfolder[1], (300, 2000))
            for_mean[file] = preprocessed_spectrum
            indexes_within_intervals = np.concatenate([np.where((interval[0] < preprocessed_spectrum[0]) & (preprocessed_spectrum[0] < interval[1]))[0] for interval in intervals])
            filtered_wavenumbers = preprocessed_spectrum[0][indexes_within_intervals]
            filtered_intensities = preprocessed_spectrum[1][indexes_within_intervals]
            new_row_raw = pd.DataFrame([filtered_intensities], columns=filtered_wavenumbers.astype(str))
            new_row_raw['Label'] = file.split('$')[0]
            new_row_raw['mg/L'] = float(file.split('$')[1])
            new_row_raw['Name'] = file
            df = pd.concat([df, new_row_raw], ignore_index=True)
        df.to_excel("./MLP-VAE-WGAN/data/preprocessed_data.xlsx")

    else:
        pass

    return df

window_df = ana_pipe("window_feature")
