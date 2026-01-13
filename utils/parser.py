import os
import numpy as np
import pandas as pd

def parse_data(main_folder_path):
    d_spectra = {}
    for s_folder in os.listdir(main_folder_path):
        final_path = os.path.join(main_folder_path, s_folder)
        if os.path.isdir(final_path): #and s_folder.startswith("S"):
            sample_count = 0
            for filename in os.listdir(final_path):
                file_path = os.path.join(final_path, filename)
                numeric_columns = ['Raman shift [1/cm]', 'Intensity']
                data = pd.read_csv(file_path, sep="\t", skiprows=14, names=numeric_columns)
                data[numeric_columns] = data[numeric_columns].astype(float)
                spectral_axis = data["Raman shift [1/cm]"].values
                spectral_data = data["Intensity"].values
                data_list = [spectral_axis, spectral_data]
                raman_spectrum = np.array(data_list)
                if "CTR" in filename: #"no sulfanamide" in filename:
                    d_spectra["control$" + str(sample_count)] = raman_spectrum
                else:
                    d_spectra[filename.split("_")[5] + "$" + str(sample_count)] = raman_spectrum
                sample_count += 1
    return d_spectra

