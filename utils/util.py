import os
import warnings
import copy
import time
import shutil
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from imblearn.combine import SMOTEENN
from lazypredict.Supervised import LazyClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA, QuadraticDiscriminantAnalysis as QDA
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from scipy.stats import chi2, zscore
from scipy.signal import find_peaks
from scipy.sparse import spdiags
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
from numpy.linalg import pinv
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d import Axes3D
from pybaselines.whittaker import airpls, arpls, asls, derpsalsa, drpls, iarpls, iasls, psalsa
import matplotlib.cm as cm
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import r2_score
from collections import defaultdict

warnings.filterwarnings('ignore')

class RamanPreprocessor:
    def __init__(self):
        pass

    # Cropper
    @staticmethod
    def cropper(raman_spectrum, *intervals):
        cropped_data = []
        for interval in intervals:
            min_value, max_value = interval
            indices_to_keep = np.where((raman_spectrum[0] >= min_value) & (raman_spectrum[0] <= max_value))
            cropped_x_array = raman_spectrum[0][indices_to_keep]
            cropped_y_array = raman_spectrum[1][indices_to_keep]
            cropped_data.append([cropped_x_array, cropped_y_array])
        return np.concatenate(cropped_data, axis=1)

    # Baseline Correction
    @staticmethod
    def ensieh_ARPLS(raman_spectrum, smoothing):
        x, raman_data = raman_spectrum
        N = len(raman_data)
        D = np.diff(np.identity(N), 2)
        H = smoothing*np.dot(D, np.transpose(D))
        w = np.ones(N)
        while True:
            W = spdiags(w, 0, N, N).toarray()
            z = np.dot(np.linalg.inv(W+H), np.dot(W, raman_data))
            d = raman_data-z
            index_d_negative = np.where(d < 0)
            dn = d[index_d_negative]
            m = np.mean(dn)
            s = np.std(dn)
            wt = np.divide(1, 1+np.exp(2*(d-(2*s-m))/s))
            if (np.linalg.norm(w-wt) / np.linalg.norm(w) < 0.05):
                break
            w = wt
        return np.array([x, z]), np.array([x, d])

    @staticmethod
    def AIRPLS(raman_spectrum, smoothing):
        x_array, y_array = raman_spectrum
        baseline_tuple = airpls(y_array, lam=smoothing, diff_order=2, max_iter=500, tol=0.001, weights=None, x_data=x_array)
        baseline_subtracted_y = y_array - baseline_tuple[0]
        return np.array([x_array, baseline_tuple[0]]), np.array([x_array, baseline_subtracted_y])

    @staticmethod
    def IARPLS(raman_spectrum, smoothing):
        x_array, y_array = raman_spectrum
        baseline_tuple = iarpls(y_array, lam=smoothing, diff_order=2, max_iter=500, tol=0.001, weights=None, x_data=x_array)
        baseline_subtracted_y = y_array - baseline_tuple[0]
        return np.array([x_array, baseline_tuple[0]]), np.array([x_array, baseline_subtracted_y])

    @staticmethod
    def ARPLS(raman_spectrum, smoothing):
        x_array, y_array = raman_spectrum
        baseline_tuple = arpls(y_array, lam=smoothing, diff_order=2, max_iter=500, tol=0.001, weights=None, x_data=x_array)
        baseline_subtracted_y = y_array - baseline_tuple[0]
        return np.array([x_array, baseline_tuple[0]]), np.array([x_array, baseline_subtracted_y])

    @staticmethod
    def ASLS(raman_spectrum, smoothing):
        x_array, y_array = raman_spectrum
        baseline_tuple = asls(y_array, lam=smoothing, diff_order=2, max_iter=500, tol=0.001, weights=None, x_data=x_array)
        baseline_subtracted_y = y_array - baseline_tuple[0]
        return np.array([x_array, baseline_tuple[0]]), np.array([x_array, baseline_subtracted_y])

    @staticmethod
    def DERPSALSA(raman_spectrum, smoothing):
        x_array, y_array = raman_spectrum
        baseline_tuple = derpsalsa(y_array, lam=smoothing, diff_order=2, max_iter=500, tol=0.001, weights=None, x_data=x_array, eta=0.8)
        baseline_subtracted_y = y_array - baseline_tuple[0]
        return np.array([x_array, baseline_tuple[0]]), np.array([x_array, baseline_subtracted_y])

    @staticmethod
    def DRPLS(raman_spectrum, smoothing):
        x_array, y_array = raman_spectrum
        baseline_tuple = drpls(y_array, lam=smoothing, diff_order=2, max_iter=500, tol=0.001, weights=None, x_data=x_array, eta=1.0) #diff_order=2 eta=1.0
        baseline_subtracted_y = y_array - baseline_tuple[0]
        return np.array([x_array, baseline_tuple[0]]), np.array([x_array, baseline_subtracted_y])

    @staticmethod
    def IASLS(raman_spectrum, smoothing):
        x_array, y_array = raman_spectrum
        baseline_tuple = iasls(y_array, lam=smoothing, diff_order=2, max_iter=500, tol=0.001, weights=None, x_data=x_array)
        baseline_subtracted_y = y_array - baseline_tuple[0]
        return np.array([x_array, baseline_tuple[0]]), np.array([x_array, baseline_subtracted_y])

    @staticmethod
    def PSALSA(raman_spectrum, smoothing):
        x_array, y_array = raman_spectrum
        baseline_tuple = psalsa(y_array, lam=smoothing, diff_order=2, max_iter=500, tol=0.001, weights=None, x_data=x_array)
        baseline_subtracted_y = y_array - baseline_tuple[0]
        return np.array([x_array, baseline_tuple[0]]), np.array([x_array, baseline_subtracted_y])
    

    @staticmethod
    def whitaker_hayes(raman_spectrum, threshold=6):  #threshold=8
        y_array = raman_spectrum[1]
        median_y = np.median(y_array)
        std_y = np.std(y_array)
        cosmic_rays = np.abs(y_array - median_y) > threshold * std_y
        x_array = raman_spectrum[0]
        x_cosmic = x_array[cosmic_rays]
        f_interp = interp1d(x_array[~cosmic_rays], y_array[~cosmic_rays], kind='quadratic', fill_value='extrapolate') # kind='quadratic'
        y_array[cosmic_rays] = f_interp(x_cosmic)
        return np.array([x_array, y_array])
    
    @staticmethod
    def remove_spikes_extended(raman_spectrum, threshold=6):
        x, y = raman_spectrum
        y = y.copy()
        dy = np.diff(y)
        mean_dy = np.mean(np.abs(dy))
        dy_std = np.std(dy)
        i = 1
        while i < len(y) - 2:
            left_diff = y[i] - y[i - 1]
            right_diff = y[i + 1] - y[i]
            if left_diff - mean_dy > threshold * dy_std and right_diff + mean_dy < -(threshold * dy_std):
                start = i - 1
                while start > 0 and (y[start] - y[start - 1]) > 0:
                    start -= 1
                end = i + 1
                while end < len(y) - 1 and (y[end + 1] - y[end]) < 0:
                    end += 1
                y[start:end + 1] = np.interp(
                    x[start:end + 1],
                    [x[start], x[end]],
                    [y[start], y[end]]
                )
                i = end
            else:
                i += 1
        return np.array([x, y])

    # Denoise
    @staticmethod
    def denoise_spectrum(raman_spectrum, window_size=15, poly_order=2):# window_size=15, poly_order=2 
        x_array = raman_spectrum[0]
        y_array = raman_spectrum[1]
        window_size = window_size + 1 if window_size % 2 == 0 else window_size
        denoised_y = savgol_filter(y_array, window_size, poly_order)
        return np.array([x_array, denoised_y])

    # Remove Negative Peaks
    @staticmethod
    def get_negatives(raman_spectrum, h=-100000, d=1.0, p=0.0025, w=-100000):
        x, y = raman_spectrum
        inverted_y = -y
        peaks, properties = find_peaks(inverted_y, height=h, distance=d, prominence=p, width=w)
        intervals = [(408, 418), (633, 643), (723, 739), (840, 850), (862, 872), (998, 1008), (1369, 1388), (1768, 1778), (1850, 1860), (1890, 1900)]
        filtered_peaks = []
        filtered_properties = {key: [] for key in properties}
        for i, peak in enumerate(peaks):
            peak_x = x[peak]
            for interval in intervals:
                if interval[0] <= peak_x <= interval[1]:
                    filtered_peaks.append(peak)
                    for key, value in properties.items():
                        filtered_properties[key].append(value[i])
                    break  # If peak is in any interval, no need to check further intervals

        # Interpolate peaks
        for i, peak_index in enumerate(filtered_peaks):
            left_index = peak_index
            right_index = peak_index
            while left_index > 0 and inverted_y[left_index] > inverted_y[left_index - 1]:
                left_index -= 1
            while right_index < len(inverted_y) - 1 and inverted_y[right_index] > inverted_y[right_index + 1]:
                right_index += 1

            left = x[left_index]
            right = x[right_index]
            if abs(right - left) > 70:
                pass
            else:
                x_between = np.linspace(left, right, right_index - left_index + 1)
                y_between = np.interp(x_between, [x[left_index], x[right_index]], [y[left_index], y[right_index]])
                y[left_index:right_index + 1] = y_between
        return np.array([x, y])

    # Normalization
    @staticmethod
    def minmax_normalize(raman_spectrum, interval=(0, 1)):
        x_array = raman_spectrum[0]
        y_array = raman_spectrum[1]
        min_value, max_value = interval
        normalized_array = (y_array - np.min(y_array)) / (np.max(y_array) - np.min(y_array))
        normalized_array = normalized_array * (max_value - min_value) + min_value
        return np.array([x_array, normalized_array])
    
    @staticmethod
    def log_minmax_normalize(raman_spectrum):
        x_array = raman_spectrum[0]
        y_array = raman_spectrum[1]
        if np.any(y_array < 0):
            y_array = np.where(y_array < 0, 0, y_array)  # Replace negative values with 0
        y_log_transformed = np.log1p(y_array)
        y_min = np.min(y_log_transformed)
        y_max = np.max(y_log_transformed)
        y_normalized = (y_log_transformed - y_min) / (y_max - y_min)
        return np.array([x_array, y_normalized])
    
    @staticmethod
    def cof_normalize(raman_spectrum): # 1405, 1735
        max_y = None
        for x, y in zip(raman_spectrum[0], raman_spectrum[1]):
            if 1590 <= x <= 1610:
            #if 1170 <= x <= 1184:
                if max_y is None or y > max_y:
                    max_y = y
        raman_data = raman_spectrum[1] / max_y
        return np.array([raman_spectrum[0], raman_data])

    @staticmethod
    def plot_spectrum(spectral_d, title, folder1, folder2):
        plt.figure(figsize=(18, 8))
        for legend, spectra in spectral_d.items():
            x, y = spectra
            plt.plot(x, y, label=f'{legend}')
            if " C" in legend:
                peaks, _ = RamanPreprocessor.get_features(spectra)
                plt.plot(x[peaks], y[peaks], 'rx')
            else:
                continue

        '''dashed_lines_x = [487, 572, 610, 718, 1015, 1029, 1090, 1134, 1171, 1224, 1294, 1335, 1550, 1590, 1858, 1930]
        for x_position in dashed_lines_x:
            plt.axvline(x_position, color='gray', linestyle='--', linewidth=0.5)'''

        '''light_regions = [(630, 650), (840, 880), (995, 1015)]
        for start, end in light_regions:
            plt.axvspan(start, end, color='red', alpha=0.5)'''

        shaded_regions = [(400, 2000)]
        for start, end in shaded_regions:
            plt.axvspan(start, end, color='lightgray', alpha=0.5)

        dashed_lines_y = [0]
        for y_position in dashed_lines_y:
            plt.axhline(y_position, color='gray', linestyle='--', linewidth=0.5)

        plt.title(title)
        plt.xlabel('Raman Shift')
        plt.ylabel('Intensity')
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        if folder1 == None or folder2 == None:
            plt.show()
        else:
            #plt.show()
            plt.savefig(f"./output/{folder1}/{folder2}/{title}.png")
        plt.close()

    def apply_pipeline(self, raman_spectrum, title, smooth, folder1, folder2, *intervals):
        spectral_d = {}
        
        original_spectrum = copy.deepcopy(raman_spectrum)

        crop_spectrum = self.cropper(copy.deepcopy(original_spectrum), *intervals)
        spectral_d["Original"] = crop_spectrum

        wh_spectrum = self.remove_spikes_extended(copy.deepcopy(crop_spectrum))
        spectral_d["Whitaker Hayes"] = wh_spectrum

        no_negatives = self.get_negatives(copy.deepcopy(wh_spectrum))
        spectral_d["No negatives"] = no_negatives

        sg_spectrum = self.denoise_spectrum(copy.deepcopy(no_negatives))
        spectral_d["Savitzky Golay"] = sg_spectrum

        baseline_DRPLS = self.DRPLS(copy.deepcopy(sg_spectrum), smoothing=smooth)
        spectral_d["DRPLS Baseline"] = baseline_DRPLS[0]
        spectral_d["DRPLS Correction"] = baseline_DRPLS[1]

        final_spectrum = self.cof_normalize(copy.deepcopy(baseline_DRPLS[1]))
        preprocessed_spectrum = copy.deepcopy(copy.deepcopy(final_spectrum))

        '''if folder1 == None or folder2 == None:
            pass
        else:
            self.plot_spectrum(spectral_d, title, folder1, folder2 + '/samples')'''
        return preprocessed_spectrum

    def plot_mean_spectra_by_label_and_concentration(self, df, folder, subfolder):
        labels = df['Label'].unique()

        for label in labels:
            label_df = df[df['Label'] == label]
            concentrations = sorted(label_df['mg/L'].unique())  # Sort for consistency
            mean_spectra_by_concentration = {}

            for concentration in concentrations:
                subset = label_df[label_df['mg/L'] == concentration]
                mean_spectrum = subset.iloc[:, :-3].mean(axis=0)
                mean_spectra_by_concentration[concentration] = mean_spectrum

            # Get colormap colors
            cmap = cm.get_cmap('cool', len(concentrations))
            colors = [cmap(i) for i in range(len(concentrations))]

            # Plotting
            plt.figure(figsize=(10, 6))
            for idx, (concentration, mean_spectrum) in enumerate(mean_spectra_by_concentration.items()):
                plt.plot(mean_spectrum.index.astype(float), mean_spectrum.values, 
                        label=f'{concentration} mg/L', color=colors[idx])

            plt.xlabel(r'Raman Shift (cm$^{-1}$)', fontsize=16)
            plt.ylabel('Intensity (a.u.)', fontsize=16)
            plt.title(f'{label.capitalize()} Mean Spectra by Concentration', fontsize=20)
            plt.legend(fontsize=12)
            plt.grid(False)
            plt.ylim(0, 3)
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)

            plt.savefig(f"./output/{folder}/{subfolder}/mean_spectra_by_concentration_{label}.png", dpi=3000)
            plt.show()

preprocessor = RamanPreprocessor()

"""
Visualization
"""

def apply_tsne(final_df, folder1, subfolder2):
    X_train = final_df.iloc[:, :-1]
    y_train = final_df.iloc[:, -1]#.map({'control': 0, 'mix': 1, 'sulfamethoxazole': 2, 'sulfapyridine': 3, 'sulfathiazole': 4}).values
    # The learning rate for t-SNE is usually in the range [10.0, 1000.0]
    tsne = TSNE(n_components=2, random_state=42, perplexity=10, init='pca', n_iter=10000, n_iter_without_progress=1000, early_exaggeration=24, learning_rate=100, method='exact', n_jobs=-1)
    X_embedded_train = tsne.fit_transform(X_train)
    plt.figure(figsize=(12, 8))
    sns.scatterplot(x=X_embedded_train[:, 0], y=X_embedded_train[:, 1], hue=y_train)#, palette='viridis')
    plt.title('t-SNE Visualization of X_train')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.legend()
    plt.savefig(f"./output/{folder1}/{subfolder2}/tsne_2d.png")
    plt.show()
    plt.close()

def apply_3d_pca(final_df, folder1, subfolder2):
    # Standardize features and apply PCA
    features = final_df.iloc[:, :-3]
    standardized_features = StandardScaler().fit_transform(features)
    principal_components = PCA(n_components=3,random_state=42).fit_transform(standardized_features)
    # Create a DataFrame with principal components and labels
    pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2', 'PC3'])
    pca_df['Label'] = final_df['Label']
    # Explained variance ratio for axis labels
    explained_variance_ratio = PCA(n_components=3).fit(standardized_features).explained_variance_ratio_
    # Plotting the 3D PCA results
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    unique_labels = pca_df['Label'].unique()
    for label in unique_labels:
        subset_df = pca_df[pca_df['Label'] == label]
        ax.scatter(subset_df['PC1'], subset_df['PC2'], subset_df['PC3'], label=label)
    # Set axis labels with explained variance ratio
    ax.set_xlabel(f'PC1: {explained_variance_ratio[0]:.2f}', labelpad=8.0)
    ax.set_ylabel(f'PC2: {explained_variance_ratio[1]:.2f}', labelpad=8.0)
    ax.set_zlabel(f'PC3: {explained_variance_ratio[2]:.2f}', labelpad=8.0)
    ax.set_title('PCA')
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    # Save and show the plot
    plt.savefig(f"./output/{folder1}/{subfolder2}/pca_{unique_labels[0]}.png")
    plt.show()
    plt.close()
    return pca_df

def apply_3d_umap(final_df, folder1, subfolder2):
    import umap
    import umap.plot
    from sklearn.preprocessing import StandardScaler
    import pandas as pd
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    label_mapping = {'control': 0.0, 'mixture': 1.0, 'mix': 1.0, 'sulfamethoxazole': 2.0, 'sulfapyridine': 3.0, 'sulfathiazole': 4.0}
    features = final_df.iloc[:, :-3]
    standardized_features = StandardScaler().fit_transform(features)
    umap_embedding = umap.UMAP(n_components=3, random_state=42, n_neighbors=100).fit_transform(standardized_features)#, final_df['Label'].map(label_mapping))
    mapper = umap.UMAP(n_components=2, random_state=42, n_neighbors=100).fit(standardized_features)#, final_df['Label'].map(label_mapping))
    
    # Create a DataFrame with UMAP components and labels
    umap_df = pd.DataFrame(data=umap_embedding, columns=['UMAP1', 'UMAP2', 'UMAP3'])
    umap_df['Label'] = final_df['Label']
    
    umap.plot.points(mapper, labels=final_df['Label'], theme='fire')
    umap.plot.connectivity(mapper, labels=final_df['Label'], theme='fire', show_points=True)
    plt.show()

    # Plotting the 3D UMAP results
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    unique_labels = umap_df['Label'].unique()
    
    for label in unique_labels:
        subset_df = umap_df[umap_df['Label'] == label]
        ax.scatter(subset_df['UMAP1'], subset_df['UMAP2'], subset_df['UMAP3'], label=label)
    
    # Set axis labels
    ax.set_xlabel('UMAP1', labelpad=8.0)
    ax.set_ylabel('UMAP2', labelpad=8.0)
    ax.set_zlabel('UMAP3', labelpad=8.0)
    ax.set_title('UMAP')
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    
    # Save and show the plot
    #plt.savefig(f"./output/{folder1}/{subfolder2}/umap_{unique_labels[0]}.png")
    plt.show()
    plt.close()
    return umap_df

def apply_pca_with_90_variance(final_df):
    features = final_df.iloc[:, :-3]
    standardized_features = StandardScaler().fit_transform(features)
    pca = PCA(random_state=42)
    pca.fit(standardized_features)
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = explained_variance.cumsum()
    n_components = next(i for i, cumulative in enumerate(cumulative_variance) if cumulative >= 0.90) + 1
    pca = PCA(n_components=n_components, random_state=42)
    principal_components = pca.fit_transform(standardized_features)
    columns = [f'PC{i+1}' for i in range(n_components)]
    pca_df = pd.DataFrame(data=principal_components, columns=columns)
    pca_df['Label'] = final_df['Label']
    #pca_df['Name'] = final_df['Name']
    return pca_df

from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import r2_score
import pandas as pd
import numpy as np

def apply_pls_with_90_variance(final_df):
    features = final_df.iloc[:, :-3]
    labels = final_df['Label']

    # Standardize features
    standardized_features = StandardScaler().fit_transform(features)

    # Convert labels to numeric codes internally for PLS regression
    numeric_labels, uniques = pd.factorize(labels)
    numeric_labels = numeric_labels.reshape(-1, 1)  # shape (n_samples, 1)

    # Find number of components explaining >=90% of variance (R2)
    max_components = min(features.shape[1], 30)
    for n in range(1, max_components + 1):
        pls = PLSRegression(n_components=n)
        pls.fit(standardized_features, numeric_labels)
        y_pred = pls.predict(standardized_features)
        score = r2_score(numeric_labels, y_pred)
        if score >= 0.95:
            break
    else:
        n = max_components

    pls = PLSRegression(n_components=n)
    pls_components = pls.fit_transform(standardized_features, numeric_labels)[0]

    columns = [f'PLS{i+1}' for i in range(n)]
    pls_df = pd.DataFrame(pls_components, columns=columns)
    pls_df['Label'] = labels.reset_index(drop=True)

    return pls_df

def apply_pca_with_90_variance_2(feats):
    standardized_features = StandardScaler().fit_transform(feats)
    pca = PCA(random_state=42)
    pca.fit_transform(standardized_features)
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = explained_variance.cumsum()
    n_components = next(i for i, cumulative in enumerate(cumulative_variance) if cumulative >= 0.90) + 1
    pca = PCA(n_components=n_components, random_state=42)
    principal_components = pca.fit_transform(standardized_features)
    columns = [f'PC{i+1}' for i in range(n_components)]
    pca_df = pd.DataFrame(data=principal_components, columns=columns)
    return pca_df

def apply_lda(final_df, folder1, subfolder2):
    features = final_df.iloc[:, :-3]
    standardized_features = StandardScaler().fit_transform(features)
    # Check the number of unique classes
    unique_labels = final_df['Label'].nunique()
    # If there are only two classes, set n_components to 1, otherwise to 2
    n_components = 1 if unique_labels == 2 else 2
    lda = LDA(n_components=n_components)
    # Perform LDA transformation
    principal_components = lda.fit_transform(standardized_features, final_df['Label'])
    # Create a dataframe for the principal components
    if n_components == 1:
        lda_df = pd.DataFrame(data=principal_components, columns=['LD1'])
        lda_df['LD2'] = 0  # Add a dummy column for LD2 for consistent plotting
    else:
        lda_df = pd.DataFrame(data=principal_components, columns=['LD1', 'LD2'])
    lda_df['Label'] = final_df['Label']
    explained_variance_ratio = lda.explained_variance_ratio_
    # Plotting
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111)
    unique_labels = lda_df['Label'].unique()
    for label in unique_labels:
        subset_df = lda_df[lda_df['Label'] == label]
        ax.scatter(subset_df['LD1'], subset_df['LD2'], label=label)
    # Set axis labels with explained variance ratio
    ax.set_xlabel(f'LD1: {explained_variance_ratio[0]:.2f}', labelpad=8.0)
    # Only set the y-axis label if we have more than one component
    if n_components == 2:
        ax.set_ylabel(f'LD2: {explained_variance_ratio[1]:.2f}', labelpad=8.0)
    ax.set_title('LDA')
    #ax.legend(loc='upper right')
    ax.legend()
    # Save and close the plot
    plt.savefig(f"{'./output/'}{folder1}/{subfolder2}/lda_{unique_labels[0]}.png")
    plt.show()
    plt.close()
    return lda_df

def apply_gda(final_df, folder1, subfolder2):
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    features = final_df.iloc[:, :-3]
    labels = final_df['Label']
    # Split data into training (80%) and testing (20%) sets
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    # Fit GDA model
    gda = QDA()
    gda.fit(X_train, y_train)
    predictions = gda.predict(X_test)
    # Evaluate model performance
    accuracy = accuracy_score(y_test, predictions)
    class_report = classification_report(y_test, predictions)
    conf_matrix = confusion_matrix(y_test, predictions)
    print(f'Accuracy: {accuracy:.4f}')
    print('Classification Report:\n', class_report)
    print('Confusion Matrix:\n', conf_matrix)
    # Create a dataframe with predictions
    gda_df = pd.DataFrame({'Predicted_Label': predictions, 'True_Label': y_test})
    return gda_df


"""
Sample Generator
"""

def oversample(df):
    '''Only working for binary classification'''
    X, y = df.iloc[:, :-2].values, df.iloc[:, -2].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    smoteenn = SMOTEENN(sampling_strategy='all', random_state=42) #'all', 'auto', minority
    X_resampled, y_resampled = smoteenn.fit_resample(X_scaled, y)
    columns = df.columns[:-2]
    df_resampled = pd.DataFrame(data=X_resampled, columns=columns)
    df_resampled['Label'] = y_resampled
    return df_resampled

"""
ML
"""

def exploratory_ml(df, folder1=None, subfolder2=None):
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=99, stratify=y) #(X, y, test_size=0.3, random_state=99, stratify=y)
    X_train, y_train = shuffle(X_train, y_train)
    clf = LazyClassifier(verbose=0,ignore_warnings=True, custom_metric=None)
    models_test, predictions_test = clf.fit(X_train, X_test, y_train, y_test)
    print(models_test)
    '''if folder1 is not None and subfolder2 is not None:
        models_test.to_excel(f"{'./output/'}{folder1}/{subfolder2}/models.xlsx")'''
    sns.set_theme(style='whitegrid')
    plt.figure(figsize=(5, 10))
    models_test['Accuracy (%)'] = models_test['Accuracy']*100
    ax = sns.barplot(y=models_test.index, x='Accuracy (%)', data=models_test)
    for i in ax.containers:
        ax.bar_label(i, fmt='%.2f')
    plt.show()


"""
Others
"""

def create_folders(directory, folder, subfolder=None):
    folder_path = os.path.join(directory, folder)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        if subfolder:
            for subfolder_name in subfolder:
                subfolder_path = os.path.join(folder_path, subfolder_name)
                os.makedirs(subfolder_path)
    else:
        pass
    return folder, subfolder
