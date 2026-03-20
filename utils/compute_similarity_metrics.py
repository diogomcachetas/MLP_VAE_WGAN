import numpy as np
import pandas as pd
from scipy.stats import entropy, wasserstein_distance
from pathlib import Path


# ============================================================
# Configuration
# ============================================================
REAL_PATH = "./Desktop/MLP-VAE-WGAN/data/ML_DL_Dataset/X_train.npy"
GEN_PATH = "./Desktop/MLP-VAE-WGAN/data/ML_DL_Dataset/generated_samples.npy"
LABEL_PATH = "./Desktop/MLP-VAE-WGAN/data/ML_DL_Dataset/y_train.npy"

N_BINS_KL = 50
EPS = 1e-10
SAVE_CSV = True


# ============================================================
# Utility functions
# ============================================================
def validate_inputs(X_real, X_gen, y):
    if X_real.shape != X_gen.shape:
        raise ValueError(
            f"Shape mismatch: X_train {X_real.shape} vs generated_samples {X_gen.shape}"
        )
    if X_real.ndim != 2:
        raise ValueError(
            f"Expected 2D arrays for X_train and generated_samples, got {X_real.ndim}D"
        )
    if y.ndim != 1:
        raise ValueError(f"Expected 1D labels array, got {y.ndim}D")
    if len(y) != X_real.shape[0]:
        raise ValueError(
            f"Label length mismatch: len(y)={len(y)} vs n_samples={X_real.shape[0]}"
        )


def paired_sample_errors(X_real, X_gen):
    diff = X_real - X_gen
    sample_mse = np.mean(diff ** 2, axis=1)
    sample_mae = np.mean(np.abs(diff), axis=1)
    return sample_mse, sample_mae


def symmetric_kl_featurewise(X1, X2, n_bins=50, eps=1e-10):
    """
    Computes feature-wise symmetric KL divergence using histograms:
    0.5 * [KL(P||Q) + KL(Q||P)]
    Returns an array of length n_features.
    """
    n_features = X1.shape[1]
    kl_values = np.zeros(n_features, dtype=np.float64)

    for j in range(n_features):
        a = X1[:, j]
        b = X2[:, j]

        lo = min(a.min(), b.min())
        hi = max(a.max(), b.max())

        if np.isclose(lo, hi):
            kl_values[j] = 0.0
            continue

        hist_a, bin_edges = np.histogram(a, bins=n_bins, range=(lo, hi), density=False)
        hist_b, _ = np.histogram(b, bins=bin_edges, density=False)

        p = hist_a.astype(np.float64) + eps
        q = hist_b.astype(np.float64) + eps

        p /= p.sum()
        q /= q.sum()

        kl_pq = entropy(p, q)
        kl_qp = entropy(q, p)
        kl_values[j] = 0.5 * (kl_pq + kl_qp)

    return kl_values


def wasserstein_featurewise(X1, X2):
    """
    Computes 1D Wasserstein distance for each feature independently.
    Returns an array of length n_features.
    """
    n_features = X1.shape[1]
    wd_values = np.zeros(n_features, dtype=np.float64)

    for j in range(n_features):
        wd_values[j] = wasserstein_distance(X1[:, j], X2[:, j])

    return wd_values


def summarise_vector(x):
    x = np.asarray(x, dtype=np.float64)
    if len(x) == 1:
        return float(x[0]), 0.0
    return float(np.mean(x)), float(np.std(x, ddof=1))


def compute_metrics_for_subset(X_real, X_gen, name, n_bins=50, eps=1e-10):
    sample_mse, sample_mae = paired_sample_errors(X_real, X_gen)
    kl_vals = symmetric_kl_featurewise(X_real, X_gen, n_bins=n_bins, eps=eps)
    wd_vals = wasserstein_featurewise(X_real, X_gen)

    mse_mean, mse_std = summarise_vector(sample_mse)
    mae_mean, mae_std = summarise_vector(sample_mae)
    kl_mean, kl_std = summarise_vector(kl_vals)
    wd_mean, wd_std = summarise_vector(wd_vals)

    return {
        "group": name,
        "n_samples": X_real.shape[0],
        "n_features": X_real.shape[1],
        "MSE_mean": mse_mean,
        "MSE_std": mse_std,
        "MAE_mean": mae_mean,
        "MAE_std": mae_std,
        "KL_sym_mean": kl_mean,
        "KL_sym_std": kl_std,
        "Wasserstein_mean": wd_mean,
        "Wasserstein_std": wd_std,
    }


def pretty_print_table(df, title):
    print("\n" + "=" * len(title))
    print(title)
    print("=" * len(title))
    print(df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))


# ============================================================
# Main
# ============================================================
def main():
    # Load data
    X_real = np.load(REAL_PATH)
    X_gen = np.load(GEN_PATH)
    y = np.load(LABEL_PATH)

    validate_inputs(X_real, X_gen, y)

    print("Loaded data successfully:")
    print(f"  X_train shape            : {X_real.shape}")
    print(f"  generated_samples shape  : {X_gen.shape}")
    print(f"  y_train shape            : {y.shape}")

    # Overall metrics
    overall_result = compute_metrics_for_subset(
        X_real, X_gen, name="overall", n_bins=N_BINS_KL, eps=EPS
    )

    # Per-class metrics
    classes = np.unique(y)
    class_results = []

    for cls in classes:
        idx = y == cls
        result = compute_metrics_for_subset(
            X_real[idx],
            X_gen[idx],
            name=f"class_{cls}",
            n_bins=N_BINS_KL,
            eps=EPS,
        )
        class_results.append(result)

    # Build tables
    overall_df = pd.DataFrame([overall_result])
    class_df = pd.DataFrame(class_results)

    # Add formatted summary columns
    for df in [overall_df, class_df]:
        df["MSE"] = df.apply(lambda r: f"{r['MSE_mean']:.6f} ± {r['MSE_std']:.6f}", axis=1)
        df["MAE"] = df.apply(lambda r: f"{r['MAE_mean']:.6f} ± {r['MAE_std']:.6f}", axis=1)
        df["KL_sym"] = df.apply(lambda r: f"{r['KL_sym_mean']:.6f} ± {r['KL_sym_std']:.6f}", axis=1)
        df["Wasserstein"] = df.apply(
            lambda r: f"{r['Wasserstein_mean']:.6f} ± {r['Wasserstein_std']:.6f}", axis=1
        )

    overall_display = overall_df[
        ["group", "n_samples", "n_features", "MSE", "MAE", "KL_sym", "Wasserstein"]
    ]
    class_display = class_df[
        ["group", "n_samples", "n_features", "MSE", "MAE", "KL_sym", "Wasserstein"]
    ]

    pretty_print_table(overall_display, "Overall metrics")
    pretty_print_table(class_display, "Per-class metrics")

    if SAVE_CSV:
        overall_df.to_csv("overall_metrics.csv", index=False)
        class_df.to_csv("per_class_metrics.csv", index=False)
        print("\nSaved:")
        print("  overall_metrics.csv")
        print("  per_class_metrics.csv")


if __name__ == "__main__":
    main()