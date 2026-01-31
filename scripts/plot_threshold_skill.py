import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from monet_stats.contingency_metrics import scores


def plot_threshold_skill_profile():
    """
    Example script to plot threshold skill profile (Aero Protocol Track A).
    """
    # Create synthetic spatial data
    lats = np.linspace(-90, 90, 50)
    lons = np.linspace(-180, 180, 50)
    obs = xr.DataArray(
        np.exp(-((lats[:, None] - 0) ** 2 + (lons[None, :] - 0) ** 2) / 1000),
        coords={"lat": lats, "lon": lons},
        dims=("lat", "lon"),
    )
    mod = obs + np.random.normal(0, 0.1, obs.shape)

    thresholds = np.linspace(0.1, 0.9, 20)

    # Vectorized computation for all thresholds at once!
    a, b, c, d = scores(obs, mod, minval=thresholds)

    # Compute metrics from counts
    denom_hss = (a + c) * (c + d) + (a + b) * (b + d)
    hss_vals = np.where(denom_hss > 0, 2 * (a * d - b * c) / denom_hss, np.nan)

    denom_pod = a + b
    pod_vals = np.where(denom_pod > 0, a / denom_pod, np.nan)

    denom_far = a + c
    far_vals = np.where(denom_far > 0, c / denom_far, np.nan)

    # Plotting (Track A: Static)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(thresholds, hss_vals, "o-", label="HSS", linewidth=2)
    ax.plot(thresholds, pod_vals, "s--", label="POD", alpha=0.7)
    ax.plot(thresholds, far_vals, "d-.", label="FAR", alpha=0.7)

    ax.set_title("Categorical Skill Profile vs. Threshold (Vectorized)", fontsize=14)
    ax.set_xlabel("Threshold Value", fontsize=12)
    ax.set_ylabel("Skill Score", fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend()

    plt.savefig("threshold_skill_profile.png")
    print("Plot saved to threshold_skill_profile.png")


if __name__ == "__main__":
    plot_threshold_skill_profile()
