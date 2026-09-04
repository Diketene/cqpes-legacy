import os
from typing import Dict

import numpy as np
import pandas as pd
import scienceplots  # noqa: F401
from matplotlib import pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

from cqpes.types import CQPESData, TrainConfig
from cqpes.utils.train import find_best_checkpoint
from cqpes.utils.workspace import ExperimentWorkspace

eV_to_wavenumber = 8065.541

_ERR_SCALE = {"eV": 1e3, "wavenumber": eV_to_wavenumber}
_ENE_SCALE = {"eV": 1.0, "wavenumber": eV_to_wavenumber}
_ERR_UNIT = {"eV": "meV", "wavenumber": "cm-1"}


def run_test(
    workdir_path: str,
    unit: str = "eV",
) -> None:
    if unit not in _ERR_SCALE:
        raise ValueError(f"unsupported unit: {unit!r} (choose 'eV' or 'wavenumber')")
    
    # lazy import
    from cqpes._env import _setup_tensorflow

    _setup_tensorflow()

    import tf_levenberg_marquardt as lm

    from cqpes.utils.model import build_network

    # 1. existing workspace
    workspace = ExperimentWorkspace.from_existing(workdir_path)
    eval_dir = workspace.get_subpath("eval")

    # 2. load metadata
    train_config = TrainConfig.from_json(
        os.path.join(
            workspace.path,
            "train.json",
        )
    )

    phys_dict = {
        k: np.load(os.path.join(workspace.path, f"{k}.npy"))
        for k in ["p_min", "p_max", "V_min", "V_max"]
    }

    subset_idx_map = {
        name: np.loadtxt(
            os.path.join(workspace.path, f"{name.lower()}_idx.txt"),
            dtype=np.int32,
        )
        for name in ["Train", "Valid", "Test"]
    }

    # 3. find best checkpoint
    best_ckpt_path, label = find_best_checkpoint(workspace.path)

    print(f"  [{'WORKDIR':^10}] {workspace.path}")
    print(f"  [{'MODEL':^10}] Target: {os.path.basename(best_ckpt_path)}")

    # 4. build network
    input_dim = len(phys_dict["p_min"]) - 1
    model = build_network(train_config, input_dim=input_dim)
    model_wrapper = lm.model.ModelWrapper(model)  # type: ignore
    model_wrapper.build(input_shape=(1, input_dim))
    model_wrapper.load_weights(best_ckpt_path)

    # 5. estimate error
    dataset = CQPESData.from_dir(train_config.data)
    X_scaled, V_true = dataset.X[:, 1:], dataset.V.reshape((-1, 1))
    y = model_wrapper(X_scaled, training=False).numpy()
    V_pred = CQPESData.unscale(y, phys_dict["V_min"], phys_dict["V_max"])

    errors = (V_pred - V_true) * _ERR_SCALE[unit]

    file_prefix = f"{os.path.basename(workspace.path)}_{label}_{unit}"

    _export_metrics(V_true, V_pred, subset_idx_map, eval_dir, file_prefix, unit)
    _plot_diagnostics(V_true, errors, subset_idx_map, eval_dir, file_prefix, unit)


def _export_metrics(
    V_true: np.ndarray,
    V_pred: np.ndarray,
    subset_idx_map: Dict[str, np.ndarray],
    output_dir: str,
    file_prefix: str,
    unit: str,
) -> None:
    stats = []
    eval_indices = {**subset_idx_map, "Total": np.arange(len(V_true))}

    for name, idx in eval_indices.items():
        y_t, y_p = V_true[idx] * _ERR_SCALE[unit], V_pred[idx] * _ERR_SCALE[unit]

        stats.append(
            {
                "Set": name,
                f"MAE ({_ERR_UNIT[unit]})": mean_absolute_error(y_t, y_p),
                f"RMSE ({_ERR_UNIT[unit]})": np.sqrt(mean_squared_error(y_t, y_p)),
                f"MaxErr ({_ERR_UNIT[unit]})": np.abs(y_t - y_p).max(),
            }
        )

    df = pd.DataFrame(stats)
    csv_path = os.path.join(output_dir, f"{file_prefix}_metrics.csv")
    df.to_csv(csv_path, index=False)

    print(f"  [{'METRICS':^10}] Stats saved to: {csv_path}")
    print("\n" + df.to_string(index=False) + "\n")


def _plot_error_scatter(
    V_true,
    errors,
    subset_idx_map: Dict[str, np.ndarray],
    output_dir: str,
    file_prefix: str,
    unit: str,
) -> None:
    plot_path = os.path.join(output_dir, f"{file_prefix}_scatter.png")

    print(f"  [{'PLOT':^10}] Generating scatter plot...")

    colors = {"Train": "b", "Valid": "g", "Test": "r"}

    with plt.style.context(["science", "no-latex"]):
        fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

        for name, idx in subset_idx_map.items():
            ax.scatter(
                V_true[idx] * _ENE_SCALE[unit],
                errors[idx],
                c=colors[name],
                alpha=0.5,
                label=name,
                s=1.0,
            )

        ax.axhline(0, color="#c0392b", linestyle="--", linewidth=1.5)
        if unit == "eV":
            ax.set_xlabel(r"$\mathrm{Ab \ Initio \ Energy \ (eV)}$")
            ax.set_ylabel(r"$\mathrm{Error \ (meV)}$")
        elif unit == "wavenumber":
            ax.set_xlabel(r"$\mathrm{Ab \ Initio \ Energy \ (cm^{-1})}$")
            ax.set_ylabel(r"$\mathrm{Error \ (cm^{-1})}$")

        ax.legend(loc="upper right", frameon=True)

        plt.savefig(plot_path, bbox_inches="tight")
        plt.close(fig)

        print(f"  [{'SAVE':^10}] Scatter plot saved as: {plot_path}")


def _plot_error_dist(
    errors: np.ndarray,
    output_dir: str,
    file_prefix: str,
    unit: str,
) -> None:
    plot_path = os.path.join(output_dir, f"{file_prefix}_hist.png")

    print(f"  [{'PLOT':^10}] Generating histogram...")

    abs_err = np.abs(errors).flatten()
    upper_bound = np.percentile(abs_err, 99.5)
    max_err = np.ceil(upper_bound)
    if unit == "eV":
        bin_width = 0.2 if max_err < 10 else 0.5
    elif unit == "wavenumber":
        n_bins = 40
        bin_width = max(max_err / n_bins, 1e-6)
    
    edges = np.arange(0.0, max_err + bin_width, bin_width)

    with plt.style.context(["science", "no-latex"]):
        fig, ax = plt.subplots(figsize=(7, 5), dpi=300)

        weights = np.ones_like(abs_err) / len(abs_err)

        ax.hist(
            abs_err,
            bins=edges,  # type: ignore
            weights=weights,
            color="b",
            edgecolor="white",
            linewidth=0.8,
            rwidth=0.9,
        )

        if unit == "eV":
            ax.set_xlabel(r"$\mathrm{Fitting \ Error \ (meV)}$", fontsize=12, fontweight="bold")
        elif unit == "wavenumber":
            ax.set_xlabel(r"$\mathrm{Fitting \ Error \ (cm^{-1})}$", fontsize=12, fontweight="bold")

        ax.set_ylabel(r"$\mathrm{Distribution}$", fontsize=12, fontweight="bold")

        mae = np.mean(abs_err).item()

        ax.axvline(
            mae,
            color="#e74c3c",
            linestyle="-",
            linewidth=1.5,
            label=f"MAE: {mae:.2f}",
        )

        plt.legend()

        plt.savefig(plot_path, bbox_inches="tight")
        plt.close(fig)

        print(f"  [{'SAVE':^10}] Histogram saved as: {plot_path}")


def _plot_diagnostics(
    V_true: np.ndarray,
    errors: np.ndarray,
    subset_idx_map: Dict[str, np.ndarray],
    output_dir: str,
    file_prefix: str,
    unit: str
) -> None:
    _plot_error_scatter(
        V_true=V_true,
        errors=errors,
        subset_idx_map=subset_idx_map,
        output_dir=output_dir,
        file_prefix=file_prefix,
        unit=unit,
    )

    _plot_error_dist(
        errors=errors,
        output_dir=output_dir,
        file_prefix=file_prefix,
        unit=unit,
    )
