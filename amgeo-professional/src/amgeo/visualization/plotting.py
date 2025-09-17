# src/amgeo/visualization/plotting.py
"""
Professional visualization tools for VES analysis
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, Tuple, List
import logging

logger = logging.getLogger(__name__)

# Set professional plotting style
plt.style.use("default")
sns.set_palette("husl")

def create_ves_sounding_plot(ab2, rhoa, fitted_rhoa=None, **kwargs):
    """Convenience function to create VES sounding plot"""
    plotter = VESPlotter()
    return plotter.plot_ves_curve(ab2, rhoa, fitted_rhoa, **kwargs)


def create_comprehensive_plot(inversion_result, ml_result=None, site_info=None):
    """Convenience function to create comprehensive results plot"""
    plotter = VESPlotter()
    return plotter.plot_comprehensive_results(inversion_result, ml_result, site_info)

class VESPlotter:
    """Professional VES data and results visualization"""

    def __init__(self, style: str = "professional"):
        self.style = style
        self.colors = {
            "observed": "#2E86AB",
            "calculated": "#A23B72",
            "model": "#F18F01",
            "uncertainty": "#C73E1D",
            "aquifer": "#3498DB",
            "confining": "#E74C3C",
        }

        # Configure matplotlib for professional output
        plt.rcParams.update(
            {
                "figure.figsize": (12, 8),
                "font.size": 11,
                "axes.linewidth": 1.2,
                "axes.spines.top": False,
                "axes.spines.right": False,
                "axes.grid": True,
                "grid.alpha": 0.3,
                "legend.frameon": True,
                "legend.fancybox": True,
                "legend.shadow": True,
            }
        )

    def plot_ves_curve(
        self,
        ab2: np.ndarray,
        rhoa_obs: np.ndarray,
        rhoa_calc: Optional[np.ndarray] = None,
        title: str = "VES Sounding Curve",
    ) -> plt.Figure:
        """Create professional VES sounding curve plot"""

        fig, ax = plt.subplots(figsize=(10, 8))

        # Observed data
        ax.loglog(
            ab2,
            rhoa_obs,
            "o",
            color=self.colors["observed"],
            markersize=8,
            markeredgecolor="white",
            markeredgewidth=1,
            label="Observed",
            alpha=0.8,
        )

        # Calculated data (if available)
        if rhoa_calc is not None:
            ax.loglog(
                ab2,
                rhoa_calc,
                "-",
                color=self.colors["calculated"],
                linewidth=3,
                label="Calculated",
                alpha=0.9,
            )

            # Calculate and display fit quality
            rms = np.sqrt(np.mean(((rhoa_obs - rhoa_calc) / rhoa_obs) ** 2)) * 100
            ax.text(
                0.02,
                0.98,
                f"RMS Error: {rms:.1f}%",
                transform=ax.transAxes,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
                fontweight="bold",
            )

        ax.set_xlabel("AB/2 (m)", fontsize=14, fontweight="bold")
        ax.set_ylabel("Apparent Resistivity (Ω·m)", fontsize=14, fontweight="bold")
        ax.set_title(title, fontsize=16, fontweight="bold", pad=20)
        ax.legend(loc="best", fontsize=12)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_resistivity_model(
        self,
        resistivities: np.ndarray,
        thicknesses: np.ndarray,
        depths: Optional[np.ndarray] = None,
        title: str = "Resistivity Model",
    ) -> plt.Figure:
        """Create professional resistivity model plot"""

        fig, ax = plt.subplots(figsize=(8, 10))

        if depths is None:
            depths = np.concatenate([[0], np.cumsum(thicknesses)])

        # Create step plot
        depth_plot = []
        res_plot = []

        for i, (depth, res) in enumerate(zip(depths[:-1], resistivities[:-1])):
            depth_plot.extend([depth, depths[i + 1]])
            res_plot.extend([res, res])

        # Add final layer
        if len(depths) > 0:
            max_depth = depths[-1] * 1.5 if depths[-1] > 0 else 100
            depth_plot.extend([depths[-1], max_depth])
            res_plot.extend([resistivities[-1], resistivities[-1]])

        ax.semilogx(
            res_plot,
            depth_plot,
            color=self.colors["model"],
            linewidth=4,
            label="Resistivity Model",
        )

        # Shade aquifer zones (20-500 Ω·m)
        for i, res in enumerate(resistivities):
            if 20 <= res <= 500:
                top = depths[i] if i < len(depths) else 0
                bottom = depths[i + 1] if i + 1 < len(depths) else max_depth
                ax.axhspan(
                    top,
                    bottom,
                    alpha=0.3,
                    color=self.colors["aquifer"],
                    label=(
                        "Potential Aquifer"
                        if i == 0
                        or "Potential Aquifer"
                        not in [
                            t.get_text()
                            for t in ax.get_legend().get_texts()
                            if ax.get_legend()
                        ]
                        else ""
                    ),
                )

        # Add layer annotations
        for i, (res, depth) in enumerate(zip(resistivities, depths)):
            mid_depth = (
                depth + (depths[i + 1] - depth) / 2
                if i + 1 < len(depths)
                else depth + 10
            )
            ax.annotate(
                f"L{i+1}\n{res:.0f} Ω·m",
                xy=(res, mid_depth),
                xytext=(10, 0),
                textcoords="offset points",
                fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                ha="left",
                va="center",
            )

        ax.set_xlabel("Resistivity (Ω·m)", fontsize=14, fontweight="bold")
        ax.set_ylabel("Depth (m)", fontsize=14, fontweight="bold")
        ax.set_title(title, fontsize=16, fontweight="bold", pad=20)
        ax.invert_yaxis()
        ax.legend(loc="best", fontsize=12)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_comprehensive_results(
        self,
        inversion_result,
        ml_result: Optional[Dict] = None,
        site_info: Optional[Dict] = None,
        ) ->plt.Figure:
        """Create comprehensive results dashboard"""

        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        # VES curve fit
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_ves_curve_subplot(ax1, inversion_result)

        # Resistivity model
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_resistivity_model_subplot(ax2, inversion_result)

        # ML prediction (if available)
        if ml_result:
            ax3 = fig.add_subplot(gs[0, 2])
            self._plot_ml_prediction_subplot(ax3, ml_result)

        # Residuals analysis
        ax4 = fig.add_subplot(gs[1, :2])
        self._plot_residuals_subplot(ax4, inversion_result)

        # Layer properties
        ax5 = fig.add_subplot(gs[1, 2])
        self._plot_layer_properties_subplot(ax5, inversion_result)

        # Summary statistics
        ax6 = fig.add_subplot(gs[2, :])
        self._plot_summary_table(ax6, inversion_result, ml_result, site_info)

        # Overall title
        site_name = site_info.get("site_name", "VES Site") if site_info else "VES Site"
        fig.suptitle(
            f"{site_name} - Professional VES Analysis Report",
            fontsize=18,
            fontweight="bold",
            y=0.95,
        )

        return fig

    def _plot_ves_curve_subplot(self, ax, result):
        """Plot VES curve in subplot"""
        ax.loglog(
            result.ab2,
            result.rhoa,
            "o",
            color=self.colors["observed"],
            markersize=6,
            label="Observed",
        )
        ax.loglog(
            result.ab2,
            result.fitted_rhoa,
            "-",
            color=self.colors["calculated"],
            linewidth=2,
            label="Calculated",
        )

        ax.set_xlabel("AB/2 (m)", fontweight="bold")
        ax.set_ylabel("App. Resistivity (Ω·m)", fontweight="bold")
        ax.set_title(f"VES Data Fit (RMS: {result.rms_error:.1f}%)", fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    def _plot_resistivity_model_subplot(self, ax, result):
        """Plot resistivity model in subplot"""
        depths = result.depths
        resistivities = result.resistivities

        # Create step plot
        depth_plot = []
        res_plot = []

        for i in range(len(resistivities)):
            if i < len(depths) - 1:
                depth_plot.extend([depths[i], depths[i + 1]])
                res_plot.extend([resistivities[i], resistivities[i]])
            else:
                max_depth = depths[-1] * 1.5 if len(depths) > 0 else 100
                depth_plot.extend([depths[i] if i < len(depths) else 0, max_depth])
                res_plot.extend([resistivities[i], resistivities[i]])

        ax.semilogx(res_plot, depth_plot, color=self.colors["model"], linewidth=3)
        ax.set_xlabel("Resistivity (Ω·m)", fontweight="bold")
        ax.set_ylabel("Depth (m)", fontweight="bold")
        ax.set_title("Resistivity vs Depth", fontweight="bold")
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3)

    def _plot_ml_prediction_subplot(self, ax, ml_result):
        """Plot ML prediction gauge in subplot"""
        prob = ml_result.get("aquifer_probability", [0])[0]
        uncertainty = ml_result.get("total_uncertainty", [0])[0]

        # Create probability gauge
        theta = np.linspace(0, np.pi, 100)
        r = 1

        # Background semicircle
        ax.plot(r * np.cos(theta), r * np.sin(theta), "k-", linewidth=2)
        ax.fill_between(
            r * np.cos(theta), 0, r * np.sin(theta), alpha=0.1, color="gray"
        )

        # Probability arc
        prob_theta = prob * np.pi
        prob_arc = np.linspace(0, prob_theta, max(1, int(prob * 50)))

        if len(prob_arc) > 0:
            color = (
                self.colors["aquifer"]
                if prob >= 0.6
                else "orange" if prob >= 0.4 else "red"
            )
            ax.plot(
                r * np.cos(prob_arc), r * np.sin(prob_arc), color=color, linewidth=6
            )

        # Labels
        ax.text(-1, -0.15, "0%", ha="center", fontweight="bold", fontsize=9)
        ax.text(1, -0.15, "100%", ha="center", fontweight="bold", fontsize=9)
        ax.text(0, 1.1, f"{prob:.1%}", ha="center", fontweight="bold", fontsize=12)
        ax.text(0, -0.4, f"±{uncertainty:.1%}", ha="center", fontsize=10)

        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-0.5, 1.3)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title("Aquifer Probability", fontweight="bold")

    def _plot_residuals_subplot(self, ax, result):
        """Plot residuals analysis in subplot"""
        residuals = (result.rhoa - result.fitted_rhoa) / result.rhoa * 100

        ax.semilogx(result.ab2, residuals, "o-", color="purple", markersize=4)
        ax.axhline(y=0, color="black", linestyle="-", alpha=0.5)
        ax.axhline(y=5, color="red", linestyle="--", alpha=0.7, label="±5%")
        ax.axhline(y=-5, color="red", linestyle="--", alpha=0.7)

        ax.set_xlabel("AB/2 (m)", fontweight="bold")
        ax.set_ylabel("Residual (%)", fontweight="bold")
        ax.set_title("Data Fit Residuals", fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    def _plot_layer_properties_subplot(self, ax, result):
        """Plot layer properties bar chart"""
        resistivities = result.resistivities
        n_layers = len(resistivities)

        layer_names = [f"L{i+1}" for i in range(n_layers)]
        colors = [
            "red" if r < 20 else "orange" if r < 100 else "green" if r < 500 else "blue"
            for r in resistivities
        ]

        bars = ax.bar(layer_names, resistivities, color=colors, alpha=0.7)
        ax.set_ylabel("Resistivity (Ω·m)", fontweight="bold")
        ax.set_title("Layer Resistivities", fontweight="bold")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3, axis="y")

        # Add value labels
        for bar, res in zip(bars, resistivities):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{res:.0f}",
                ha="center",
                va="bottom",
                fontweight="bold",
                fontsize=9,
            )

    def _plot_summary_table(self, ax, inversion_result, ml_result, site_info):
        """Create summary statistics table"""
        ax.axis("off")

        # Prepare summary data
        summary_data = [["Parameter", "Value", "Interpretation"]]

        # Inversion results
        summary_data.extend(
            [
                [
                    "RMS Error (%)",
                    f"{inversion_result.rms_error:.2f}",
                    "Data fit quality",
                ],
                ["Chi-squared", f"{inversion_result.chi2:.4f}", "Model-data misfit"],
                [
                    "Number of Layers",
                    f"{len(inversion_result.resistivities)}",
                    "Model complexity",
                ],
                [
                    "Investigation Depth (m)",
                    (
                        f"{inversion_result.depths[-1]:.0f}"
                        if len(inversion_result.depths) > 0
                        else "N/A"
                    ),
                    "Maximum depth explored",
                ],
            ]
        )

        # ML results (if available)
        if ml_result:
            prob = ml_result.get("aquifer_probability", [0])[0]
            uncertainty = ml_result.get("total_uncertainty", [0])[0]
            summary_data.extend(
                [
                    ["Aquifer Probability", f"{prob:.1%}", "ML prediction"],
                    [
                        "Prediction Uncertainty",
                        f"±{uncertainty:.1%}",
                        "Confidence level",
                    ],
                ]
            )

        # Site information (if available)
        if site_info:
            summary_data.extend(
                [
                    [
                        "Site",
                        site_info.get("site_name", "Unknown"),
                        "Site identification",
                    ],
                    [
                        "Survey Date",
                        site_info.get("survey_date", "Unknown"),
                        "Data collection date",
                    ],
                ]
            )

        # Create table
        table = ax.table(
            cellText=summary_data[1:],
            colLabels=summary_data[0],
            cellLoc="left",
            loc="center",
            colWidths=[0.3, 0.3, 0.4],
        )

        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)

        # Style the table
        for i in range(len(summary_data[0])):
            table[(0, i)].set_facecolor("#4472C4")
            table[(0, i)].set_text_props(weight="bold", color="white")

        ax.set_title("Analysis Summary", fontsize=14, fontweight="bold", pad=20)