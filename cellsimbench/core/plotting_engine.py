"""
Plotting engine for CellSimBench framework.

Generates comprehensive visualizations for benchmark results including
metric comparisons, variance analysis, and DEG quantile impact plots.
"""

import logging
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any, Optional, TYPE_CHECKING
import scanpy as sc
from omegaconf import DictConfig
from scipy import stats

# Suppress all warnings for cleaner output
warnings.filterwarnings('ignore')

# Suppress all warnings for cleaner output
warnings.filterwarnings('ignore')

from .variance_analyzer import VarianceAnalyzer
from .data_manager import DataManager

if TYPE_CHECKING:
    from .deg_quantile_analyzer import DEGQuantileAnalyzer

log = logging.getLogger(__name__)


class PlottingEngine:
    """Generate comprehensive plots for benchmark results.
    
    Creates various visualization types including violin plots for metrics,
    variance rank plots for mode collapse detection, and DEG quantile
    impact plots for performance stratification.
    
    Attributes:
        data_manager: DataManager instance for accessing data.
        results_dict: Dictionary containing benchmark results.
        all_predictions: Dictionary mapping model names to predictions.
        split_name: Name of the split being evaluated.
        output_dir: Directory for saving plots.
        config: Configuration object.
        plots_dir: Directory for plot outputs.
        variance_analyzer: VarianceAnalyzer instance.
        deg_quantile_analyzer: DEGQuantileAnalyzer instance.
        
    Example:
        >>> engine = PlottingEngine(data_manager, results, predictions, 'split', output_dir, config)
        >>> engine.generate_all_plots()
    """
    
    def __init__(self, data_manager: DataManager, results_dict: Dict[str, Any], 
                 all_predictions: Dict[str, sc.AnnData], split_name: str, 
                 output_dir: Path, config: DictConfig) -> None:
        """Initialize PlottingEngine with benchmark results.
        
        Args:
            data_manager: DataManager for accessing ground truth and metadata.
            results_dict: Dictionary containing all benchmark results.
            all_predictions: Dictionary mapping model names to prediction AnnData.
            split_name: Name of the split being evaluated.
            output_dir: Directory where plots will be saved.
            config: Configuration object with plotting settings.
        """
        self.data_manager = data_manager
        self.results_dict = results_dict
        self.all_predictions = all_predictions
        self.split_name = split_name
        self.output_dir = Path(output_dir)
        self.config = config
        self.dpi = int(getattr(self.config.output.plotting, "dpi", 300))
        
        # Create plots directory
        self.plots_dir = self.output_dir / "plots"
        self.plots_dir.mkdir(exist_ok=True)
        
        # Initialize helper analyzers
        self.variance_analyzer = VarianceAnalyzer(data_manager)
        self.deg_quantile_analyzer: Optional["DEGQuantileAnalyzer"] = None
        if self.config.output.plotting.deg_quantile_impact:
            from .deg_quantile_analyzer import DEGQuantileAnalyzer
            
            self.deg_quantile_analyzer = DEGQuantileAnalyzer(data_manager)
        
        # Set up plotting style
        plt.style.use('default')
        sns.set_palette("Set2")
        self.model_display_names = {
            "epistasis_specific": "Epistasis (Specific)",
            "epistasis_shared": "Epistasis (Agnostic)",
            "epistasis_hybrid": "Epistasis (Hybrid)",
            "linear_regression": "Linear Regression",
            "linear_regression_wmse": "Linear Regression (wMSE)",
            "onehot_linear": "Linear Regression",
            "onehot_linear_wmse": "Linear Regression (wMSE)",
            "random_perturbation": "Random Perturbation",
        }

    def _format_model_name(self, model_name: str) -> str:
        """Return a display-friendly model name."""
        return self.model_display_names.get(model_name, model_name)
        
    def generate_all_plots(self) -> None:
        """Generate all configured plots.
        
        Creates plots based on configuration settings including condition
        comparisons, variance ranks, and DEG quantile impacts.
        """
        log.info("Generating plots...")
        
        # Generate modern strip plots from detailed metrics
        self.plot_metrics_stripplot()
        self.plot_lse_diagnostics()
        
        plotting_config = self.config.output.plotting
        
        if plotting_config.variance_rank:
            self.plot_variance_rank()
            
        if plotting_config.deg_quantile_impact:
            self.plot_deg_quantile_impact()
            
        log.info(f"All plots saved to {self.plots_dir}")

    def plot_lse_diagnostics(self) -> None:
        """Plot LSE diagnostics if present in model predictions."""
        for model_name, adata in self.all_predictions.items():
            if "lse_diagnostics" not in adata.uns:
                continue
            diag = adata.uns["lse_diagnostics"]
            variant = diag.get("variant", "unknown")
            epsilon = np.asarray(diag.get("epsilon", np.array([])), dtype=float)
            eps_norm = float(diag.get("epsilon_norm", np.nan))

            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            if epsilon.size > 0:
                sns.histplot(epsilon, ax=axes[0], kde=True, bins=50)
                axes[0].set_title("Epsilon entries")
                axes[0].set_xlabel("epsilon")
            else:
                axes[0].set_axis_off()
                axes[0].text(0.5, 0.5, "No epsilon data", ha="center", va="center")

            if variant == "shared":
                c_val = diag.get("c", np.nan)
                axes[1].axis("off")
                axes[1].text(
                    0.05,
                    0.6,
                    f"c = {c_val:.4f}\n||eps|| = {eps_norm:.4f}",
                    fontsize=12,
                )
            else:
                c_sum_abs = np.asarray(diag.get("c_sum_abs", np.array([])), dtype=float)
                if c_sum_abs.size > 0:
                    sns.histplot(c_sum_abs, ax=axes[1], kde=True, bins=50)
                    axes[1].set_title("|c_p + c_q| distribution")
                    axes[1].set_xlabel("|c_p + c_q|")
                else:
                    axes[1].set_axis_off()
                    axes[1].text(0.5, 0.5, "No |c_p + c_q| data", ha="center", va="center")
                if variant == "hybrid":
                    c_shared = float(diag.get("c_shared", np.nan))
                    fig.suptitle(f"c_shared = {c_shared:.4f} | ||eps|| = {eps_norm:.4f}")
                else:
                    fig.suptitle(f"||eps|| = {eps_norm:.4f}")

            output_path = self.plots_dir / f"lse_{model_name}_c_eps.png"
            fig.tight_layout()
            fig.savefig(output_path, dpi=self.dpi)
            plt.close(fig)
    
    def plot_metrics_stripplot(self) -> None:
        """Create professional strip plots for all metrics.
        
        Generates publication-quality strip plots with:
        - Dark grey points for individual measurements with jittering
        - Dark red horizontal lines for means
        - Different colors for baselines
        - Models sorted by mean performance
        """
        log.info("Creating professional strip plots...")
        
        # Load detailed metrics from CSV
        detailed_csv_path = self.output_dir / 'detailed_metrics.csv'
        if not detailed_csv_path.exists():
            log.warning("detailed_metrics.csv not found, skipping strip plots")
            return
            
        df = pd.read_csv(detailed_csv_path)
        
        # Get unique metrics
        metrics = df['metric'].unique()
        metrics_to_plot = getattr(self.config.output.plotting, "metrics_to_plot", None)
        if metrics_to_plot:
            metrics = [metric for metric in metrics if metric in set(metrics_to_plot)]
        
        # Define baseline models (these will be colored differently)
        baseline_keywords = [
            'control_mean',
            'dataset_mean',
            'technical_duplicate',
            'additive',
            'linear',
            'interpolated_duplicate',
            'epistasis',
            'onehot',
            'target_scaling',
            'random_perturbation',
        ]
        
        # Set up professional plot style with larger fonts
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.rcParams['font.family'] = 'DejaVu Sans'
        plt.rcParams['font.size'] = 14
        plt.rcParams['axes.labelsize'] = 16
        plt.rcParams['axes.titlesize'] = 18
        plt.rcParams['xtick.labelsize'] = 13
        plt.rcParams['ytick.labelsize'] = 13
        plt.rcParams['legend.fontsize'] = 13
        plt.rcParams['figure.titlesize'] = 20
        
        # Create plot for each metric
        for metric in metrics:
            self._create_stripplot(df, metric, baseline_keywords)
    
    def _create_stripplot(self, df: pd.DataFrame, metric: str, baseline_keywords: List[str]) -> None:
        """Create a single strip plot for a metric.
        
        Args:
            df: DataFrame with detailed metrics
            metric: Name of the metric to plot
            baseline_keywords: Keywords to identify baseline models
        """
        # Filter data for this metric AND exclude ground_truth (it's the reference, not a baseline to compare)
        metric_df = df[(df['metric'] == metric) & (df['model'] != 'ground_truth')].copy()
        
        # Rename control to control_mean for display
        metric_df['model'] = metric_df['model'].replace('control', 'control_mean')
        metric_df['model_raw'] = metric_df['model']
        metric_df['model_display'] = metric_df['model_raw'].map(self._format_model_name)
        
        if metric_df.empty:
            return
            
        # Order models by median, while retaining means for reference lines.
        model_order_values = metric_df.groupby('model_raw')['value'].median()
        model_means = metric_df.groupby('model_raw')['value'].mean()
        use_calibrated_focus_scale = self._use_calibrated_focus_scale(metric)
        use_r2_negative_focus_scale = self._use_r2_negative_focus_scale(metric)
        
        # Determine sort order based on metric type.
        # All calibrated metrics are lower-is-better by construction.
        if self._is_lower_better_metric(metric):
            ordered_models = model_order_values.sort_values(ascending=True).index.tolist()
        else:
            ordered_models = model_order_values.sort_values(ascending=False).index.tolist()
        
        # Reorder models by median performance (best to worst).
        display_order = [self._format_model_name(name) for name in ordered_models]
        metric_df['model_display'] = pd.Categorical(
            metric_df['model_display'],
            categories=display_order,
            ordered=True,
        )
        
        # Identify baseline models
        def is_baseline(model_name):
            return any(keyword in model_name.lower() for keyword in baseline_keywords)
        
        # Calculate figure size based on number of models
        n_models = len(ordered_models)
        fig_width = max(10, min(20, n_models * 1.2))
        fig_height = 7
        
        # Create figure with white background
        fig, ax = plt.subplots(figsize=(fig_width, fig_height), facecolor='white')
        ax.set_facecolor('white')
        
        # Color scheme
        baseline_color = '#2E86AB'  # Deep blue for baselines
        model_color = '#424242'  # Dark grey for regular models
        mean_color = '#A23B72'  # Dark red/burgundy for mean lines
        
        # Create color palette for swarmplot
        palette = []
        for model_name in ordered_models:
            if is_baseline(model_name):
                palette.append(baseline_color)
            else:
                palette.append(model_color)
        
        # Use stripplot for fast, clean visualization with high-density data
        use_focus_scale = use_calibrated_focus_scale or use_r2_negative_focus_scale
        plot_value_col = 'value_plot' if use_focus_scale else 'value'
        if use_calibrated_focus_scale:
            self._set_calibrated_focus_params(metric_df['value'].to_numpy(dtype=float))
            metric_df['value_plot'] = self._calibrated_focus_transform(
                metric_df['value'].to_numpy(dtype=float)
            )
        elif use_r2_negative_focus_scale:
            self._set_r2_negative_focus_params(metric_df['value'].to_numpy(dtype=float))
            metric_df['value_plot'] = self._r2_negative_focus_transform(
                metric_df['value'].to_numpy(dtype=float)
            )
        sns.stripplot(data=metric_df, x='model_display', y=plot_value_col, 
                     palette=palette,
                     size=3,  # Smaller points for high density
                     alpha=0.4,  # More transparent for overlapping points
                     jitter=True,  # Built-in jittering
                     ax=ax)
        
        # Add mean lines and optional text
        show_median = bool(getattr(self.config.output.plotting, "stripplot_show_median", False))
        median_color = '#2F855A'  # Green tone for median reference lines
        for i, model_name in enumerate(ordered_models):
            model_mean = float(model_means.loc[model_name])
            mean_y = (
                self._calibrated_focus_transform(np.array([model_mean], dtype=float))[0]
                if use_calibrated_focus_scale
                else self._r2_negative_focus_transform(np.array([model_mean], dtype=float))[0]
                if use_r2_negative_focus_scale
                else model_mean
            )
            # Add mean line
            ax.hlines(mean_y, i - 0.35, i + 0.35, 
                     colors=mean_color, 
                     linewidth=2.5,
                     zorder=3)

            if show_median:
                model_values = metric_df.loc[metric_df['model_raw'] == model_name, 'value'].to_numpy(dtype=float)
                model_median = float(np.nanmedian(model_values)) if model_values.size > 0 else np.nan
                if np.isfinite(model_median):
                    median_y = (
                        self._calibrated_focus_transform(np.array([model_median], dtype=float))[0]
                        if use_calibrated_focus_scale
                        else self._r2_negative_focus_transform(np.array([model_median], dtype=float))[0]
                        if use_r2_negative_focus_scale
                        else model_median
                    )
                    ax.hlines(
                        median_y,
                        i - 0.35,
                        i + 0.35,
                        colors=median_color,
                        linewidth=2.0,
                        linestyles=':',
                        zorder=3,
                    )
            
            # Add mean value text only if not too many models
            if n_models <= 8:  # Reduced threshold for cleaner plots
                ax.text(i, mean_y, f'{model_mean:.3f}', 
                       horizontalalignment='center',
                       verticalalignment='bottom',
                       fontsize=12,
                       color=mean_color,
                       fontweight='bold')
        
        # Customize axes
        ax.set_xticks(range(len(model_means)))
        ax.set_xticklabels(display_order, rotation=45, ha='right')
        ax.set_xlabel('Model', fontweight='bold')
        ax.set_ylabel(self._format_metric_name(metric), fontweight='bold')
        ax.set_title(f'{self._format_metric_name(metric)} Comparison', 
                    fontweight='bold', pad=25)
        if use_calibrated_focus_scale:
            self._apply_calibrated_focus_ticks(ax, metric_df['value'].to_numpy(dtype=float))
        elif use_r2_negative_focus_scale:
            self._apply_r2_negative_focus_ticks(ax, metric_df['value'].to_numpy(dtype=float))
        
        # Grid styling
        ax.grid(True, axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
        ax.set_axisbelow(True)
        
        # Remove top and right spines for cleaner look
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(0.5)
        ax.spines['bottom'].set_linewidth(0.5)
        
        # Add reference line at 0 for certain metrics
        if any(keyword in metric.lower() for keyword in ['pearson', 'r2', 'delta']):
            zero_y = (
                self._calibrated_focus_transform(np.array([0.0], dtype=float))[0]
                if use_calibrated_focus_scale
                else self._r2_negative_focus_transform(np.array([0.0], dtype=float))[0]
                if use_r2_negative_focus_scale
                else 0.0
            )
            ax.axhline(y=zero_y, color='black', linestyle=':', linewidth=1, alpha=0.5, zorder=1)
        
        # Add legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor=model_color, 
                  markersize=10, alpha=0.6, label='Models'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor=baseline_color, 
                  markersize=10, alpha=0.6, label='Baselines'),
            Line2D([0], [0], color=mean_color, linewidth=3, label='Mean')
        ]
        if show_median:
            legend_elements.append(
                Line2D([0], [0], color=median_color, linewidth=2.5, linestyle=':', label='Median')
            )
        ax.legend(handles=legend_elements, loc='best', frameon=True, 
                 fancybox=True, shadow=False, framealpha=0.9)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save figure with high DPI for publication quality
        filename = f"stripplot_{metric}.png"
        fig.savefig(self.plots_dir / filename, 
                   dpi=300,  # High DPI for publication quality
                   bbox_inches='tight',
                   facecolor='white',
                   edgecolor='none')
        plt.close(fig)
        
        log.info(f"Saved {filename}")

    def _use_calibrated_focus_scale(self, metric: str) -> bool:
        """Return True when metric is calibrated to [0, 1]."""
        metric_lower = metric.lower()
        return metric_lower.startswith('c') or metric_lower.startswith('fc')

    def _is_lower_better_metric(self, metric: str) -> bool:
        """Return True when lower values indicate better performance."""
        metric_lower = metric.lower()
        if self._use_calibrated_focus_scale(metric_lower):
            return True
        return metric_lower in {'mse', 'wmse', 'fmse'}

    def _use_r2_negative_focus_scale(self, metric: str) -> bool:
        """Return True for uncalibrated R2-family metrics."""
        metric_lower = metric.lower()
        return ("r2" in metric_lower) and not (
            metric_lower.startswith('c') or metric_lower.startswith('fc')
        )

    def _set_calibrated_focus_params(
        self, raw_values: np.ndarray, target_compressed_fraction: float = 0.3
    ) -> None:
        """Set dynamic high-value compression so >1 takes ~30% of y-axis."""
        raw_values = np.asarray(raw_values, dtype=float)
        finite = raw_values[np.isfinite(raw_values)]
        if finite.size == 0:
            self._calibrated_focus_high_alpha = 1.0
            return
        y_max = float(np.nanmax(finite))
        if y_max <= 1.0:
            self._calibrated_focus_high_alpha = 1.0
            return
        # Keep [0,1] dominant; solve alpha from desired compressed fraction.
        target_ratio = target_compressed_fraction / max(1e-6, 1.0 - target_compressed_fraction)
        high_span_raw = max(1e-6, y_max - 1.0)
        alpha = target_ratio * 1.0 / high_span_raw
        self._calibrated_focus_high_alpha = float(np.clip(alpha, 0.01, 1.0))

    def _calibrated_focus_transform(self, values: np.ndarray) -> np.ndarray:
        """Piecewise transform: linear in [0,1], compressed above 1."""
        values = np.asarray(values, dtype=float)
        transformed = np.empty_like(values, dtype=float)

        mask_low = values < 0
        mask_mid = (values >= 0) & (values <= 1)
        mask_high = values > 1

        # Keep negatives linear (calibrated metrics should rarely be <0).
        transformed[mask_low] = values[mask_low]
        transformed[mask_mid] = values[mask_mid]
        high_alpha = getattr(self, "_calibrated_focus_high_alpha", 1.0)
        transformed[mask_high] = 1.0 + high_alpha * (values[mask_high] - 1.0)

        return transformed

    def _apply_calibrated_focus_ticks(self, ax: plt.Axes, raw_values: np.ndarray) -> None:
        """Set readable ticks for calibrated focus scale in original units."""
        raw_values = np.asarray(raw_values, dtype=float)
        finite_raw = raw_values[np.isfinite(raw_values)]
        if finite_raw.size == 0:
            return

        tick_candidates = np.array(
            [0.0, 0.25, 0.5, 0.75, 1.0, 1.1, 1.25, 1.5, 2.0, 3.0, 5.0, 10.0, 20.0, 50.0, 100.0],
            dtype=float,
        )
        transformed_candidates = self._calibrated_focus_transform(tick_candidates)

        # Hard floor at 0 in raw-value space, per calibrated metric convention.
        y_min = float(self._calibrated_focus_transform(np.array([0.0], dtype=float))[0])
        y_max = float(np.nanmax(self._calibrated_focus_transform(finite_raw)))
        pad = 0.05 * (y_max - y_min if y_max > y_min else 1.0)
        ax.set_ylim(y_min, y_max + pad)

        tick_mask = (transformed_candidates >= y_min) & (transformed_candidates <= (y_max + pad))
        tick_positions = transformed_candidates[tick_mask]
        tick_values = tick_candidates[tick_mask]
        if tick_positions.size == 0:
            return

        # Limit tick count to avoid overlapping y-labels.
        max_ticks = 8
        if tick_positions.size > max_ticks:
            must_keep = np.isin(tick_values, np.array([0.0, 0.5, 1.0], dtype=float))
            keep_idx = np.where(must_keep)[0].tolist()
            remaining_idx = [idx for idx in range(tick_positions.size) if idx not in keep_idx]
            slots_left = max_ticks - len(keep_idx)
            if slots_left > 0 and remaining_idx:
                sampled = np.linspace(0, len(remaining_idx) - 1, slots_left, dtype=int)
                keep_idx.extend(remaining_idx[i] for i in sampled)
            keep_idx = sorted(set(keep_idx))
            tick_positions = tick_positions[keep_idx]
            tick_values = tick_values[keep_idx]

        # Enforce minimum separation in transformed space to avoid overlapping labels
        # around transition points (e.g. 1.0 vs 1.1).
        min_gap = 0.08
        selected_idx: List[int] = []
        for idx, pos in enumerate(tick_positions):
            if not selected_idx:
                selected_idx.append(idx)
                continue
            if (pos - tick_positions[selected_idx[-1]]) >= min_gap:
                selected_idx.append(idx)
        if selected_idx[-1] != (tick_positions.size - 1):
            selected_idx.append(tick_positions.size - 1)
        tick_positions = tick_positions[selected_idx]
        tick_values = tick_values[selected_idx]

        ax.set_yticks(tick_positions)
        labels = []
        for value in tick_values:
            if value in {0.0, 1.0}:
                labels.append(f"{int(value)}")
            elif abs(value) < 1:
                labels.append(f"{value:.2f}".rstrip('0').rstrip('.'))
            elif abs(value) < 10:
                labels.append(f"{value:.1f}".rstrip('0').rstrip('.'))
            else:
                labels.append(f"{value:.0f}")
        ax.set_yticklabels(labels)

    def _set_r2_negative_focus_params(
        self, raw_values: np.ndarray, max_negative_fraction: float = 0.25
    ) -> None:
        """Set compression for <0 region so it takes <= 25% of y-axis."""
        raw_values = np.asarray(raw_values, dtype=float)
        finite = raw_values[np.isfinite(raw_values)]
        if finite.size == 0:
            self._r2_negative_focus_alpha = 1.0
            return
        y_min = float(np.nanmin(finite))
        y_max = float(np.nanmax(finite))
        if y_min >= 0.0:
            self._r2_negative_focus_alpha = 1.0
            return
        pos_span = max(1e-6, y_max - 0.0)
        neg_span = max(1e-6, 0.0 - y_min)
        target_ratio = max_negative_fraction / max(1e-6, 1.0 - max_negative_fraction)
        alpha = target_ratio * pos_span / neg_span
        self._r2_negative_focus_alpha = float(np.clip(alpha, 0.01, 1.0))

    def _r2_negative_focus_transform(self, values: np.ndarray) -> np.ndarray:
        """Piecewise transform: linear for y>=0, compressed for y<0."""
        values = np.asarray(values, dtype=float)
        transformed = values.copy()
        alpha = getattr(self, "_r2_negative_focus_alpha", 1.0)
        mask_neg = values < 0.0
        transformed[mask_neg] = alpha * values[mask_neg]
        return transformed

    def _apply_r2_negative_focus_ticks(self, ax: plt.Axes, raw_values: np.ndarray) -> None:
        """Set readable ticks for compressed uncalibrated R2 scale."""
        raw_values = np.asarray(raw_values, dtype=float)
        finite_raw = raw_values[np.isfinite(raw_values)]
        if finite_raw.size == 0:
            return
        transformed = self._r2_negative_focus_transform(finite_raw)
        y_min = float(np.nanmin(transformed))
        y_max = float(np.nanmax(transformed))
        pad = 0.05 * (y_max - y_min if y_max > y_min else 1.0)
        ax.set_ylim(y_min, y_max + pad)

        tick_candidates = np.array([-4.0, -3.0, -2.0, -1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0], dtype=float)
        tick_positions = self._r2_negative_focus_transform(tick_candidates)
        tick_mask = (tick_positions >= y_min) & (tick_positions <= (y_max + pad))
        tick_positions = tick_positions[tick_mask]
        tick_values = tick_candidates[tick_mask]
        if tick_positions.size == 0:
            return
        ax.set_yticks(tick_positions)
        labels = []
        for value in tick_values:
            if value in {0.0, 1.0, -1.0, -2.0, -3.0, -4.0}:
                labels.append(f"{int(value)}")
            else:
                labels.append(f"{value:.2f}".rstrip('0').rstrip('.'))
        ax.set_yticklabels(labels)
    
    def _format_metric_name(self, metric: str) -> str:
        """Format metric name for display.
        
        Args:
            metric: Raw metric name
            
        Returns:
            Formatted metric name for display
        """
        # Define nice names for metrics
        metric_names = {
            'mse': 'MSE',
            'wmse': 'Weighted MSE',
            'cmse': 'Calibrated MSE',
            'cwmse': 'Calibrated Weighted MSE',
            'fcmse': 'Filtered Calibrated MSE',
            'fmse': 'Filtered MSE',
            'fpearson': 'Filtered Pearson',
            'fcpearson': 'Filtered Calibrated Pearson',
            'cpearson': 'Calibrated Pearson',
            'fr2': 'Filtered R²',
            'fcr2': 'Filtered Calibrated R²',
            'cr2': 'Calibrated R²',
            'pearson_deltactrl': 'Pearson Δ Control',
            'pearson_deltactrl_degs': 'Pearson Δ Control (DEGs)',
            'cpearson_deltactrl': 'Calibrated Pearson Δ Control',
            'weighted_pearson_deltactrl': 'Weighted Pearson Δ Control',
            'cweighted_pearson_deltactrl': 'Calibrated Weighted Pearson Δ Control',
            'fpearson_deltactrl': 'Filtered Pearson Δ Control',
            'fcpearson_deltactrl': 'Filtered Calibrated Pearson Δ Control',
            'pearson_deltapert': 'Pearson Δ Perturbation',
            'pearson_deltapert_degs': 'Pearson Δ Perturbation (DEGs)',
            'cpearson_deltapert': 'Calibrated Pearson Δ Perturbation',
            'weighted_pearson_deltapert': 'Weighted Pearson Δ Perturbation',
            'cweighted_pearson_deltapert': 'Calibrated Weighted Pearson Δ Perturbation',
            'fpearson_deltapert': 'Filtered Pearson Δ Perturbation',
            'fcpearson_deltapert': 'Filtered Calibrated Pearson Δ Perturbation',
            'r2_deltactrl': 'R² Δ Control',
            'r2_deltactrl_degs': 'R² Δ Control (DEGs)',
            'cr2_deltactrl': 'Calibrated R² Δ Control',
            'fr2_deltactrl': 'Filtered R² Δ Control',
            'fcr2_deltactrl': 'Filtered Calibrated R² Δ Control',
            'r2_deltapert': 'R² Δ Perturbation',
            'r2_deltapert_degs': 'R² Δ Perturbation (DEGs)',
            'cr2_deltapert': 'Calibrated R² Δ Perturbation',
            'fr2_deltapert': 'Filtered R² Δ Perturbation',
            'fcr2_deltapert': 'Filtered Calibrated R² Δ Perturbation',
            'weighted_r2_deltactrl': 'Weighted R² Δ Control',
            'cweighted_r2_deltactrl': 'Calibrated Weighted R² Δ Control',
            'weighted_r2_deltapert': 'Weighted R² Δ Perturbation',
            'cweighted_r2_deltapert': 'Calibrated Weighted R² Δ Perturbation',
            'pds': 'PDS',
            'pds_wmse': 'PDS (wMSE)',
            'pds_pearson_deltapert': 'PDS (Pearson Δ Perturbation)',
            'pds_weighted_pearson_deltapert': 'PDS (Weighted Pearson Δ Perturbation)',
            'pds_r2_deltapert': 'PDS (R² Δ Perturbation)',
            'pds_weighted_r2_deltapert': 'PDS (Weighted R² Δ Perturbation)',
            'cpds': 'Calibrated PDS',
            'cpds_wmse': 'Calibrated PDS (wMSE)',
            'cpds_pearson_deltapert': 'Calibrated PDS (Pearson Δ Perturbation)',
            'cpds_weighted_pearson_deltapert': 'Calibrated PDS (Weighted Pearson Δ Perturbation)',
            'cpds_r2_deltapert': 'Calibrated PDS (R² Δ Perturbation)',
            'cpds_weighted_r2_deltapert': 'Calibrated PDS (Weighted R² Δ Perturbation)',
        }
        
        return metric_names.get(metric, metric.replace('_', ' ').title())
    

    def plot_variance_rank(self) -> None:
        """Create variance rank plots for each covariate separately.
        
        Generates plots showing per-gene variance across perturbations,
        useful for detecting mode collapse in model predictions.
        """
        log.info("Creating variance rank plots...")
        
        # Get available covariates
        available_covariates = self._get_available_covariates()
        
        for covariate in available_covariates:
            self._plot_covariate_variance_rank(covariate)
    
    def _plot_covariate_variance_rank(self, covariate: str) -> None:
        """Create variance rank plot for a specific covariate.
        
        Args:
            covariate: Covariate value to analyze.
        """
        # Calculate variances for this covariate
        variance_data = self.variance_analyzer.calculate_covariate_variances(
            self.all_predictions, covariate, self.split_name
        )
       
        if not variance_data:
            log.warning(f"No variance data for covariate {covariate}")
            return
        
        # Filter out mean baselines
        filtered_variance_data = {k: v for k, v in variance_data.items() 
                                if 'mean_baseline' not in k.lower() and 'ctrl_baselines' not in k.lower()}
        
        if not filtered_variance_data:
            log.warning(f"No non-mean-baseline data for covariate {covariate}")
            return
        
        # Create rank plot
        top_k = self.config.output.plotting.variance_analysis.top_k_genes
        
        plt.figure(figsize=(12, 6))
        
        # Get ground truth variances for sorting
        ground_truth_key = None
        for key in filtered_variance_data.keys():
            if 'technical_duplicate' in key.lower() or 'ground_truth' in key.lower():
                ground_truth_key = key
                break
        
        if ground_truth_key is None:
            # Use first available as reference
            ground_truth_key = list(filtered_variance_data.keys())[0]
            
        ground_truth_var = filtered_variance_data[ground_truth_key]
        sorted_indices = np.argsort(ground_truth_var)[::-1][:top_k]
        ranks = np.arange(1, len(sorted_indices) + 1)
        
        # Plot each model with higher alpha
        colors = plt.cm.Set2(np.linspace(0, 1, len(filtered_variance_data)))
        
        for i, (model_name, variances) in enumerate(filtered_variance_data.items()):
            sorted_variances = variances[sorted_indices]
            
            if model_name == ground_truth_key:
                color = 'black'
                alpha = 0.9
                size = 4
                zorder = 10
            else:
                color = colors[i]
                alpha = 0.8  # Increased from 0.7
                size = 3
                zorder = 5
                
            plt.scatter(ranks, sorted_variances, label=self._format_model_name(model_name), 
                       s=size, alpha=alpha, zorder=zorder, color=color)
            
            # Add moving average as dashed line (slightly darker)
            if len(sorted_variances) >= 50:  # Only if we have enough points
                # Calculate moving average with window size proportional to data size
                window_size = max(10, len(sorted_variances) // 50)
                moving_avg = np.convolve(sorted_variances, np.ones(window_size)/window_size, mode='valid')
                moving_avg_ranks = ranks[:len(moving_avg)]
                
                # Make color darker
                if model_name == ground_truth_key:
                    line_color = 'black'
                else:
                    # Make color darker by reducing brightness
                    line_color = tuple([c * 0.6 for c in color[:3]] + [1.0])
                
                plt.plot(moving_avg_ranks, moving_avg, color=line_color, 
                        linestyle='--', linewidth=2, alpha=0.8, zorder=zorder+1)
        
        plt.xlabel(f'Gene Rank (by {self._format_model_name(ground_truth_key)} Variance)', fontsize=12)
        plt.ylabel('Gene-wise Variance', fontsize=12)
        plt.title(f'Variance Rank Plot - {covariate} (Top {top_k} genes)', fontsize=14, fontweight='bold')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        
        plt.tight_layout()
        
        # Save plot
        filename = f"variance_rank_{covariate}.png"
        plt.savefig(self.plots_dir / filename, dpi=self.config.output.plotting.dpi, 
                   bbox_inches='tight')
        plt.close()
        
        log.info(f"Saved {filename}")
    
    def plot_deg_quantile_impact(self) -> None:
        """Create DEG quantile impact plots for each covariate separately.
        
        Generates plots showing model performance stratified by perturbation
        strength as measured by DEG count.
        """
        log.info("Creating DEG quantile impact plots...")
        
        available_covariates = self._get_available_covariates()
        
        for covariate in available_covariates:
            self._plot_covariate_deg_quantiles(covariate)
    
    def _plot_covariate_deg_quantiles(self, covariate: str) -> None:
        """Create DEG quantile plots for a specific covariate.
        
        Args:
            covariate: Covariate value to analyze.
        """
        # Get perturbations for this covariate
        covariate_perturbations: List[str] = []
        
        if self.split_name == 'aggregated_folds':
            # For aggregated results, extract perturbations from predictions directly
            sample_predictions = next(iter(self.all_predictions.values()))
            obs_df = sample_predictions.obs
            # Get unique perturbations for this specific covariate
            for _, row in obs_df.iterrows():
                if row['covariate'] == covariate:
                    if row['condition'] not in covariate_perturbations:
                        covariate_perturbations.append(row['condition'])
        else:
            # Use standard DataManager method for regular splits
            cov_pert_pairs = self.data_manager.get_covariate_condition_pairs(self.split_name, 'test')
            for cov, pert in cov_pert_pairs:
                if cov == covariate:
                    covariate_perturbations.append(pert)
        
        if not covariate_perturbations:
            log.warning(f"No perturbations found for covariate {covariate}")
            return
        
        # Assign to quantiles
        n_quantiles = self.config.output.plotting.deg_quantiles.n_quantiles
        quantile_assignments = self.deg_quantile_analyzer.assign_perturbations_to_quantiles(
            covariate_perturbations, covariate, n_quantiles
        )
        
        # Prepare quantile data
        quantile_data = []
        
        for model_name, model_results in self.results_dict['models'].items():
            for metric_name, cov_pert_scores in model_results['metrics'].items():
                for cov_pert_key, score in cov_pert_scores.items():
                    if cov_pert_key.startswith(f"{covariate}_"):
                        perturbation = cov_pert_key.split('_', 1)[1]
                        if perturbation in quantile_assignments:
                            quantile_idx = quantile_assignments[perturbation]
                            quantile_label = f"Q{quantile_idx+1}"
                            
                            quantile_data.append({
                                'Model': model_name,
                                'Metric': metric_name,
                                'Perturbation': perturbation,
                                'Quantile': quantile_label,
                                'QuantileIdx': quantile_idx,
                                'Score': score
                            })
        
        if not quantile_data:
            log.warning(f"No quantile data for covariate {covariate}")
            return
            
        df_quantile = pd.DataFrame(quantile_data)
        
        # Create plots for all configured metrics
        for metric in self.config.output.plotting.metrics_to_plot:
            if metric in df_quantile['Metric'].unique():
                self._create_deg_quantile_plot(df_quantile, metric, covariate)
                # self._create_deg_quantile_ridge_plot(df_quantile, metric, covariate)
    
    def _create_deg_quantile_plot(self, df_quantile: pd.DataFrame, metric: str, covariate: str) -> None:
        """Create a DEG quantile impact plot for a specific metric.
        
        Args:
            df_quantile: DataFrame containing quantile data.
            metric: Metric name to plot.
            covariate: Covariate value for plot title.
        """
        df_metric = df_quantile[(df_quantile['Metric'] == metric) & (df_quantile['Model'] != 'ground_truth')].copy()
        
        # Rename control to control_mean for display
        df_metric['Model'] = df_metric['Model'].replace('control', 'control_mean')
        df_metric['ModelRaw'] = df_metric['Model']
        df_metric['Model'] = df_metric['ModelRaw'].map(self._format_model_name)
        
        if df_metric.empty:
            return
        
        # Sort by quantile index to ensure proper ordering on x-axis
        df_metric = df_metric.sort_values('QuantileIdx')
        
        # Set categorical order for proper x-axis ordering
        quantile_order = sorted(df_metric['Quantile'].unique(), key=lambda x: int(x[1:]))
        df_metric['Quantile'] = pd.Categorical(df_metric['Quantile'], categories=quantile_order, ordered=True)
        
        # Create figure with white background
        fig, ax = plt.subplots(figsize=(14, 7), facecolor='white')
        ax.set_facecolor('white')
        
        # Color scheme matching strip plots
        mean_color = '#A23B72'  # Dark red/burgundy for mean lines
        
        # Identify baseline models
        baseline_keywords = [
            'control_mean',
            'dataset_mean',
            'technical_duplicate',
            'additive',
            'linear',
            'epistasis',
            'onehot',
            'target_scaling',
            'random_perturbation',
        ]
        def is_baseline(model_name):
            return any(keyword in model_name.lower() for keyword in baseline_keywords)
        
        # Create unique color palette for each model
        unique_models = df_metric['Model'].unique()
        n_models = len(unique_models)
        
        # Use categorical colors that are easily distinguishable
        # Define specific colors for baselines and models
        baseline_colors_categorical = ['#1f77b4', '#17becf', '#9467bd', '#2ca02c']  # Blues, cyan, purple, green
        model_colors_categorical = ['#d62728', '#ff7f0e', '#e377c2', '#8c564b', '#bcbd22', '#7f7f7f']  # Red, orange, pink, brown, olive, grey
        
        palette = {}
        baseline_idx = 0
        model_idx = 0
        for model_name in unique_models:
            raw_name = df_metric[df_metric['Model'] == model_name]['ModelRaw'].iloc[0]
            if is_baseline(raw_name):
                # Use categorical baseline colors, cycling if needed
                palette[model_name] = baseline_colors_categorical[baseline_idx % len(baseline_colors_categorical)]
                baseline_idx += 1
            else:
                # Use categorical model colors, cycling if needed
                palette[model_name] = model_colors_categorical[model_idx % len(model_colors_categorical)]
                model_idx += 1
        
        # Manually create swarmplot with proper separation for each model
        model_width = 0.6 / n_models  # Narrower width allocated to each model within a quantile
        
        # Plot each model separately
        for model_idx, model in enumerate(unique_models):
            model_data = df_metric[df_metric['Model'] == model]
            color = palette[model]
            
            # Calculate x positions for this model
            x_positions = []
            y_positions = []
            
            for q_idx, quantile in enumerate(quantile_order):
                quantile_data = model_data[model_data['Quantile'] == quantile]['Score'].values
                
                if len(quantile_data) > 0:
                    # Base x position for this quantile
                    base_x = q_idx
                    # Offset for this model within the quantile
                    model_offset = (model_idx - (n_models - 1) / 2) * model_width
                    
                    # Create swarm positions for this model in this quantile
                    for y_val in quantile_data:
                        x_pos = base_x + model_offset + np.random.uniform(-model_width/6, model_width/6)  # Tighter jitter
                        x_positions.append(x_pos)
                        y_positions.append(y_val)
            
            # Plot all points for this model with larger size
            ax.scatter(x_positions, y_positions, color=color, s=30, alpha=0.5,  
                      label=model, edgecolors='none')
        
        # Add mean lines for each quantile-model combination
        for q_idx, quantile in enumerate(quantile_order):
            for model_idx, model in enumerate(unique_models):
                model_quantile_data = df_metric[(df_metric['Quantile'] == quantile) & 
                                               (df_metric['Model'] == model)]['Score']
                if not model_quantile_data.empty:
                    mean_val = model_quantile_data.mean()
                    
                    # Calculate x position for this model within the quantile (using same narrower width)
                    base_x = q_idx
                    model_offset = (model_idx - (n_models - 1) / 2) * (0.6 / n_models)  # Match the narrower width
                    x_pos = base_x + model_offset
                    
                    # Add mean line (adjusted for narrower columns)
                    ax.hlines(mean_val, x_pos - (0.6 / n_models)/3, x_pos + (0.6 / n_models)/3,
                             colors=mean_color, linewidth=4, zorder=3)  # Thicker line for better visibility
        
        # Set x-axis labels
        ax.set_xticks(range(len(quantile_order)))
        ax.set_xticklabels(quantile_order)
        
        # Perform paired t-tests between baselines (after initial plot setup)
        self._add_paired_ttest_to_plot(df_metric, metric, quantile_order, unique_models, ax, n_models)
        
        # Add legend
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.title(f'{metric.upper()} by DEG Quantile - {covariate}', fontsize=14, fontweight='bold')
        plt.xlabel('DEG Count Quantile', fontsize=12)
        plt.ylabel(f'{metric.upper()} Score', fontsize=12)
        plt.grid(axis='y', alpha=0.3)
        
        # Handle R² metrics: cap y-axis at -1 and annotate points below
        r2_metrics = ['r2_deltactrl', 'r2_deltactrl_degs', 'r2_deltapert', 'r2_deltapert_degs', 
                     'weighted_r2_deltactrl', 'weighted_r2_deltapert']
        if metric in r2_metrics:
            # Count points below -1
            points_below = (df_metric['Score'] < -1).sum()
            if points_below > 0:
                # Set y-axis minimum to -1
                current_ylim = ax.get_ylim()
                ax.set_ylim(-1, current_ylim[1])
                
                # Add annotation about capped points
                ax.text(0.02, 0.98, f'{points_below} points < -1 (capped)', 
                       transform=ax.transAxes, fontsize=10, 
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                       verticalalignment='top')
        
        # Add horizontal line for certain metrics
        if metric in [
            'pearson_deltactrl_degs',
            'pearson_deltapert_degs',
            'weighted_pearson_deltactrl',
            'weighted_pearson_deltapert',
            'weighted_r2_deltactrl',
            'weighted_r2_deltapert',
        ]:
            ax.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.7)
        
        plt.tight_layout()
        
        # Save plot
        filename = f"deg_quantile_impact_{metric}_{covariate}.png"
        plt.savefig(self.plots_dir / filename, dpi=self.config.output.plotting.dpi, 
                   bbox_inches='tight')
        plt.close()
        
        log.info(f"Saved {filename}")
    
    def _add_paired_ttest_to_plot(self, df_metric: pd.DataFrame, metric: str, 
                                   quantile_order: List[str], unique_models: np.ndarray,
                                   ax: plt.Axes, n_models: int) -> None:
        """Add paired t-test statistics between baselines to the plot.
        
        Args:
            df_metric: DataFrame with metric data for all models and quantiles
            metric: Name of the metric being plotted
            quantile_order: Ordered list of quantile labels
            unique_models: Array of unique model names
            ax: Matplotlib axes object
            n_models: Total number of models
        """
        # Determine which baselines to compare based on metric name
        if 'deltactrl' in metric.lower():
            # For deltactrl metrics, compare mean baseline vs technical duplicate
            baseline1_keywords = ['mean_baseline', 'split_mean']
            baseline2_keywords = ['technical_duplicate', 'technical duplicate']
        else:
            # For all other metrics, compare ctrl baseline vs technical duplicate
            baseline1_keywords = ['ctrl_baseline', 'ctrl baseline']
            baseline2_keywords = ['technical_duplicate', 'technical duplicate']
        
        # Find the baseline models
        baseline1_model = None
        baseline2_model = None
        
        for model in unique_models:
            model_lower = model.lower()
            if any(keyword in model_lower for keyword in baseline1_keywords):
                baseline1_model = model
            elif any(keyword in model_lower for keyword in baseline2_keywords):
                baseline2_model = model
        
        if baseline1_model is None or baseline2_model is None:
            return  # Can't perform t-test if baselines not found
        
        # Perform paired t-test for each quantile
        t_stats = []
        p_values = []
        
        for quantile in quantile_order:
            # Get perturbations in this quantile
            quantile_data = df_metric[df_metric['Quantile'] == quantile]
            perturbations = quantile_data['Perturbation'].unique()
            
            # Collect paired values
            baseline1_values = []
            baseline2_values = []
            
            for pert in perturbations:
                b1_val = quantile_data[(quantile_data['Model'] == baseline1_model) & 
                                       (quantile_data['Perturbation'] == pert)]['Score'].values
                b2_val = quantile_data[(quantile_data['Model'] == baseline2_model) & 
                                       (quantile_data['Perturbation'] == pert)]['Score'].values
                
                if len(b1_val) > 0 and len(b2_val) > 0:
                    baseline1_values.append(b1_val[0])
                    baseline2_values.append(b2_val[0])
            
            # Perform paired t-test if we have paired data
            if len(baseline1_values) >= 2:  # Need at least 2 pairs for t-test
                t_stat, p_val = stats.ttest_rel(baseline1_values, baseline2_values)
                t_stats.append(t_stat)
                p_values.append(p_val)
            else:
                raise ValueError(f"No paired data found for {quantile}")
        
        # Apply Bonferroni correction
        n_tests = len([p for p in p_values if not np.isnan(p)])
        if n_tests > 0:
            p_values_corrected = [p * n_tests if not np.isnan(p) else np.nan for p in p_values]
        else:
            p_values_corrected = p_values
        
        # Find positions of the two baselines
        baseline1_idx = list(unique_models).index(baseline1_model)
        baseline2_idx = list(unique_models).index(baseline2_model)
        
        # First, extend y-axis to make room for statistics boxes if needed
        current_ylim = ax.get_ylim()
        y_range = current_ylim[1] - current_ylim[0]
        
        # Check if we have any statistics to display
        has_stats = any(not np.isnan(t) for t in t_stats)
        
        if has_stats:
            # Extend the upper y-limit by 15% to make room for stats boxes
            new_upper = current_ylim[1] + y_range * 0.15
            ax.set_ylim(current_ylim[0], new_upper)
            
            # Recalculate y_lim after extension
            y_lim = ax.get_ylim()
            
            # Position stats boxes in the extended area (top 10% of plot)
            stats_y_pos = y_lim[1] - y_range * 0.12
        
        # Add text annotations between the baselines for each quantile
        for q_idx, (quantile, t_stat, p_val) in enumerate(zip(quantile_order, t_stats, p_values_corrected)):
            if not np.isnan(t_stat):
                # Calculate x position between the two baselines
                pos1 = q_idx + (baseline1_idx - (n_models - 1) / 2) * (0.6 / n_models)
                pos2 = q_idx + (baseline2_idx - (n_models - 1) / 2) * (0.6 / n_models)
                x_pos = (pos1 + pos2) / 2
                
                # Format the text - use scientific notation for p-values
                p_text = f"p={p_val:.2e}"
                
                text = f"t={t_stat:.2f}\n{p_text}"
                
                # Add text annotation
                ax.text(x_pos, stats_y_pos, text, 
                       fontsize=8, ha='center', va='center',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                                edgecolor='gray', alpha=0.8))
    
    def _create_deg_quantile_ridge_plot(self, df_quantile: pd.DataFrame, metric: str, covariate: str) -> None:
        """Create a ridge plot of normalized technical duplicate values across quantiles.
        
        Args:
            df_quantile: DataFrame containing quantile data.
            metric: Metric name to plot.
            covariate: Covariate value for plot title.
        """
        df_metric = df_quantile[df_quantile['Metric'] == metric].copy()
        
        if df_metric.empty:
            return
        
        # Sort by quantile index to ensure proper ordering
        df_metric = df_metric.sort_values('QuantileIdx')
        
        # Set categorical order for proper ordering
        quantile_order = sorted(df_metric['Quantile'].unique(), key=lambda x: int(x[1:]))
        df_metric['Quantile'] = pd.Categorical(df_metric['Quantile'], categories=quantile_order, ordered=True)
        
        # Determine which baseline to use for normalization
        if 'deltactrl' in metric.lower():
            # For deltactrl metrics, use mean baseline
            baseline_keywords = ['mean_baseline', 'split_mean']
        else:
            # For all other metrics, use ctrl baseline
            baseline_keywords = ['ctrl_baseline', 'ctrl baseline']
        
        # Find technical duplicate and baseline models
        tech_dup_keywords = ['technical_duplicate', 'technical duplicate']
        
        baseline_model = None
        tech_dup_model = None
        
        for model in df_metric['Model'].unique():
            model_lower = model.lower()
            if any(keyword in model_lower for keyword in baseline_keywords):
                baseline_model = model
            elif any(keyword in model_lower for keyword in tech_dup_keywords):
                tech_dup_model = model
        
        if baseline_model is None or tech_dup_model is None:
            log.warning(f"Could not find required baselines for ridge plot: baseline={baseline_model}, tech_dup={tech_dup_model}")
            return
        
        # Calculate normalized values for each quantile
        normalized_data = []
        
        for quantile in quantile_order:
            quantile_data = df_metric[df_metric['Quantile'] == quantile]
            perturbations = quantile_data['Perturbation'].unique()
            
            for pert in perturbations:
                tech_dup_val = quantile_data[(quantile_data['Model'] == tech_dup_model) & 
                                            (quantile_data['Perturbation'] == pert)]['Score'].values
                baseline_val = quantile_data[(quantile_data['Model'] == baseline_model) & 
                                            (quantile_data['Perturbation'] == pert)]['Score'].values
                
                if len(tech_dup_val) > 0 and len(baseline_val) > 0:
                    normalized_val = tech_dup_val[0] - baseline_val[0]
                    normalized_data.append({
                        'Quantile': quantile,
                        'QuantileIdx': int(quantile[1:]) - 1,
                        'NormalizedValue': normalized_val,
                        'Perturbation': pert
                    })
        
        if not normalized_data:
            log.warning(f"No normalized data for ridge plot")
            return
        
        df_ridge = pd.DataFrame(normalized_data)
        
        # Create ridge plot
        fig, ax = plt.subplots(figsize=(12, 8), facecolor='white')
        ax.set_facecolor('white')
        
        # Color for the ridge distributions
        ridge_color = '#2E86AB'  # Blue color for consistency
        
        # Create ridge plot with KDE
        n_quantiles = len(quantile_order)
        
        # Set up the plot spacing
        overlap = 0.6  # How much the distributions overlap
        
        # Calculate global x range for consistent KDE
        x_min_global = df_ridge['NormalizedValue'].min()
        x_max_global = df_ridge['NormalizedValue'].max()
        x_range_global = x_max_global - x_min_global
        
        # Extend x range to prevent KDE truncation and leave room for stats
        x_plot_min = x_min_global - x_range_global * 0.2
        x_plot_max = x_max_global + x_range_global * 0.4  # Extra space on right for stats
        
        # Store mean values and KDE heights for vertical lines
        quantile_means = []
        kde_heights = {}  # Store the actual KDE height for each quantile
        
        for i, quantile in enumerate(reversed(quantile_order)):  # Reverse so Q5 is at top
            quantile_data = df_ridge[df_ridge['Quantile'] == quantile]['NormalizedValue'].values
            
            if len(quantile_data) > 1:
                # Calculate mean for this quantile
                mean_val = np.mean(quantile_data)
                quantile_means.append((i, mean_val))
                
                # Calculate KDE
                from scipy.stats import gaussian_kde
                try:
                    kde = gaussian_kde(quantile_data)
                    
                    # Create x range for KDE - use extended range to prevent truncation
                    x_vals = np.linspace(x_plot_min, x_plot_max, 300)
                    
                    # Calculate KDE values
                    kde_vals = kde(x_vals)
                    
                    # Normalize KDE height
                    kde_vals = kde_vals / kde_vals.max() * overlap
                    
                    # Store the actual KDE height at the mean position for the vertical line
                    kde_at_mean = kde(mean_val)[0] / kde(x_vals).max() * overlap
                    kde_heights[i] = kde_at_mean
                    
                    # Offset for this quantile
                    y_offset = i
                    
                    # Plot the KDE as a filled area
                    ax.fill_between(x_vals, y_offset, kde_vals + y_offset, 
                                   alpha=0.7, color=ridge_color, edgecolor='black', linewidth=1)
                    
                    # Add jittered points
                    jitter = np.random.normal(0, 0.02, len(quantile_data))
                    ax.scatter(quantile_data, np.full_like(quantile_data, y_offset) + jitter + 0.05,
                             s=20, alpha=0.5, color='darkred', zorder=5)
                    
                except np.linalg.LinAlgError:
                    # If KDE fails (e.g., singular matrix), just plot points
                    jitter = np.random.normal(0, 0.05, len(quantile_data))
                    ax.scatter(quantile_data, np.full_like(quantile_data, i) + jitter,
                             s=30, alpha=0.6, color=ridge_color)
            else:
                # If only one point, just plot it
                if len(quantile_data) == 1:
                    ax.scatter(quantile_data, [i], s=50, alpha=0.8, color=ridge_color)
                    quantile_means.append((i, quantile_data[0]))
        
        # Add vertical line at zero (grey, behind everything)
        ax.axvline(x=0, color='grey', linestyle='--', linewidth=1.5, alpha=0.5, zorder=0)
        
        # Add vertical mean lines for each quantile
        mean_color = '#A23B72'  # Dark red/burgundy for mean lines (same as other plots)
        
        for y_pos, mean_val in quantile_means:
            # Get the actual KDE height for this quantile
            kde_height = kde_heights.get(y_pos, overlap * 0.9)
            
            # Draw vertical line from baseline to actual KDE height at the mean (behind other elements)
            ax.plot([mean_val, mean_val], [y_pos, y_pos + kde_height], 
                   color=mean_color, linestyle='--', linewidth=2, alpha=0.7, zorder=1)
            
            # Add mean value text below the line (in black)
            ax.text(mean_val, y_pos - 0.05, f'μ={mean_val:.3f}', 
                   ha='center', va='top', fontsize=8, color='black', 
                   fontweight='bold')
        
        # Perform paired t-tests for each quantile
        t_stats = []
        p_values = []
        
        for quantile in quantile_order:
            quantile_values = df_ridge[df_ridge['Quantile'] == quantile]['NormalizedValue'].values
            
            if len(quantile_values) >= 2:
                # Paired t-test against zero (null hypothesis: mean = 0)
                t_stat, p_val = stats.ttest_1samp(quantile_values, 0)
                t_stats.append(t_stat)
                p_values.append(p_val)
            else:
                t_stats.append(np.nan)
                p_values.append(np.nan)
        
        # Apply Bonferroni correction
        n_tests = len([p for p in p_values if not np.isnan(p)])
        if n_tests > 0:
            p_values_corrected = [p * n_tests if not np.isnan(p) else np.nan for p in p_values]
        else:
            p_values_corrected = p_values
        
        # Add t-test statistics boxes on the right
        stats_x_position = x_max_global + x_range_global * 0.15  # Position stats to the right of data
        
        for i, (quantile, t_stat, p_val) in enumerate(zip(reversed(quantile_order), 
                                                           reversed(t_stats), 
                                                           reversed(p_values_corrected))):
            if not np.isnan(t_stat):
                # Format p-value in scientific notation
                p_text = f"p={p_val:.2e}"
                stats_text = f"t={t_stat:.2f}\n{p_text}"
                
                ax.text(stats_x_position, i + 0.3, stats_text,
                       fontsize=9, ha='left', va='center',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                edgecolor='gray', alpha=0.9))
        
        # Set x-axis limits to include stats boxes
        ax.set_xlim(x_plot_min, x_plot_max)
        
        # Set y-axis labels
        ax.set_yticks(range(n_quantiles))
        ax.set_yticklabels(reversed(quantile_order))
        
        # Labels and title
        baseline_type = "Mean Baseline" if 'deltactrl' in metric.lower() else "Control Baseline"
        ax.set_xlabel(f'Technical Duplicate - {baseline_type}', fontsize=12)
        ax.set_ylabel('DEG Count Quantile', fontsize=12)
        ax.set_title(f'{metric.upper()} Ridge Plot: Normalized Technical Duplicate - {covariate}', 
                    fontsize=14, fontweight='bold')
        
        # Add grid
        ax.grid(axis='x', alpha=0.3)
        
        # Remove top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        
        # Save plot
        filename = f"deg_quantile_ridge_{metric}_{covariate}.png"
        plt.savefig(self.plots_dir / filename, dpi=self.config.output.plotting.dpi, 
                   bbox_inches='tight')
        plt.close()
        
        log.info(f"Saved {filename}")
    
    def _get_available_covariates(self) -> List[str]:
        """Get list of available covariates from results.
        
        Returns:
            Sorted list of unique covariate values.
        """
        covariates = set()
        
        for model_results in self.results_dict['models'].values():
            for cov_pert_key in model_results['metrics']['mse'].keys():
                covariate = cov_pert_key.split('_')[0]
                covariates.add(covariate)
        
        return sorted(list(covariates)) 