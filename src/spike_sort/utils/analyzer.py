"""
Results Analysis Module for Spike Sorting Analysis

Provides tools for analyzing and visualizing experiment results.
"""

import json
import pickle
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.figure import Figure
import warnings

warnings.filterwarnings('ignore')


class ResultsAnalyzer:
    """
    Class for analyzing and visualizing experiment results.
    """
    
    def __init__(self, results: Optional[List[Dict[str, Any]]] = None):
        """
        Initialize the results analyzer.
        
        Args:
            results: List of experiment results (optional, can load later)
        """
        self.results = results if results is not None else []
        self.df = None
        
        if self.results:
            self._create_dataframe()
    
    def load_results(self, filepath: str):
        """
        Load results from file.
        
        Args:
            filepath: Path to results file (.json or .pkl)
        """
        filepath = Path(filepath)
        
        if filepath.suffix == '.json':
            with open(filepath, 'r') as f:
                self.results = json.load(f)
        elif filepath.suffix == '.pkl':
            with open(filepath, 'rb') as f:
                self.results = pickle.load(f)
        else:
            raise ValueError(f"Unsupported file format: {filepath.suffix}")
        
        self._create_dataframe()
        print(f"Loaded {len(self.results)} results from {filepath}")
    
    def _create_dataframe(self):
        """Create a pandas DataFrame from results for easier analysis."""
        if not self.results:
            self.df = pd.DataFrame()
            return
        
        # Extract relevant information into flat structure
        rows = []
        for result in self.results:
            if not result.get('success', False):
                continue
            
            row = {
                'experiment_idx': result.get('experiment_idx'),
                'dim_reduction_method': result.get('dim_reduction_method'),
                'clustering_method': result.get('clustering_method'),
                'dataset_name': result.get('dataset_name'),
                'success': result.get('success'),
            }
            
            # Add dimensionality reduction metadata
            dim_meta = result.get('dim_reduction_metadata', {})
            row['dim_reduction_time'] = dim_meta.get('computation_time')
            row['dim_reduction_memory'] = dim_meta.get('memory_usage_mb')
            
            # Add clustering metadata
            clust_meta = result.get('clustering_metadata', {})
            row['clustering_time'] = clust_meta.get('computation_time')
            row['n_clusters'] = clust_meta.get('n_clusters')
            
            # Add evaluation metrics
            eval_results = result.get('evaluation', {})
            for metric, value in eval_results.items():
                row[metric] = value
            
            # Add parameter information as strings for grouping
            row['dim_params_str'] = str(result.get('dim_reduction_params', {}))
            row['clust_params_str'] = str(result.get('clustering_params', {}))
            
            rows.append(row)
        
        self.df = pd.DataFrame(rows)
    
    def get_summary_statistics(self) -> pd.DataFrame:
        """
        Get summary statistics for all metrics.
        
        Returns:
            DataFrame with summary statistics
        """
        if self.df is None or self.df.empty:
            return pd.DataFrame()
        
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        return self.df[numeric_cols].describe()
    
    def get_successful_results(self) -> List[Dict[str, Any]]:
        return [r for r in self.results if r.get('success')]
    
    def get_best_configurations(self, 
                               metric: str = 'silhouette_score',
                               n_top: int = 10,
                               higher_is_better: bool = True) -> pd.DataFrame:
        """
        Get the best performing configurations.
        
        Args:
            metric: Metric to rank by
            n_top: Number of top configurations to return
            higher_is_better: Whether higher values are better
            
        Returns:
            DataFrame with top configurations
        """
        if self.df is None or self.df.empty:
            return pd.DataFrame()
        
        # Filter out rows with missing metric values
        valid_df = self.df[self.df[metric].notna()].copy()
        
        if valid_df.empty:
            print(f"No valid results for metric: {metric}")
            return pd.DataFrame()
        
        # Sort by metric
        sorted_df = valid_df.sort_values(by=metric, ascending=not higher_is_better)
        
        # Select relevant columns
        cols = [
            'experiment_idx', 'dim_reduction_method', 'clustering_method',
            metric, 'n_clusters', 'dim_reduction_time', 'clustering_time'
        ]
        cols = [c for c in cols if c in sorted_df.columns]
        
        return sorted_df[cols].head(n_top)
    
    def create_performance_heatmap(self, 
                                  metric: str = 'silhouette_score',
                                  aggregation: str = 'mean',
                                  figsize: Tuple[int, int] = (12, 8),
                                  save_path: Optional[str] = None) -> Figure:
        """
        Create a heatmap showing performance across method combinations.
        
        Args:
            metric: Metric to visualize
            aggregation: How to aggregate multiple results ('mean', 'max', 'median')
            figsize: Figure size
            save_path: Optional path to save the figure
            
        Returns:
            Matplotlib Figure object
        """
        if self.df is None or self.df.empty:
            print("No data available for visualization")
            return None
        
        # Filter valid data
        valid_df = self.df[self.df[metric].notna()].copy()
        
        if valid_df.empty:
            print(f"No valid data for metric: {metric}")
            return None
        
        # Aggregate by method combination
        if aggregation == 'mean':
            pivot_data = valid_df.pivot_table(
                values=metric,
                index='dim_reduction_method',
                columns='clustering_method',
                aggfunc='mean'
            )
        elif aggregation == 'max':
            pivot_data = valid_df.pivot_table(
                values=metric,
                index='dim_reduction_method',
                columns='clustering_method',
                aggfunc='max'
            )
        elif aggregation == 'median':
            pivot_data = valid_df.pivot_table(
                values=metric,
                index='dim_reduction_method',
                columns='clustering_method',
                aggfunc='median'
            )
        else:
            raise ValueError(f"Unknown aggregation: {aggregation}")
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap='RdYlGn', 
                   center=pivot_data.mean().mean(), ax=ax, cbar_kws={'label': metric})
        
        ax.set_title(f'{metric.replace("_", " ").title()} ({aggregation})', 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Clustering Method', fontsize=12)
        ax.set_ylabel('Dimensionality Reduction Method', fontsize=12)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Heatmap saved to: {save_path}")
        
        return fig
    
    def create_method_comparison_barplot(self,
                                        metric: str = 'silhouette_score',
                                        method_type: str = 'dim_reduction',
                                        figsize: Tuple[int, int] = (12, 6),
                                        save_path: Optional[str] = None) -> Figure:
        """
        Create a bar plot comparing methods.
        
        Args:
            metric: Metric to compare
            method_type: 'dim_reduction' or 'clustering'
            figsize: Figure size
            save_path: Optional path to save the figure
            
        Returns:
            Matplotlib Figure object
        """
        if self.df is None or self.df.empty:
            print("No data available for visualization")
            return None
        
        valid_df = self.df[self.df[metric].notna()].copy()
        
        if valid_df.empty:
            print(f"No valid data for metric: {metric}")
            return None
        
        # Select method column
        if method_type == 'dim_reduction':
            method_col = 'dim_reduction_method'
            title_prefix = 'Dimensionality Reduction'
        elif method_type == 'clustering':
            method_col = 'clustering_method'
            title_prefix = 'Clustering'
        else:
            raise ValueError(f"Unknown method_type: {method_type}")
        
        # Aggregate by method
        method_stats = valid_df.groupby(method_col)[metric].agg(['mean', 'std', 'count'])
        method_stats = method_stats.sort_values('mean', ascending=False)
        
        # Create bar plot
        fig, ax = plt.subplots(figsize=figsize)
        
        x = np.arange(len(method_stats))
        bars = ax.bar(x, method_stats['mean'], yerr=method_stats['std'], 
                     capsize=5, alpha=0.7, color='steelblue', edgecolor='black')
        
        ax.set_xlabel('Method', fontsize=12)
        ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=12)
        ax.set_title(f'{title_prefix} Methods: {metric.replace("_", " ").title()}',
                    fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(method_stats.index, rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for i, (bar, mean_val) in enumerate(zip(bars, method_stats['mean'])):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{mean_val:.3f}',
                   ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Bar plot saved to: {save_path}")
        
        return fig
    
    def create_runtime_comparison(self,
                                 figsize: Tuple[int, int] = (14, 6),
                                 save_path: Optional[str] = None) -> Figure:
        """
        Create a comparison of runtime across methods.
        
        Args:
            figsize: Figure size
            save_path: Optional path to save the figure
            
        Returns:
            Matplotlib Figure object
        """
        if self.df is None or self.df.empty:
            print("No data available for visualization")
            return None
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Dimensionality reduction runtime
        dim_runtime = self.df.groupby('dim_reduction_method')['dim_reduction_time'].mean().sort_values()
        dim_runtime.plot(kind='barh', ax=ax1, color='coral', edgecolor='black')
        ax1.set_xlabel('Mean Runtime (seconds)', fontsize=11)
        ax1.set_ylabel('Dimensionality Reduction Method', fontsize=11)
        ax1.set_title('Dimensionality Reduction Runtime', fontsize=12, fontweight='bold')
        ax1.grid(axis='x', alpha=0.3)
        
        # Clustering runtime
        clust_runtime = self.df.groupby('clustering_method')['clustering_time'].mean().sort_values()
        clust_runtime.plot(kind='barh', ax=ax2, color='lightblue', edgecolor='black')
        ax2.set_xlabel('Mean Runtime (seconds)', fontsize=11)
        ax2.set_ylabel('Clustering Method', fontsize=11)
        ax2.set_title('Clustering Runtime', fontsize=12, fontweight='bold')
        ax2.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Runtime comparison saved to: {save_path}")
        
        return fig
    
    def create_scatter_plot(self,
                           x_metric: str,
                           y_metric: str,
                           color_by: str = 'dim_reduction_method',
                           figsize: Tuple[int, int] = (10, 8),
                           save_path: Optional[str] = None) -> Figure:
        """
        Create a scatter plot of two metrics.
        
        Args:
            x_metric: Metric for x-axis
            y_metric: Metric for y-axis
            color_by: Column to use for coloring points
            figsize: Figure size
            save_path: Optional path to save the figure
            
        Returns:
            Matplotlib Figure object
        """
        if self.df is None or self.df.empty:
            print("No data available for visualization")
            return None
        
        valid_df = self.df[(self.df[x_metric].notna()) & (self.df[y_metric].notna())].copy()
        
        if valid_df.empty:
            print(f"No valid data for metrics: {x_metric}, {y_metric}")
            return None
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create scatter plot with colors
        unique_categories = valid_df[color_by].unique()
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_categories)))
        
        for category, color in zip(unique_categories, colors):
            mask = valid_df[color_by] == category
            ax.scatter(valid_df[mask][x_metric], valid_df[mask][y_metric],
                      label=category, alpha=0.6, s=100, color=color, edgecolors='black')
        
        ax.set_xlabel(x_metric.replace('_', ' ').title(), fontsize=12)
        ax.set_ylabel(y_metric.replace('_', ' ').title(), fontsize=12)
        ax.set_title(f'{y_metric.replace("_", " ").title()} vs {x_metric.replace("_", " ").title()}',
                    fontsize=14, fontweight='bold')
        ax.legend(title=color_by.replace('_', ' ').title(), bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Scatter plot saved to: {save_path}")
        
        return fig
    
    def create_hyperparameter_analysis(self,
                                      method_name: str,
                                      method_type: str,
                                      metric: str = 'silhouette_score',
                                      figsize: Tuple[int, int] = (12, 6),
                                      save_path: Optional[str] = None) -> Figure:
        """
        Analyze the effect of hyperparameters on performance.
        
        Args:
            method_name: Name of the method to analyze
            method_type: 'dim_reduction' or 'clustering'
            metric: Metric to analyze
            figsize: Figure size
            save_path: Optional path to save the figure
            
        Returns:
            Matplotlib Figure object
        """
        if self.df is None or self.df.empty:
            print("No data available for visualization")
            return None
        
        # Filter by method
        if method_type == 'dim_reduction':
            method_col = 'dim_reduction_method'
        elif method_type == 'clustering':
            method_col = 'clustering_method'
        else:
            raise ValueError(f"Unknown method_type: {method_type}")
        
        method_df = self.df[self.df[method_col] == method_name].copy()
        
        if method_df.empty:
            print(f"No data for method: {method_name}")
            return None
        
        # Group by parameter configuration
        param_col = 'dim_params_str' if method_type == 'dim_reduction' else 'clust_params_str'
        param_stats = method_df.groupby(param_col)[metric].agg(['mean', 'std', 'count'])
        param_stats = param_stats.sort_values('mean', ascending=False)
        
        # Create bar plot
        fig, ax = plt.subplots(figsize=figsize)
        
        x = np.arange(len(param_stats))
        bars = ax.bar(x, param_stats['mean'], yerr=param_stats['std'],
                     capsize=5, alpha=0.7, color='mediumseagreen', edgecolor='black')
        
        ax.set_xlabel('Parameter Configuration', fontsize=12)
        ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=12)
        ax.set_title(f'{method_name}: Hyperparameter Analysis ({metric.replace("_", " ").title()})',
                    fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([f'Config {i+1}' for i in range(len(param_stats))], 
                          rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Hyperparameter analysis saved to: {save_path}")
        
        return fig
    
    def export_summary_table(self, 
                            output_path: str,
                            format: str = 'csv',
                            metric: str = 'silhouette_score'):
        """
        Export a summary table of results.
        
        Args:
            output_path: Path to save the table
            format: Output format ('csv', 'latex', 'markdown', 'excel')
            metric: Primary metric to include
        """
        if self.df is None or self.df.empty:
            print("No data available for export")
            return
        
        # Create summary by method combination
        summary = self.df.groupby(['dim_reduction_method', 'clustering_method']).agg({
            metric: ['mean', 'std', 'count'],
            'n_clusters': 'mean',
            'dim_reduction_time': 'mean',
            'clustering_time': 'mean'
        }).round(4)
        
        # Flatten column names
        summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
        summary = summary.reset_index()
        
        # Export based on format
        output_path = Path(output_path)
        
        if format == 'csv':
            summary.to_csv(output_path, index=False)
        elif format == 'latex':
            latex_str = summary.to_latex(index=False)
            with open(output_path, 'w') as f:
                f.write(latex_str)
        elif format == 'markdown':
            markdown_str = summary.to_markdown(index=False)
            with open(output_path, 'w') as f:
                f.write(markdown_str)
        elif format == 'excel':
            summary.to_excel(output_path, index=False)
        else:
            raise ValueError(f"Unknown format: {format}")
        
        print(f"Summary table exported to: {output_path}")
    
    def generate_full_report(self, 
                            output_dir: str,
                            metrics: List[str] = None):
        """
        Generate a complete analysis report with all visualizations.
        
        Args:
            output_dir: Directory to save all outputs
            metrics: List of metrics to analyze (default: common metrics)
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if metrics is None:
            metrics = ['silhouette_score', 'davies_bouldin_index', 
                      'calinski_harabasz_index', 'adjusted_rand_index']
        
        print(f"\nGenerating full analysis report in: {output_dir}")
        print("="*70)
        
        # 1. Summary statistics
        print("\n1. Exporting summary statistics...")
        summary_stats = self.get_summary_statistics()
        summary_stats.to_csv(output_dir / 'summary_statistics.csv')
        
        # 2. Best configurations for each metric
        print("2. Exporting best configurations...")
        for metric in metrics:
            if metric in self.df.columns:
                best_configs = self.get_best_configurations(metric=metric, n_top=10)
                if not best_configs.empty:
                    best_configs.to_csv(output_dir / f'best_configs_{metric}.csv', index=False)
        
        # 3. Performance heatmaps
        print("3. Creating performance heatmaps...")
        for metric in metrics:
            if metric in self.df.columns:
                self.create_performance_heatmap(
                    metric=metric,
                    save_path=output_dir / f'heatmap_{metric}.png'
                )
                plt.close()
        
        # 4. Method comparison bar plots
        print("4. Creating method comparison plots...")
        for metric in metrics:
            if metric in self.df.columns:
                self.create_method_comparison_barplot(
                    metric=metric,
                    method_type='dim_reduction',
                    save_path=output_dir / f'barplot_dimreduction_{metric}.png'
                )
                plt.close()
                
                self.create_method_comparison_barplot(
                    metric=metric,
                    method_type='clustering',
                    save_path=output_dir / f'barplot_clustering_{metric}.png'
                )
                plt.close()
        
        # 5. Runtime comparison
        print("5. Creating runtime comparison...")
        self.create_runtime_comparison(
            save_path=output_dir / 'runtime_comparison.png'
        )
        plt.close()
        
        # 6. Export summary tables
        print("6. Exporting summary tables...")
        for metric in metrics:
            if metric in self.df.columns:
                self.export_summary_table(
                    output_path=output_dir / f'summary_table_{metric}.csv',
                    format='csv',
                    metric=metric
                )
        
        print("\n" + "="*70)
        print(f"Report generation complete! All files saved to: {output_dir}")
        print("="*70 + "\n")
