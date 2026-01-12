"""
Parameter Analysis and Diversity Reporting for SwellSight Pipeline
Handles parameter diversity analysis, quality assessment, and comprehensive reporting
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Tuple, Optional
from collections import defaultdict
import logging
from pathlib import Path
import json

logger = logging.getLogger(__name__)

class ParameterAnalyzer:
    """Handles parameter diversity analysis and quality assessment"""
    
    def __init__(self):
        self.analysis_results = {}
    
    def analyze_parameter_diversity(self, parameter_sets: List[Dict[str, Any]], 
                                  parameter_ranges: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Perform comprehensive parameter diversity analysis
        
        Args:
            parameter_sets: List of parameter dictionaries
            parameter_ranges: Optional parameter range definitions for validation
            
        Returns:
            Dictionary with diversity analysis results
        """
        try:
            if not parameter_sets:
                return {'error': 'No parameter sets provided for analysis'}
            
            # Extract parameter statistics
            parameter_statistics = defaultdict(list)
            
            for param_set in parameter_sets:
                parameters = param_set.get('parameters', {})
                for key, value in parameters.items():
                    if isinstance(value, (int, float)):
                        parameter_statistics[key].append(value)
            
            # Calculate diversity metrics for each parameter
            diversity_metrics = {}
            
            for param_name, values in parameter_statistics.items():
                if values:
                    values_array = np.array(values)
                    
                    # Basic statistics
                    mean_val = np.mean(values_array)
                    std_val = np.std(values_array)
                    min_val = np.min(values_array)
                    max_val = np.max(values_array)
                    median_val = np.median(values_array)
                    
                    # Advanced diversity metrics
                    unique_values = len(np.unique(values_array))
                    total_values = len(values_array)
                    uniqueness_ratio = unique_values / total_values
                    
                    # Coefficient of variation (normalized standard deviation)
                    cv = std_val / mean_val if mean_val != 0 else 0
                    
                    # Range utilization
                    param_range = max_val - min_val
                    range_utilization = 1.0  # Default
                    
                    if parameter_ranges and param_name in parameter_ranges:
                        config_range = parameter_ranges[param_name]
                        if 'min' in config_range and 'max' in config_range:
                            theoretical_range = config_range['max'] - config_range['min']
                            range_utilization = param_range / theoretical_range if theoretical_range > 0 else 0
                    
                    # Calculate overall diversity score
                    diversity_score = (uniqueness_ratio + range_utilization + min(cv, 1.0)) / 3.0
                    
                    diversity_metrics[param_name] = {
                        'mean': float(mean_val),
                        'std': float(std_val),
                        'min': float(min_val),
                        'max': float(max_val),
                        'median': float(median_val),
                        'range': float(param_range),
                        'unique_values': unique_values,
                        'total_values': total_values,
                        'uniqueness_ratio': uniqueness_ratio,
                        'coefficient_of_variation': cv,
                        'range_utilization': range_utilization,
                        'diversity_score': diversity_score
                    }
            
            # Analyze parameter combinations
            combination_analysis = self._analyze_parameter_combinations(parameter_sets)
            
            # Generate recommendations
            recommendations = self._generate_diversity_recommendations(
                diversity_metrics, combination_analysis
            )
            
            analysis_results = {
                'parameter_metrics': diversity_metrics,
                'combination_analysis': combination_analysis,
                'recommendations': recommendations,
                'overall_diversity_score': combination_analysis.get('uniqueness_rate', 0.0),
                'analysis_summary': self._create_analysis_summary(diversity_metrics, combination_analysis)
            }
            
            self.analysis_results = analysis_results
            return analysis_results
            
        except Exception as e:
            logger.error(f"Error analyzing parameter diversity: {e}")
            return {'error': f'Analysis failed: {str(e)}'}
    
    def _analyze_parameter_combinations(self, parameter_sets: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze parameter combination uniqueness and quality distribution"""
        try:
            param_signatures = set()
            quality_distribution = defaultdict(int)
            
            for param_set in parameter_sets:
                parameters = param_set.get('parameters', {})
                validation = param_set.get('validation', {})
                
                if isinstance(parameters, dict):
                    # Create signature from numeric parameters (rounded for reasonable uniqueness)
                    numeric_params = {}
                    for k, v in parameters.items():
                        if isinstance(v, (int, float)):
                            if isinstance(v, float):
                                numeric_params[k] = round(v, 3)
                            else:
                                numeric_params[k] = v
                    
                    param_signature = tuple(sorted(numeric_params.items()))
                    param_signatures.add(param_signature)
                    
                    # Track quality distribution
                    quality_score = validation.get('quality_score', 0.0)
                    quality_bin = f"{quality_score:.1f}"
                    quality_distribution[quality_bin] += 1
            
            unique_combinations = len(param_signatures)
            total_combinations = len(parameter_sets)
            uniqueness_rate = unique_combinations / total_combinations if total_combinations > 0 else 0
            
            return {
                'total_combinations': total_combinations,
                'unique_combinations': unique_combinations,
                'duplicate_combinations': total_combinations - unique_combinations,
                'uniqueness_rate': uniqueness_rate,
                'quality_distribution': dict(quality_distribution)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing parameter combinations: {e}")
            return {'error': f'Combination analysis failed: {str(e)}'}
    
    def _generate_diversity_recommendations(self, diversity_metrics: Dict[str, Any], 
                                          combination_analysis: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on diversity analysis"""
        recommendations = []
        
        try:
            # Check individual parameter diversity
            low_diversity_params = []
            for param_name, metrics in diversity_metrics.items():
                if metrics['diversity_score'] < 0.5:
                    low_diversity_params.append(param_name)
            
            if low_diversity_params:
                recommendations.append(f"⚠️  Consider increasing variation for: {', '.join(low_diversity_params)}")
            else:
                recommendations.append("✅ All parameters show good individual diversity")
            
            # Check overall combination diversity
            uniqueness_rate = combination_analysis.get('uniqueness_rate', 0)
            if uniqueness_rate < 0.8:
                recommendations.append("💡 Consider adjusting parameter ranges to increase combination diversity")
            else:
                recommendations.append("✅ Excellent parameter combination diversity")
            
            # Check quality distribution
            quality_dist = combination_analysis.get('quality_distribution', {})
            total_combinations = combination_analysis.get('total_combinations', 1)
            
            high_quality_count = sum(count for quality_bin, count in quality_dist.items() 
                                   if float(quality_bin) >= 0.8)
            high_quality_ratio = high_quality_count / total_combinations
            
            if high_quality_ratio < 0.7:
                recommendations.append("⚠️  Consider adjusting parameter ranges to improve quality scores")
            else:
                recommendations.append("✅ Good quality score distribution")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            return ["Unable to generate recommendations due to error"]
    
    def _create_analysis_summary(self, diversity_metrics: Dict[str, Any], 
                               combination_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Create a summary of the analysis results"""
        try:
            # Calculate overall parameter diversity
            if diversity_metrics:
                avg_diversity = np.mean([metrics['diversity_score'] for metrics in diversity_metrics.values()])
                avg_uniqueness = np.mean([metrics['uniqueness_ratio'] for metrics in diversity_metrics.values()])
                avg_range_utilization = np.mean([metrics['range_utilization'] for metrics in diversity_metrics.values()])
            else:
                avg_diversity = avg_uniqueness = avg_range_utilization = 0.0
            
            # Quality assessment
            quality_dist = combination_analysis.get('quality_distribution', {})
            total_combinations = combination_analysis.get('total_combinations', 1)
            
            high_quality_count = sum(count for quality_bin, count in quality_dist.items() 
                                   if float(quality_bin) >= 0.8)
            medium_quality_count = sum(count for quality_bin, count in quality_dist.items() 
                                     if 0.6 <= float(quality_bin) < 0.8)
            low_quality_count = total_combinations - high_quality_count - medium_quality_count
            
            return {
                'total_parameters_analyzed': len(diversity_metrics),
                'average_parameter_diversity': avg_diversity,
                'average_uniqueness_ratio': avg_uniqueness,
                'average_range_utilization': avg_range_utilization,
                'combination_uniqueness_rate': combination_analysis.get('uniqueness_rate', 0),
                'quality_assessment': {
                    'high_quality_ratio': high_quality_count / total_combinations,
                    'medium_quality_ratio': medium_quality_count / total_combinations,
                    'low_quality_ratio': low_quality_count / total_combinations
                },
                'overall_assessment': self._get_overall_assessment(avg_diversity, combination_analysis.get('uniqueness_rate', 0))
            }
            
        except Exception as e:
            logger.error(f"Error creating analysis summary: {e}")
            return {'error': f'Summary creation failed: {str(e)}'}
    
    def _get_overall_assessment(self, avg_diversity: float, uniqueness_rate: float) -> str:
        """Get overall assessment based on diversity metrics"""
        combined_score = (avg_diversity + uniqueness_rate) / 2.0
        
        if combined_score >= 0.9:
            return "Exceptional"
        elif combined_score >= 0.8:
            return "Excellent"
        elif combined_score >= 0.7:
            return "Very Good"
        elif combined_score >= 0.6:
            return "Good"
        elif combined_score >= 0.5:
            return "Moderate"
        else:
            return "Poor"
    
    def create_diversity_visualizations(self, output_dir: str = "./outputs/visualizations") -> Dict[str, str]:
        """
        Create visualizations for parameter diversity analysis
        
        Args:
            output_dir: Directory to save visualization files
            
        Returns:
            Dictionary with paths to created visualization files
        """
        try:
            if not self.analysis_results:
                return {'error': 'No analysis results available for visualization'}
            
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            visualization_files = {}
            
            # Set up matplotlib style
            plt.style.use('default')
            sns.set_palette("husl")
            
            diversity_metrics = self.analysis_results.get('parameter_metrics', {})
            combination_analysis = self.analysis_results.get('combination_analysis', {})
            
            # 1. Parameter diversity scores bar chart
            if diversity_metrics:
                fig, ax = plt.subplots(figsize=(12, 6))
                
                param_names = list(diversity_metrics.keys())
                diversity_scores = [metrics['diversity_score'] for metrics in diversity_metrics.values()]
                
                bars = ax.bar(param_names, diversity_scores, color='skyblue', alpha=0.7)
                ax.set_ylabel('Diversity Score')\n                ax.set_title('Parameter Diversity Scores')\n                ax.set_ylim(0, 1.0)\n                \n                # Add value labels on bars\n                for bar, score in zip(bars, diversity_scores):\n                    height = bar.get_height()\n                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,\n                            f'{score:.3f}', ha='center', va='bottom')\n                \n                plt.xticks(rotation=45, ha='right')\n                plt.tight_layout()\n                \n                diversity_chart_path = output_path / 'parameter_diversity_scores.png'\n                plt.savefig(diversity_chart_path, dpi=300, bbox_inches='tight')\n                plt.close()\n                \n                visualization_files['diversity_scores'] = str(diversity_chart_path)\n            \n            # 2. Parameter distribution histograms\n            if diversity_metrics and len(diversity_metrics) > 0:\n                n_params = len(diversity_metrics)\n                cols = min(3, n_params)\n                rows = (n_params + cols - 1) // cols\n                \n                fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))\n                if n_params == 1:\n                    axes = [axes]\n                elif rows == 1:\n                    axes = [axes]\n                else:\n                    axes = axes.flatten()\n                \n                for i, (param_name, metrics) in enumerate(diversity_metrics.items()):\n                    if i < len(axes):\n                        ax = axes[i]\n                        \n                        # Create histogram data (simulated from statistics)\n                        mean = metrics['mean']\n                        std = metrics['std']\n                        min_val = metrics['min']\n                        max_val = metrics['max']\n                        \n                        # Generate sample data for visualization\n                        sample_data = np.random.normal(mean, std, 1000)\n                        sample_data = np.clip(sample_data, min_val, max_val)\n                        \n                        ax.hist(sample_data, bins=30, alpha=0.7, color='lightcoral')\n                        ax.set_title(f'{param_name} Distribution')\n                        ax.set_xlabel('Value')\n                        ax.set_ylabel('Frequency')\n                        ax.axvline(mean, color='red', linestyle='--', label=f'Mean: {mean:.3f}')\n                        ax.legend()\n                \n                # Hide unused subplots\n                for i in range(n_params, len(axes)):\n                    axes[i].set_visible(False)\n                \n                plt.tight_layout()\n                \n                distribution_chart_path = output_path / 'parameter_distributions.png'\n                plt.savefig(distribution_chart_path, dpi=300, bbox_inches='tight')\n                plt.close()\n                \n                visualization_files['distributions'] = str(distribution_chart_path)\n            \n            # 3. Quality distribution pie chart\n            quality_dist = combination_analysis.get('quality_distribution', {})\n            if quality_dist:\n                fig, ax = plt.subplots(figsize=(10, 8))\n                \n                # Group quality scores into categories\n                high_quality = sum(count for quality_bin, count in quality_dist.items() \n                                 if float(quality_bin) >= 0.8)\n                medium_quality = sum(count for quality_bin, count in quality_dist.items() \n                                   if 0.6 <= float(quality_bin) < 0.8)\n                low_quality = sum(count for quality_bin, count in quality_dist.items() \n                                if float(quality_bin) < 0.6)\n                \n                categories = ['High Quality (≥0.8)', 'Medium Quality (0.6-0.8)', 'Low Quality (<0.6)']\n                values = [high_quality, medium_quality, low_quality]\n                colors = ['lightgreen', 'gold', 'lightcoral']\n                \n                # Filter out zero values\n                non_zero_data = [(cat, val, col) for cat, val, col in zip(categories, values, colors) if val > 0]\n                if non_zero_data:\n                    categories, values, colors = zip(*non_zero_data)\n                    \n                    wedges, texts, autotexts = ax.pie(values, labels=categories, colors=colors, \n                                                     autopct='%1.1f%%', startangle=90)\n                    ax.set_title('Parameter Quality Distribution')\n                    \n                    quality_chart_path = output_path / 'quality_distribution.png'\n                    plt.savefig(quality_chart_path, dpi=300, bbox_inches='tight')\n                    plt.close()\n                    \n                    visualization_files['quality_distribution'] = str(quality_chart_path)\n            \n            return visualization_files\n            \n        except Exception as e:\n            logger.error(f\"Error creating visualizations: {e}\")\n            return {'error': f'Visualization creation failed: {str(e)}'}\n    \n    def generate_comprehensive_report(self, output_path: str = \"./outputs/parameter_analysis_report.json\") -> bool:\n        \"\"\"\n        Generate comprehensive parameter analysis report\n        \n        Args:\n            output_path: Path to save the report\n            \n        Returns:\n            True if report generated successfully, False otherwise\n        \"\"\"\n        try:\n            if not self.analysis_results:\n                logger.error(\"No analysis results available for report generation\")\n                return False\n            \n            # Create comprehensive report\n            report = {\n                'report_metadata': {\n                    'generated_at': str(np.datetime64('now')),\n                    'report_type': 'Parameter Diversity Analysis Report',\n                    'analysis_version': '2.0'\n                },\n                'analysis_results': self.analysis_results,\n                'executive_summary': self._create_executive_summary(),\n                'detailed_findings': self._create_detailed_findings(),\n                'recommendations': self.analysis_results.get('recommendations', []),\n                'technical_details': {\n                    'methodology': 'Advanced parameter diversity analysis with uniqueness, range utilization, and quality assessment',\n                    'metrics_calculated': [\n                        'Uniqueness ratio',\n                        'Range utilization',\n                        'Coefficient of variation',\n                        'Diversity score',\n                        'Combination uniqueness',\n                        'Quality distribution'\n                    ]\n                }\n            }\n            \n            # Save report\n            output_file = Path(output_path)\n            output_file.parent.mkdir(parents=True, exist_ok=True)\n            \n            with open(output_file, 'w') as f:\n                json.dump(report, f, indent=2, default=str)\n            \n            logger.info(f\"Parameter analysis report saved to: {output_file}\")\n            return True\n            \n        except Exception as e:\n            logger.error(f\"Error generating comprehensive report: {e}\")\n            return False\n    \n    def _create_executive_summary(self) -> Dict[str, Any]:\n        \"\"\"Create executive summary of analysis results\"\"\"\n        try:\n            summary = self.analysis_results.get('analysis_summary', {})\n            combination_analysis = self.analysis_results.get('combination_analysis', {})\n            \n            return {\n                'overall_assessment': summary.get('overall_assessment', 'Unknown'),\n                'total_parameters_analyzed': summary.get('total_parameters_analyzed', 0),\n                'combination_uniqueness': f\"{combination_analysis.get('uniqueness_rate', 0)*100:.1f}%\",\n                'average_parameter_diversity': f\"{summary.get('average_parameter_diversity', 0):.3f}\",\n                'quality_assessment': summary.get('quality_assessment', {}),\n                'key_findings': [\n                    f\"Analyzed {summary.get('total_parameters_analyzed', 0)} parameter types\",\n                    f\"Achieved {combination_analysis.get('uniqueness_rate', 0)*100:.1f}% combination uniqueness\",\n                    f\"Overall assessment: {summary.get('overall_assessment', 'Unknown')}\"\n                ]\n            }\n            \n        except Exception as e:\n            logger.error(f\"Error creating executive summary: {e}\")\n            return {'error': 'Failed to create executive summary'}\n    \n    def _create_detailed_findings(self) -> Dict[str, Any]:\n        \"\"\"Create detailed findings from analysis results\"\"\"\n        try:\n            diversity_metrics = self.analysis_results.get('parameter_metrics', {})\n            combination_analysis = self.analysis_results.get('combination_analysis', {})\n            \n            findings = {\n                'parameter_analysis': {},\n                'combination_analysis': combination_analysis,\n                'diversity_assessment': {},\n                'quality_assessment': {}\n            }\n            \n            # Parameter-specific findings\n            for param_name, metrics in diversity_metrics.items():\n                findings['parameter_analysis'][param_name] = {\n                    'diversity_score': metrics['diversity_score'],\n                    'uniqueness_ratio': metrics['uniqueness_ratio'],\n                    'range_utilization': metrics['range_utilization'],\n                    'coefficient_of_variation': metrics['coefficient_of_variation'],\n                    'assessment': self._assess_parameter_diversity(metrics['diversity_score'])\n                }\n            \n            # Overall diversity assessment\n            if diversity_metrics:\n                avg_diversity = np.mean([m['diversity_score'] for m in diversity_metrics.values()])\n                findings['diversity_assessment'] = {\n                    'average_diversity_score': avg_diversity,\n                    'diversity_rating': self._assess_parameter_diversity(avg_diversity),\n                    'parameters_with_low_diversity': [\n                        name for name, metrics in diversity_metrics.items()\n                        if metrics['diversity_score'] < 0.5\n                    ]\n                }\n            \n            return findings\n            \n        except Exception as e:\n            logger.error(f\"Error creating detailed findings: {e}\")\n            return {'error': 'Failed to create detailed findings'}\n    \n    def _assess_parameter_diversity(self, diversity_score: float) -> str:\n        \"\"\"Assess parameter diversity based on score\"\"\"\n        if diversity_score >= 0.8:\n            return \"Excellent\"\n        elif diversity_score >= 0.6:\n            return \"Good\"\n        elif diversity_score >= 0.4:\n            return \"Moderate\"\n        else:\n            return \"Poor\"\n\n\n# Convenience functions for direct use in notebooks\ndef analyze_parameter_diversity(parameter_sets: List[Dict[str, Any]], \n                              parameter_ranges: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:\n    \"\"\"Analyze parameter diversity for a list of parameter sets\"\"\"\n    analyzer = ParameterAnalyzer()\n    return analyzer.analyze_parameter_diversity(parameter_sets, parameter_ranges)\n\ndef create_parameter_report(parameter_sets: List[Dict[str, Any]], \n                          parameter_ranges: Optional[Dict[str, Any]] = None,\n                          output_path: str = \"./outputs/parameter_analysis_report.json\") -> bool:\n    \"\"\"Create comprehensive parameter analysis report\"\"\"\n    analyzer = ParameterAnalyzer()\n    analyzer.analyze_parameter_diversity(parameter_sets, parameter_ranges)\n    return analyzer.generate_comprehensive_report(output_path)\n