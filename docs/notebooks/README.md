# Jupyter Notebooks for Monet Stats

This directory contains comprehensive Jupyter notebooks demonstrating the usage of Monet Stats for climate analysis and model evaluation.

## Available Notebooks

### 1. Basic Statistical Analysis ([`01_basic_statistical_analysis.ipynb`](01_basic_statistical_analysis.ipynb))

- **Focus**: Introduction to core statistical metrics and workflows
- **Topics**: Error metrics, skill scores, correlation analysis, visualization
- **Data**: Temperature datasets with station-wise analysis
- **Level**: Beginner

### 2. Model-Observation Comparison ([`02_model_observation_comparison.ipynb`](02_model_observation_comparison.ipynb))

- **Focus**: Comprehensive model verification techniques
- **Topics**: Multi-variable analysis (temperature, precipitation, wind), performance by variable range, seasonal analysis
- **Data**: Temperature, precipitation, and wind datasets
- **Level**: Intermediate

<<<<<<< HEAD:docs/notebooks/README.md
### 3. Spatial Analysis and Downscaling ([`03_spatial_analysis_downscaling_fixed.ipynb`](03_spatial_analysis_downscaling_fixed.ipynb))
=======
### 3. Spatial Analysis and Downscaling ([`03_spatial_analysis_downscaling.ipynb`](03_spatial_analysis_downscaling.ipynb))
>>>>>>> main:notebooks/README.md

- **Focus**: Spatial verification and downscaling workflows
- **Topics**: Spatial correlation, pattern verification, grid-scale analysis
- **Data**: Spatial temperature fields with xarray integration
- **Level**: Advanced

### 4. Ensemble Analysis and Multi-Model Evaluation ([`04_ensemble_analysis_multi_model.ipynb`](04_ensemble_analysis_multi_model.ipynb))

- **Focus**: Ensemble forecasting and multi-model integration
- **Topics**: Ensemble statistics, probabilistic verification, member ranking
- **Data**: Ensemble forecast data with multiple members
- **Level**: Advanced

### 5. Performance Optimization for Large Datasets ([`05_performance_optimization_large_datasets.ipynb`](05_performance_optimization_large_datasets.ipynb))

- **Focus**: Efficient computation for large-scale atmospheric data
- **Topics**: Chunked processing, parallel computing, memory optimization
- **Data**: Synthetic large datasets for performance testing
- **Level**: Advanced

### 6. Integration Examples ([`06_integration_examples.ipynb`](06_integration_examples.ipynb))

- **Focus**: Holistic analysis combining multiple metrics and techniques
- **Topics**: Multi-dataset integration, machine learning integration, comprehensive dashboards
- **Data**: Combined temperature, precipitation, wind, and spatial data
- **Level**: Expert

## Getting Started

1. **Prerequisites**: Ensure you have Monet Stats installed with all dependencies:

   ```bash
   pip install monet-stats
   pip install matplotlib seaborn pandas numpy xarray scikit-learn psutil
   ```

2. **Example Datasets**: The notebooks use example datasets generated in the `data/` directory. Run:

   ```bash
   cd data
   python generate_example_datasets.py
   ```

3. **Running Notebooks**: Start Jupyter and open any notebook:
   ```bash
   jupyter notebook
   ```

## Notebook Structure

Each notebook follows a consistent structure:

- **Introduction**: Overview of the analysis approach
- **Data Loading**: Loading and preparing datasets
- **Analysis Workflow**: Step-by-step statistical analysis
- **Visualization**: Charts and plots for interpretation
- **Results Summary**: Key findings and insights
- **Best Practices**: Tips and recommendations

## Key Features Demonstrated

### Statistical Metrics

- Error metrics (MAE, RMSE, MBE, MAPE, MPE)
- Correlation metrics (Pearson, Spearman, R²)
- Skill scores (NSE, mNSE, rNSE, NSEm)
- Contingency analysis for categorical data
- Spatial verification metrics

### Advanced Techniques

- Ensemble analysis and verification
- Probabilistic forecasting evaluation
- Machine learning integration
- Performance optimization strategies
- Multi-variable integration workflows

### Visualization

- Comprehensive performance dashboards
- Time series analysis plots
- Spatial distribution maps
- Radar charts for multi-metric comparison
- Statistical diagnostic plots

## Learning Path

1. **Start with** [`01_basic_statistical_analysis.ipynb`](01_basic_statistical_analysis.ipynb) for foundational concepts
2. **Progress to** [`02_model_observation_comparison.ipynb`](02_model_observation_comparison.ipynb) for multi-variable analysis
3. **Explore** [`03_spatial_analysis_downscaling_fixed.ipynb`](03_spatial_analysis_downscaling_fixed.ipynb) for spatial applications
4. **Study** [`04_ensemble_analysis_multi_model.ipynb`](04_ensemble_analysis_multi_model.ipynb) for ensemble techniques
5. **Optimize** with [`05_performance_optimization_large_datasets.ipynb`](05_performance_optimization_large_datasets.ipynb) for large data
6. **Integrate** using [`06_integration_examples.ipynb`](06_integration_examples.ipynb) for comprehensive workflows

## Contributing

We welcome contributions to improve these notebooks! Please:

1. Follow the existing structure and style
2. Add clear documentation and explanations
3. Include visualizations where appropriate
4. Test with different datasets as needed
5. Submit pull requests with detailed descriptions

## Support

For questions or issues:

- Check the [Monet Stats documentation](../docs/)
- Review existing issues on GitHub
- Create a new issue with detailed description
- Contact the development team

---

_These notebooks provide practical examples and workflows for using Monet Stats in atmospheric sciences research and applications._
