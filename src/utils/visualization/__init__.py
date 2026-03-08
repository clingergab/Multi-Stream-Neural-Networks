"""Visualization utilities for Multi-Stream Neural Networks."""

from .training_plots import (
    plot_training_curves,
    plot_pathway_importance,
    plot_mixing_weights_evolution,
    plot_loss_landscape
)

from .pathway_analysis import (
    visualize_pathway_activations,
    plot_pathway_comparison,
    analyze_pathway_contributions,
    create_pathway_summary_plot
)

from .mixing_weights import (
    scalar_weights,
    channel_weights,
    dynamic_weights,
    spatial_weights
)

from .stream_visualization import (
    FeatureMapVisualizer,
    StreamContributionVisualizer,
    StreamGradCAM,
    IntegrationWeightVisualizer,
    find_misclassified,
    compare_samples,
)

from .stream_analysis import (
    StreamRedundancyAnalyzer,
    PerClassDominanceAnalyzer,
    ActivationDivergenceAnalyzer,
    IntegrationWeightEvolutionVisualizer,
    reset_bn_stats,
)

__all__ = [
    'plot_training_curves',
    'plot_pathway_importance',
    'plot_mixing_weights_evolution',
    'plot_loss_landscape',
    'visualize_pathway_activations',
    'plot_pathway_comparison',
    'analyze_pathway_contributions',
    'create_pathway_summary_plot',
    'scalar_weights',
    'channel_weights',
    'dynamic_weights',
    'spatial_weights',
    'FeatureMapVisualizer',
    'StreamContributionVisualizer',
    'StreamGradCAM',
    'IntegrationWeightVisualizer',
    'find_misclassified',
    'compare_samples',
    'StreamRedundancyAnalyzer',
    'PerClassDominanceAnalyzer',
    'ActivationDivergenceAnalyzer',
    'IntegrationWeightEvolutionVisualizer',
    'reset_bn_stats',
]
