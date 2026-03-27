"""Visualization module for cyclone trajectory and intensity prediction."""

from src.visualization.trajectory_plots import (
    plot_storm_track_comparison,
    predict_trajectory_classification,
    predict_trajectory_regression,
    plot_multi_storm_grid,
    plot_error_along_track,
    generate_test_visualizations,
)

from src.visualization.intensity_plots import (
    plot_intensity_evolution,
    plot_intensity_change_comparison,
    plot_decadal_intensity_stats,
    create_storm_animation,
    create_intensity_comparison_animation,
    generate_intensity_visualizations,
)

__all__ = [
    # Trajectory
    "plot_storm_track_comparison",
    "predict_trajectory_classification",
    "predict_trajectory_regression",
    "plot_multi_storm_grid",
    "plot_error_along_track",
    "generate_test_visualizations",
    # Intensity
    "plot_intensity_evolution",
    "plot_intensity_change_comparison",
    "plot_decadal_intensity_stats",
    "create_storm_animation",
    "create_intensity_comparison_animation",
    "generate_intensity_visualizations",
]
