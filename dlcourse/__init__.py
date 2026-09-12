"""Shared helpers for the Deep Learning course.

Every notebook starts with:

    import sys, pathlib
    sys.path.append(str(pathlib.Path.cwd().parents[2]))   # repo root
    from dlcourse import load_shapes, train, evaluate, show_grid

Keeping the boilerplate here means each notebook shows the idea being taught,
not forty lines of plumbing.
"""
from .data import (
    DATA_DIR,
    SHAPE_CLASSES,
    load_shapes,
    load_digits_npz,
    load_detection,
    load_segmentation,
    load_ssl_pool,
    load_sentiment,
    load_corpus,
    load_captions,
    ArrayImageDataset,
)
from .training import train, evaluate, History, set_seed, count_parameters
from .viz import show_grid, plot_history, plot_confusion, show_boxes, show_masks

__all__ = [
    "DATA_DIR", "SHAPE_CLASSES",
    "load_shapes", "load_digits_npz", "load_detection", "load_segmentation",
    "load_ssl_pool", "load_sentiment", "load_corpus", "load_captions",
    "ArrayImageDataset",
    "train", "evaluate", "History", "set_seed", "count_parameters",
    "show_grid", "plot_history", "plot_confusion", "show_boxes", "show_masks",
]
