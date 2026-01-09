"""
Voice Filters Module for Chatterbox
Cat & Daniel: Collapse Protocol - Ghost Voice Processing
"""

from .ghost_filters import (
    apply_ghost_filter,
    process_file,
    get_mode_names,
    get_mode_descriptions,
    create_gradio_filter_component,
    GHOST_MODES,
)

__all__ = [
    "apply_ghost_filter",
    "process_file",
    "get_mode_names",
    "get_mode_descriptions",
    "create_gradio_filter_component",
    "GHOST_MODES",
]
