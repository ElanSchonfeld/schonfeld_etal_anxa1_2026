#!/usr/bin/env python3
"""Figure 5 supplementary panel: DaTscan % depletion by time since motor onset, MNI montage."""
from fig5_common import FROZEN, PANELS
from slice_render import render_montage

render_montage(FROZEN / "slices_staging_time.npz", PANELS / "staging_time")
