#!/usr/bin/env python3
"""Fig 5 panel - DaTscan final % depletion by genotype (idiopathic / LRRK2 / GBA carriers), MNI montage."""
from fig5_common import FROZEN, PANELS
from slice_render import render_montage

render_montage(FROZEN / "slices_genetics_spatial.npz", PANELS / "genetics_spatial")
