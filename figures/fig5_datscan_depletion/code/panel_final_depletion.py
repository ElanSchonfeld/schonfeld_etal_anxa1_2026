#!/usr/bin/env python3
"""Fig 5 panel - DaTscan PD-vs-HC group % dopamine depletion (axial + coronal MNI montage)."""
from fig5_common import FROZEN, PANELS
from slice_render import render_montage

render_montage(FROZEN / "slices_final_depletion.npz", PANELS / "final_depletion")
