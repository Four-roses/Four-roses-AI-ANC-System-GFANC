"""Project-wide configuration for band-energy-gated fixed-filter ANC."""

fs = 8000
control_band = (50, 300)
filter_len = 256
secondary_path_len = 128
num_filters = 8
frame_len = 256
block_size = 64
step_size = 0.03
regularization = 1e-6
leakage = 0.999
rho_adapt = 0.2
y_limit = 0.3
adapt_norm_limit = 0.5
adapt_tap_clip = 0.05
band_gate_top_k = 3
band_gate_threshold_ratio = 0.2
band_energy_smoothing = 0.8
band_gate_min_hold_frames = 2
normalize_selected_filters = True
w_gfanc_norm_limit = 1.0
simulation_duration_sec = 5.0
test_tones_hz = (80, 160, 240)
path_random_seed = 7
primary_path_len = 128
output_dir = "outputs"
checkpoint_dir = "checkpoints"

use_gfanc = False
use_fxnlms_refinement = True
