_base_ = ["./univ2x_coop_e2e_diffusion_plan_fulltrain.py"]

# Evaluation-only overrides (no training / resume).
load_from = None
resume_from = None
custom_hooks = []
checkpoint_config = None

# Val split is used as data.test in base config.
data = dict(workers_per_gpu=2)
