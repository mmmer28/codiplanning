_base_ = ["./univ2x_sub_inf_e2e.py"]

# Fine-tune only the infrastructure planning head on top of the official
# sub-inf stage-2 checkpoint. Perception / map / motion / occ stay frozen.
work_dir = "projects/work_dirs_e2e_univ2x/univ2x_sub_inf_planning_fulltrain"
load_from = "ckpts/univ2x_sub_inf_stg2.pth"

trainable_param_prefixes = ("model_ego_agent.planning_head.",)
custom_hooks = [
    dict(
        type="TrainableParamPrefixHook",
        trainable_param_prefixes=trainable_param_prefixes,
        priority="VERY_HIGH",
    ),
    dict(
        type="BestLatestPlanningCheckpointHook",
        interval=1,
        metric_keys=("planning.loss_ade",),
        priority="NORMAL",
    ),
]

optimizer = dict(type="AdamW", lr=5e-5, weight_decay=0.01)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(
    policy="CosineAnnealing",
    warmup="linear",
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3,
)

total_epochs = 20
runner = dict(type="EpochBasedRunner", max_epochs=total_epochs)
checkpoint_config = None
evaluation = dict(interval=2)
data = dict(workers_per_gpu=2)
