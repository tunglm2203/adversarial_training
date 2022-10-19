

CHECKPOINT_FILE="log_files/pgd_default-baselineexperiment-Hybrid_dual_bn/pgd_default-baselineexperiment-Hybrid_dual_bn-Madry_mixture_bn--epochs_110-weight_decay_0.0005-lr_max_0.1-epsilon_0.03137254901960784-num_steps_10-step_size_0.00784313725490196-seed_0-/20220124192050_1ugzmzhx/modle-epoch110.pt"
NORM_LAYER="vanilla"

python evaluation_replace_bn_mean_std.py \
    --batch-size 128 \
    --num-steps 10 \
    --step-size 2 \
    --beta 6 \
    --seed 0 \
    --norm_layer $NORM_LAYER \
    --eval_base \
    --bn_name normal \
    --dual_bn \
    --swap_dual_bn \
    --checkpoint_file $CHECKPOINT_FILE

