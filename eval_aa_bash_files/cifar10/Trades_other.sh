

BN_NAME="pgd"


CHECKPOINT_FILE="log_files/pgd_default-baselineexperiment-vanilla_trades_with_bn_none/pgd_default-baselineexperiment-vanilla_trades_with_bn_none-trades_vanilla-beta_6.0--epochs_110-weight_decay_0.0005-lr_max_0.1-epsilon_0.03137254901960784-num_steps_10-step_size_0.00784313725490196-seed_0-/20220125183348_2jor11lo/modle-epoch110.pt"
NORM_LAYER="none"

python evaluation_replace_bn_mean_std.py \
    --batch-size 128 \
    --num-steps 10 \
    --step-size 2 \
    --beta 6 \
    --seed 0 \
    --norm_layer $NORM_LAYER \
    --eval_base \
    --bn_name normal \
    --checkpoint_file $CHECKPOINT_FILE



CHECKPOINT_FILE="log_files/baselineexperiment-vanilla_trades_with_GN/baselineexperiment-vanilla_trades_with_GN-trades_vanilla-beta_6.0--epochs_110-weight_decay_0.0005-lr_max_0.1-epsilon_0.06274509803921569-num_steps_10-step_size_0.00784313725490196-seed_0-/20220119145934_3gaxcszo/modle-epoch110.pt"
NORM_LAYER="GN"

python evaluation_replace_bn_mean_std.py \
    --batch-size 128 \
    --num-steps 10 \
    --step-size 2 \
    --beta 6 \
    --seed 0 \
    --norm_layer $NORM_LAYER \
    --eval_base \
    --bn_name normal \
    --checkpoint_file $CHECKPOINT_FILE


CHECKPOINT_FILE="log_files/baselineexperiment-vanilla_trades_with_IN/baselineexperiment-vanilla_trades_with_IN-trades_vanilla-beta_6.0--epochs_110-weight_decay_0.0005-lr_max_0.1-epsilon_0.06274509803921569-num_steps_10-step_size_0.00784313725490196-seed_0-/20220119133030_1nr8zsm9/modle-epoch110.pt"
NORM_LAYER="IN"

python evaluation_replace_bn_mean_std.py \
    --batch-size 128 \
    --num-steps 10 \
    --step-size 2 \
    --beta 6 \
    --seed 0 \
    --norm_layer $NORM_LAYER \
    --eval_base \
    --bn_name normal \
    --checkpoint_file $CHECKPOINT_FILE
