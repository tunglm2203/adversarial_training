BN_NAME="normal"


CHECKPOINT_FILE="log_files/baselineexperiment-trades_cat/baselineexperiment-trades_cat-trades_cat--epochs_110-weight_decay_0.0005-lr_max_0.1-epsilon_0.06274509803921569-num_steps_10-step_size_0.00784313725490196-seed_0-/20220118083544_z4a403ri/modle-epoch110.pt"
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
    --checkpoint_file $CHECKPOINT_FILE
