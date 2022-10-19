

NORM_LAYER="vanilla"
BN_NAME="pgd"


CHECKPOINT_FILE="
log_files/pgd_default-baselineexperiment-Hybrid_cat/pgd_default-baselineexperiment-Hybrid_cat-Hybrid_cat--epochs_110-weight_decay_0.0005-lr_max_0.1-epsilon_0.03137254901960784-num_steps_10-step_size_0.00784313725490196-seed_0-/20220124131802_1jbfqvxs/modle-epoch110.pt
log_files/pgd_default-baselineexperiment-Hybrid_cat/pgd_default-baselineexperiment-Hybrid_cat-Hybrid_cat--epochs_110-weight_decay_0.0005-lr_max_0.1-epsilon_0.06274509803921569-num_steps_10-step_size_0.00784313725490196-seed_0-/20220122114839_1ioxc6ka/modle-epoch110.pt
log_files/baselineexperiment-trades_cat/baselineexperiment-trades_cat-trades_cat--epochs_110-weight_decay_0.0005-lr_max_0.1-epsilon_0.06274509803921569-num_steps_10-step_size_0.00784313725490196-seed_0-/20220118083544_z4a403ri/modle-epoch110.pt
"

for ckpt in $CHECKPOINT_FILE; do
python evaluation_replace_bn_mean_std.py \
    --batch-size 128 \
    --num-steps 10 \
    --step-size 2 \
    --beta 6 \
    --seed 0 \
    --norm_layer $NORM_LAYER \
    --eval_base \
    --bn_name pgd \
    --checkpoint_file $ckpt
done


