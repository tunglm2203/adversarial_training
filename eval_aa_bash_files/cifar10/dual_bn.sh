

NORM_LAYER="vanilla"
BN_NAME="pgd"


CHECKPOINT_FILE="
log_files/pgd_default-baselineexperiment-Madry_mixture_bn/pgd_default-baselineexperiment-Madry_mixture_bn-Madry_mixture_bn--epochs_110-weight_decay_0.0005-lr_max_0.1-epsilon_0.06274509803921569-num_steps_10-step_size_0.00784313725490196-seed_0-/20220120063559_1cmh754s/modle-epoch110.pt
log_files/baselineexperiment-trades_dual_bn/baselineexperiment-trades_dual_bn-trades_dual_bn--epochs_110-weight_decay_0.0005-lr_max_0.1-epsilon_0.06274509803921569-num_steps_10-step_size_0.00784313725490196-seed_0-/20220118083745_5na0tv7p/modle-epoch110.pt
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
    --dual_bn \
    --checkpoint_file $ckpt
done


