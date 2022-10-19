

NORM_LAYER="vanilla"

BN_NAME="pgd"


CHECKPOINT_FILE="
log_files/baselineexperiment-Madry_cat_loss_oct_clean/baselineexperiment-Madry_cat_loss_oct_clean-Madry_cat_loss--epochs_110-weight_decay_0.0005-lr_max_0.1-epsilon_0.06274509803921569-num_steps_10-step_size_0.00784313725490196-seed_0-/20220118083649_1s46lxy8/modle-epoch110.pt
"

for ckpt in $CHECKPOINT_FILE; do
python evaluation_replace_bn_mean_std.py \
    --batch-size 128 \
    --num-steps 10 \
    --step-size 2 \
    --beta 6 \
    --seed 0 \
    --norm_layer $NORM_LAYER \
    --learn_clean \
    --bn_name normal \
    --checkpoint_file $ckpt
 
done

