

NORM_LAYER="vanilla"
BN_NAME="pgd"


CHECKPOINT_FILE="
log_files/pgd_default-baselineexperiment-Madry_dual_bn/pgd_default-baselineexperiment-Madry_dual_bn-Madry_dual_bn--epochs_110-weight_decay_0.0005-lr_max_0.1-epsilon_0.03137254901960784-num_steps_10-step_size_0.00784313725490196-seed_0-/20220125011500_m4s49wk4/modle-epoch110.pt
log_files/pgd_default-baselineexperiment-vanilla_trades_ae2gt/pgd_default-baselineexperiment-vanilla_trades_ae2gt-trades_ae2gt_vanilla--epochs_110-weight_decay_0.0005-lr_max_0.1-epsilon_0.06274509803921569-num_steps_10-step_size_0.00784313725490196-seed_0-/20220122082350_pdj96ymh/modle-epoch110.pt"

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


sh dual_bn.sh
