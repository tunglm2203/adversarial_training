

NORM_LAYER="Disentangling_LP"
BN_NAME="pgd"


CHECKPOINT_FILE="
log_files/pgd_default-baselineexperiment-Hybrid_cat_Disentangling_LP/pgd_default-baselineexperiment-Hybrid_cat_Disentangling_LP-Hybrid_cat--epochs_110-weight_decay_0.0005-lr_max_0.1-epsilon_0.03137254901960784-num_steps_10-step_size_0.00784313725490196-seed_0-/20220124144241_rjziaybx/modle-epoch110.pt
log_files/pgd_default-baselineexperiment-Hybrid_cat_Disentangling_LP/pgd_default-baselineexperiment-Hybrid_cat_Disentangling_LP-Hybrid_cat--epochs_110-weight_decay_0.0005-lr_max_0.1-epsilon_0.06274509803921569-num_steps_10-step_size_0.00784313725490196-seed_0-/20220123074608_39tn1ldb/modle-epoch110.pt
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


