

NORM_LAYER="Disentangling_StatP"


CHECKPOINT_FILE="
log_files/pgd_default-baselineexperiment-Hybrid_Disentangling_StatP/pgd_default-baselineexperiment-Hybrid_Disentangling_StatP-Madry_mixture_bn--epochs_110-weight_decay_0.0005-lr_max_0.1-epsilon_0.06274509803921569-num_steps_10-step_size_0.00784313725490196-seed_0-/20220126105905_185t28pk/modle-epoch110.pt
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
    --bn_name normal \
    --dual_bn \
    --checkpoint_file $ckpt
    
done


