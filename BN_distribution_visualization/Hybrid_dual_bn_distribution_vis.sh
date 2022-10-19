python BN_distribution_visualization/BN_distribution_visualization.py \
    --batch-size 128 \
    --epochs 110 \
    --weight-decay 5e-4 \
    --lr 0.1 \
    --momentum 0.9 \
    --epsilon 16 \
    --num-steps 10 \
    --step-size 2 \
    --beta 6 \
    --seed 0 \
    --out-dir baselineexperiment-trades_dual_bn \
    --norm_layer vanilla \
    --training_method trades_dual_bn \
    --dual_bn \
    --opt-level O0 \
    --resume_dir log_files/pgd_default-baselineexperiment-Hybrid_dual_bn/pgd_default-baselineexperiment-Hybrid_dual_bn-Madry_mixture_bn--epochs_110-weight_decay_0.0005-lr_max_0.1-epsilon_0.03137254901960784-num_steps_10-step_size_0.00784313725490196-seed_0-/20220124192050_1ugzmzhx/modle-epoch110.pt

