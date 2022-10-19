
# # epsilon 8
# python BN_distribution_visualization/zcs_visualize_hybrid_dual_bn.py \
#     --batch-size 128 \
#     --epochs 100 \
#     --weight-decay 5e-4 \
#     --lr 0.1 \
#     --momentum 0.9 \
#     --epsilon 8 \
#     --num-steps 10 \
#     --step-size 2 \
#     --beta 6 \
#     --seed 0 \
#     --out-dir baselineexperiment-Madry_mixture_bn \
#     --norm_layer vanilla \
#     --resume_dir log_files/pgd_default-baselineexperiment-Hybrid_dual_bn/pgd_default-baselineexperiment-Hybrid_dual_bn-Madry_mixture_bn--epochs_110-weight_decay_0.0005-lr_max_0.1-epsilon_0.03137254901960784-num_steps_10-step_size_0.00784313725490196-seed_0-/20220124192050_1ugzmzhx/modle-epoch110.pt

# epsilon 16
python BN_distribution_visualization/zcs_visualize_hybrid_dual_bn.py \
    --batch-size 128 \
    --epochs 100 \
    --weight-decay 5e-4 \
    --lr 0.1 \
    --momentum 0.9 \
    --epsilon 16 \
    --num-steps 10 \
    --step-size 2 \
    --beta 6 \
    --seed 0 \
    --out-dir baselineexperiment-Madry_mixture_bn \
    --norm_layer vanilla \
    --resume_dir ../OCT/code/log_files/pgd_default-baselineexperiment-Hybrid_dual_bn_eps16/pgd_default-baselineexperiment-Hybrid_dual_bn_eps16-Madry_mixture_bn--epochs_110-weight_decay_0.0005-lr_max_0.1-epsilon_0.06274509803921569-num_steps_10-step_size_0.00784313725490196-seed_0-/20220927151021_dllgp4or/modle-epoch110.pt

