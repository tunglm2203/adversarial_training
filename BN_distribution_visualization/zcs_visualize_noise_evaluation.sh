python BN_distribution_visualization/zcs_visualize_noise_evaluation.py \
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
    --noise_size 1.0 \
    --resume_dir ../OCT/code/log_files/pgd_default-baselineexperiment-Clean_loss/pgd_default-baselineexperiment-Clean_loss-Clean_loss--epochs_110-weight_decay_0.0005-lr_max_0.1-epsilon_0.03137254901960784-num_steps_10-step_size_0.00784313725490196-seed_0-/20220927142405_26u2t7xx/modle-epoch110.pt


