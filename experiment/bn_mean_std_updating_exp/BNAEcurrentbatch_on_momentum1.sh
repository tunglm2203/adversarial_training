python evaluation_replace_bn_mean_std.py \
    --batch-size 128 \
    --epochs 110 \
    --weight-decay 5e-4 \
    --lr 0.1 \
    --momentum 0.9 \
    --epsilon 8 \
    --num-steps 10 \
    --step-size 2 \
    --beta 6 \
    --seed 0 \
    --norm_layer BN2dStrongerAT \
    --learn_clean \
    --learn_adv \
    --eval_pgd \
    --bn_name pgd \
    --checkpoint_file log_files/pgd_default-baselineexperiment-Madry_loss_BN2dStrongerAT/pgd_default-baselineexperiment-Madry_loss_BN2dStrongerAT-Madry_loss--epochs_110-weight_decay_0.0005-lr_max_0.1-epsilon_0.03137254901960784-num_steps_10-step_size_0.00784313725490196-seed_0-/20220125165544_yorc3i2i/modle-epoch110.pt






