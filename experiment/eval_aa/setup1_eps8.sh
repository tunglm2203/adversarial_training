

python evaluation_AA.py \
    --batch-size 512 \
    --epochs 110 \
    --weight-decay 5e-4 \
    --lr 0.1 \
    --momentum 0.9 \
    --epsilon 8 \
    --num-steps 10 \
    --step-size 2 \
    --beta 6 \
    --test_pgd \
    --norm_layer vanilla \
    --ckpt-dir log_files/pgd_default-baselineexperiment-Hybrid_single_bn/pgd_default-baselineexperiment-Hybrid_single_bn-Hybrid_single_bn--epochs_110-weight_decay_0.0005-lr_max_0.1-epsilon_0.03137254901960784-num_steps_10-step_size_0.00784313725490196-seed_0-/20220124211801_1rd5n0ho/modle-epoch110.pt
