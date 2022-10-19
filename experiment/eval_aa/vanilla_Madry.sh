

python evaluation_AA.py \
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
    --norm_layer vanilla \
    --test_pgd \
    --ckpt-dir log_files/pgd_default-baselineexperiment-Madry_loss/pgd_default-baselineexperiment-Madry_loss-Madry_loss--epochs_110-weight_decay_0.0005-lr_max_0.1-epsilon_0.06274509803921569-num_steps_10-step_size_0.00784313725490196-seed_0-/20220120051002_3vut6h0l/modle-epoch110.pt


