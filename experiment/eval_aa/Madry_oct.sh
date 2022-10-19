

python evaluation_AA.py \
    --batch-size 512 \
    --epochs 110 \
    --weight-decay 5e-4 \
    --lr 0.1 \
    --momentum 0.9 \
    --epsilon 16 \
    --num-steps 10 \
    --step-size 2 \
    --beta 6 \
    --test_normal \
    --norm_layer oct_clean \
    --ckpt-dir log_files/baselineexperiment-Madry_cat_loss_oct_clean/baselineexperiment-Madry_cat_loss_oct_clean-Madry_cat_loss--epochs_110-weight_decay_0.0005-lr_max_0.1-epsilon_0.06274509803921569-num_steps_10-step_size_0.00784313725490196-seed_0-/20220118083649_1s46lxy8/modle-epoch110.pt
