python evaluation_replace_bn_mean_std.py \
    --batch-size 512 \
    --epochs 110 \
    --weight-decay 5e-4 \
    --lr 0.1 \
    --momentum 0.9 \
    --epsilon 16 \
    --num-steps 10 \
    --step-size 2 \
    --beta 6 \
    --seed 0 \
    --norm_layer oct_clean \
    --eval_base \
    --eval_aa \
    --test-num-steps 20 \
    --bn_name pgd \
    --checkpoint_file log_files/baselineexperiment-vanilla_trades_oct_adv/baselineexperiment-vanilla_trades_oct_adv-trades_cat--epochs_110-weight_decay_0.0005-lr_max_0.1-epsilon_0.06274509803921569-num_steps_10-step_size_0.00784313725490196-seed_0-/20220119121508_76m5l56h/modle-epoch110.pt


