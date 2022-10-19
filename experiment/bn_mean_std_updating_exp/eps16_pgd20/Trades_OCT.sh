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
    --learn_clean \
    --eval_aa \
    --eval_pgd \
    --test-num-steps 20 \
    --bn_name normal \
    --checkpoint_file log_files/baselineexperiment-vanilla_trades_oct_clean/baselineexperiment-vanilla_trades_oct_clean-trades_cat--epochs_110-weight_decay_0.0005-lr_max_0.1-epsilon_0.06274509803921569-num_steps_10-step_size_0.00784313725490196-seed_0-/20220118083322_ppi5t5bt/modle-epoch110.pt



    