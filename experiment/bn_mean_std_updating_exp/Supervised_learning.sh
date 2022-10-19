
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
    --norm_layer vanilla \
    --learn_clean \
    --learn_adv \
    --eval_pgd \
    --eval_aa \
    --eval_base \
    --checkpoint_file log_files/pgd_default-baselineexperiment-Supervised_training/pgd_default-baselineexperiment-Supervised_training-Finetune_BN_on_clean--epochs_110-weight_decay_0.0005-lr_max_0.1-epsilon_0.06274509803921569-num_steps_10-step_size_0.00784313725490196-seed_0-/20220121032445_37y9hmna/modle-epoch110.pt
    # --checkpoint_file log_files/BAT-baselineexperiment-Supervised_training/BAT-baselineexperiment-Supervised_training-Finetune_BN_on_clean--epochs_110-weight_decay_0.0005-lr_max_0.1-epsilon_0.06274509803921569-num_steps_10-step_size_0.00784313725490196-seed_0-/20220924150313_3dcwo3i3/modle-epoch110.pt 
    # --checkpoint_file log_files/pgd_default-baselineexperiment-Madry_loss/pgd_default-baselineexperiment-Madry_loss-Madry_loss--epochs_110-weight_decay_0.0005-lr_max_0.1-epsilon_0.03137254901960784-num_steps_10-step_size_0.00784313725490196-seed_0-/20220124131721_23n8qspd/modle-epoch110.pt


