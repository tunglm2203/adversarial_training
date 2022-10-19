

python unified_network_zk_main_bn_default_pgd.py \
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
    --out-dir baselineexperiment-Finetune_BN_on_clean \
    --norm_layer vanilla \
    --training_method Finetune_BN_on_clean \
    --opt-level O0 \
    --wandb_project  CrossTraining_eps8_epoch110 \
    --resume \
    --resume_dir 'log_files/BAT-baselineexperiment-Madry_loss/BAT-baselineexperiment-Madry_loss-Madry_loss--epochs_110-weight_decay_0.0005-lr_max_0.1-epsilon_0.03137254901960784-num_steps_10-step_size_0.00784313725490196-seed_0-/20220917140925_1m8qx5ip/modle-epoch110.pt'


