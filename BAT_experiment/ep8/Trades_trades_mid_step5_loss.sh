

python unified_network_zk_main_bn_default_pgd.py \
    --batch-size 128 \
    --epochs 110 \
    --weight-decay 5e-4 \
    --lr 0.1 \
    --momentum 0.9 \
    --epsilon 8 \
    --num-steps 10 \
    --step-size 2 \
    --beta 1 \
    --seed 0 \
    --out-dir trades_bat_loss-trades_mid_step5_loss \
    --norm_layer vanilla \
    --training_method trades_mid_step5_loss \
    --opt-level O0 \
    --lr_decay vanilla\
    --wandb_project  bridged_adversarial_training
