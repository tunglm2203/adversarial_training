

python unified_network_zk_main_bn_default_pgd.py \
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
    --out-dir baselineexperiment-Madry_dual_bn_fix_cnn_clean_bn_lr1e-2 \
    --norm_layer vanilla \
    --training_method Madry_dual_bn_fix_cnn_clean_bn \
    --dual_bn \
    --opt-level O0 \
    --lr_decay vanilla\
    --wandb_project  CrossTraining_eps8_epoch110 \
    --lr_clean 0.01

