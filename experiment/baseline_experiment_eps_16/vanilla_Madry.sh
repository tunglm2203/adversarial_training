

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
    --out-dir clean_adv_distribution-Madry_loss \
    --norm_layer vanilla \
    --training_method Madry_loss \
    --opt-level O0

