

python unified_network_zk_main.py \
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
    --out-dir baselineexperiment-trades_dual_bn \
    --norm_layer vanilla \
    --training_method trades_dual_bn \
    --dual_bn \
    --opt-level O0

