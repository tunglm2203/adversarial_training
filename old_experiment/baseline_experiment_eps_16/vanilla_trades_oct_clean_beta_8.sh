

python unified_network_zk_main.py \
    --batch-size 128 \
    --epochs 110 \
    --weight-decay 5e-4 \
    --lr 0.1 \
    --momentum 0.9 \
    --epsilon 16 \
    --num-steps 10 \
    --step-size 2 \
    --beta 8 \
    --seed 0 \
    --out-dir baselineexperiment-vanilla_trades_oct_clean-beta8 \
    --norm_layer oct_clean \
    --training_method trades_cat \
    --opt-level O0

