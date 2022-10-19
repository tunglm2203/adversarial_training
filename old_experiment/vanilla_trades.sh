

python unified_network_zk_main.py \
    --batch-size 128 \
    --epochs 76 \
    --weight-decay 2e-4 \
    --lr 0.1 \
    --momentum 0.9 \
    --epsilon 8 \
    --num-steps 10 \
    --step-size 2 \
    --beta 6 \
    --seed 0 \
    --out-dir test_vanilla_trades_amp \
    --norm_layer vanilla \
    --training_method trades_vanilla


# python zk_main.py \
#     --batch-size 128 \
#     --epochs 76 \
#     --weight-decay 2e-4 \
#     --lr 0.1 \
#     --momentum 0.9 \
#     --epsilon 8 \
#     --num-steps 10 \
#     --step-size 2 \
#     --beta 6 \
#     --seed 0 \
#     --out-dir test_vanilla_trades_amp \
#     --model resnet18 \
#     --training_method trades_vanilla
