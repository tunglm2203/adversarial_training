

python unified_network_zk_main_bn_default_pgd.py \
    --dataset cifar10 \
    --dataset_dir /workspace/ssd2_4tb/ssd1/data/imagenette2 \
    --architecture resnet50 \
    --batch-size 128 \
    --epochs 110 \
    --weight-decay 5e-4 \
    --lr 0.1 \
    --momentum 0.9 \
    --epsilon 16 \
    --num-steps 4 \
    --step-size 4 \
    --beta 6 \
    --seed 0 \
    --out-dir resnet50-Hybrid_cat_Disentangling_LP \
    --norm_layer Disentangling_LP \
    --training_method Hybrid_cat \
    --opt-level O0 \
    --dual_bn



