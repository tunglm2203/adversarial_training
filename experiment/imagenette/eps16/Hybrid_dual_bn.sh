

python unified_network_zk_main_bn_default_pgd.py \
    --dataset imagenette \
    --dataset_dir /workspace/ssd2_4tb/ssd1/data/imagenette2 \
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
    --out-dir Imagenette_bug_classifier1000-Hybrid_dual_bn \
    --norm_layer vanilla \
    --training_method Madry_mixture_bn \
    --opt-level O0 \
    --dual_bn

