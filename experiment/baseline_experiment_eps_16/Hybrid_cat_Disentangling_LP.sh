

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
    --out-dir bug_classifier_number-Hybrid_cat_Disentangling_LP \
    --norm_layer Disentangling_LP \
    --training_method Hybrid_cat \
    --opt-level O0 \
    --dual_bn


