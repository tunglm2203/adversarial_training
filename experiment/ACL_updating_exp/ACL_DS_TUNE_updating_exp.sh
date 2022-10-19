python evaluation_replace_bn_mean_std.py \
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
    --norm_layer vanilla \
    --learn_clean \
    --learn_adv \
    --eval_pgd \
    --bn_name pgd \
    --ACL \
    --checkpoint_file experiment/ACL_updating_exp/ACL_DS_TUNE.pt




