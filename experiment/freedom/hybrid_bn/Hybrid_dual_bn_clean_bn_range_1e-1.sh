
clean_bn_range=1e-1
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
    --out-dir baselineexperiment-Hybrid_dual_bn-clean_bn_range_$clean_bn_range \
    --norm_layer vanilla \
    --training_method Madry_mixture_bn \
    --dual_bn \
    --opt-level O0 \
    --lr_decay vanilla\
    --wandb_project  CrossTraining_eps8_epoch110 \
    --clean_bn_range $clean_bn_range


