

python unified_network_zk_main_bn_default_pgd.py \
    --batch-size 128 \
    --epochs 55 \
    --weight-decay 5e-4 \
    --lr 0.1 \
    --momentum 0.9 \
    --epsilon 8 \
    --num-steps 10 \
    --step-size 2 \
    --beta 6 \
    --seed 0 \
    --out-dir baselineexperiment-Madry_loss_remove_affine \
    --norm_layer remove_affine \
    --training_method Madry_loss \
    --opt-level O0 \
    --lr_decay fast55\
    --wandb_project  CrossTraining_eps8
