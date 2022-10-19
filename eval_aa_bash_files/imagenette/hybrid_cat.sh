

NORM_LAYER="vanilla"


CHECKPOINT_FILE="
log_files/BAT-Imagenette-Hybrid_cat/BAT-Imagenette-Hybrid_cat-Hybrid_cat--epochs_110-weight_decay_0.0005-lr_max_0.1-epsilon_0.06274509803921569-num_steps_10-step_size_0.00784313725490196-seed_0-/20220410144702_18t2yt5i/modle-epoch110.pt"


for ckpt in $CHECKPOINT_FILE; do
python evaluation_replace_bn_mean_std.py \
    --dataset imagenette \
    --dataset_dir /workspace/ssd2_4tb/ssd1/data/imagenette2 \
    --batch-size 128 \
    --num-steps 10 \
    --step-size 2 \
    --beta 6 \
    --seed 0 \
    --norm_layer $NORM_LAYER \
    --eval_base \
    --bn_name pgd \
    --dual_bn \
    --checkpoint_file $ckpt
    
done


