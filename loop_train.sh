export CUDA_VISIBLE_DEVICES=2
for FOLD in $(seq 0 4)
do
    python train.py --stage='train'\
    --config='yaml/tcga_lung_subtype/CAMIL.yaml'  --fold=$FOLD
    python train.py --stage='test'\
    --config='yaml/tcga_lung_subtype/CAMIL.yaml'  --fold=$FOLD
done
