export CUDA_VISIBLE_DEVICES=2
python train.py --stage='train' --config='Camelyon/CAMIL.yaml'  --fold=0
python train.py --stage='test' --config='Camelyon/CAMIL.yaml'  --fold=0
