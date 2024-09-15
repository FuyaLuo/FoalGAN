
###############FoalGAN_FLIR
python train.py --name FoalGAN_FLIR --dataroot ./FLIR_datasets/ --n_domains 2 --niter 100 --niter_decay 0 --loadSize 288 --fineSize 256 --resize_or_crop crop --net_Gen_type gen_v1 --netS_start_epoch 20 --updateGT_start_epoch 30 --netS_end_epoch 75 --IR_prob_th 0.95 --lambda_sc 1.0 --lambda_sga 0.5 --lambda_CGR 1.0 --gpu_ids 4

