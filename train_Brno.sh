

###############FoalGAN_Brno
python train.py --name FoalGAN_Brno --dataroot ./Brno_datasets/ --n_domains 2 --niter 160 --niter_decay 0 --loadSize 288 \
--fineSize 256 --resize_or_crop crop --net_Gen_type gen_v1 --IR_edge_path ./Brno_IR_edge_map/ \
--Vis_edge_path ./Brno_Vis_edge_map/ --Vis_mask_path ./Brno_Vis_seg_mask/ --IR_mask_path ./Brno_IR_seg_mask/ \
--IR_FG_txt ./Brno_txt_file/IR_FG_list.txt --Vis_FG_txt ./Brno_txt_file/Vis_FG_list.txt --FB_Sample_Vis_txt \
./Brno_txt_file/FB_Sample_Vis.txt --FB_Sample_IR_txt ./Brno_txt_file/FB_Sample_IR.txt --IR_patch_classratio_txt \
./Brno_txt_file/IR_patch_classratio.txt --netS_start_epoch 50 --updateGT_start_epoch 60 --netS_end_epoch 110 --IR_prob_th 0.95 \
--lambda_sc 1.0 --lambda_sga 0.5 --lambda_CGR 1.0 --gpu_ids 0

