clear all;
clc;


Seg_Mask_folder = 'Folder for saving the fused and denoised segmentation masks';
des_txt_file = 'Path to txt file';

dir_file = dir(fullfile(Seg_Mask_folder, '*.png'));
file_names = {dir_file.name};

fid = fopen(des_txt_file, 'a+');
for i = 1:length(file_names)
    original_filename = file_names{1, i};
    Seg_Mask_file = [Seg_Mask_folder, original_filename];
    Seg_Mask = double(imread(Seg_Mask_file));
    FG_mask = zeros(size(Seg_Mask));
    FG_mask(Seg_Mask == 6) = 1;
    FG_mask(Seg_Mask == 7) = 1;
    FG_mask(Seg_Mask == 17) = 1;
    
    filter_mask_256 = ones(256,256);
    corr_256_res = conv2(FG_mask, filter_mask_256, 'same');
    
    corr_256_res_max = max(max(corr_256_res));
    [h_256, w_256] = find(corr_256_res == corr_256_res_max);
    if corr_256_res_max > 512
        [pos_h, pos_w] = sample_coor(h_256, w_256, 256);
        %%%Saving the name of the images containing the SOC and the coordinates of the upper left corner of the maximum ROI.
        fprintf(fid, '%s %d %d\n', original_filename, pos_h, pos_w);
    end
    
end
fclose(fid);