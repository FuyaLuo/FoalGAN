clear all;
clc;

%%%%%%%%Brno
Cityscape_Mask_folder = 'Folder of segmentation masks predicted by the Cityscape dataset';
Mapillary_Mask_folder = 'Folder of segmentation masks predicted by the Mapillary dataset';
des_path_original = 'Folder for saving the fused object masks';

dir_file = dir(fullfile(Mapillary_Mask_folder, '*.png'));
file_names = {dir_file.name};

if ~exist(des_path_original, 'dir'), mkdir(des_path_original);end
% Street Light index in Mapillary vistas dataset:44
for i = 1:length(file_names)
    original_filename = file_names{1, i};
    new_imgname = strrep(original_filename, '_fake_0.png', '.png');
    CS_Mask_file = [Cityscape_Mask_folder, original_filename];%%%%%%%Brno
    MVD_Mask_file = [Mapillary_Mask_folder, original_filename];

    CS_Mask = double(imread(CS_Mask_file));
    MVD_Mask = double(imread(MVD_Mask_file));
    mask_out_person = ObjMaskOverlap(CS_Mask, [11 12], MVD_Mask, [19:22], 1);
    mask_out_car = ObjMaskOverlap(CS_Mask, 13, MVD_Mask, 55, 0);
    mask_out_truck = ObjMaskOverlap(CS_Mask, 14, MVD_Mask, 61, 0);
    mask_out_bus = ObjMaskOverlap(CS_Mask, 15, MVD_Mask, 54, 0);
    mask_out_train = ObjMaskOverlap(CS_Mask, 16, MVD_Mask, 58, 0);
    mask_out_motorcycle = ObjMaskOverlap(CS_Mask, 17, MVD_Mask, 57, 0);
    mask_out_bicycle = ObjMaskOverlap(CS_Mask, 18, MVD_Mask, 52, 0);
    
    mask_StreetLight_MVD = zeros(size(CS_Mask));
    mask_TrafficLight_MVD = zeros(size(CS_Mask));
    mask_TSFront_MVD = zeros(size(CS_Mask));
    mask_TSback_MVD = zeros(size(CS_Mask));
    mask_StreetLight_MVD(MVD_Mask == 44) = 1;
    mask_TrafficLight_MVD(MVD_Mask == 48) = 1;
    mask_TSFront_MVD(MVD_Mask == 49) = 1;
    mask_TSback_MVD(MVD_Mask == 50) = 1;
    
    %%%%%%%%No Car to reduce noisy region.            
    mask_obj_indicate = mask_out_person + mask_out_truck + mask_out_bus + ...
                    mask_out_train + mask_out_motorcycle + mask_out_bicycle + ...
                    mask_StreetLight_MVD + mask_TrafficLight_MVD + mask_TSFront_MVD + ...
                    mask_TSback_MVD;
                
    mask_obj_all = mask_out_person * 11 + mask_out_truck * 14 + ...
                    mask_out_bus * 15 + mask_out_train * 16 + mask_out_motorcycle * 17 + ...
                    mask_out_bicycle * 18 + mask_StreetLight_MVD * 12 + mask_TrafficLight_MVD * 6 + ...
                    (mask_TSFront_MVD + mask_TSback_MVD) * 7;
    
    mask_CS_fuse_MVD_TO = (ones(size(CS_Mask)) - mask_obj_indicate) * 255 + mask_obj_all;
    img_output = fullfile(des_path_original, new_imgname);
    imwrite(uint8(mask_CS_fuse_MVD_TO), img_output, 'png');
end
