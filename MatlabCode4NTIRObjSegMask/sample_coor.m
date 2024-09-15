function [pos_h, pos_w] = sample_coor(input_pos_h, input_pos_w, patch_width)

img_h = 288;
img_w = 360;
mask_patch = zeros(img_h, img_w);

norm_h = input_pos_h - 0.5 * patch_width + 1;
norm_w = input_pos_w - 0.5 * patch_width + 1;
for j = 1:length(input_pos_h)
    if (norm_h(j) > 0) && (norm_w(j) > 0)
        new_h = norm_h(j);
        new_w = norm_w(j);
        break
    end
end

mask_patch(new_h:(new_h + patch_width - 1), new_w:(new_w + patch_width - 1)) = ones(patch_width, patch_width);
% figure(2), imshow(mask_patch, []);
filter_mask_256 = ones(256, 256);
corr_256_res = conv2(mask_patch, filter_mask_256);
corr_256_new = corr_256_res(128:415, 128:487);
corr_256_res_max = max(max(corr_256_new));

[h_256_list, w_256_list] = find(corr_256_new == corr_256_res_max);
h_new_list = h_256_list - 128;
w_new_list = w_256_list - 128;

for i = 1:length(h_new_list)
    if (h_new_list(i) > 0) && (h_new_list(i) < 34) && (w_new_list(i) > 0) && (w_new_list(i) < 106)
        pos_h_list = h_new_list(i);
        pos_w_list = w_new_list(i);
    end
end


if length(pos_h_list) > 1
    pos_h = pos_h_list(1);
    pos_w = pos_w_list(1);
else
    pos_h = pos_h_list;
    pos_w = pos_w_list;
end


