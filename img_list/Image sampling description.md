
# Image sampling

Based on the correspondence between thermal infrared images and RGB images, we manually divide the dataset into daytime and nighttime parts according to the lighting conditions of RGB images.

## FLIR
After removing some noisy images (e.g., all-white images), we finally obtained 5447 daytime color (DC) images and 2899 nighttime thermal infrared (NTIR) images for the training of the NTIR2DC task. Similarly, we were able to collect 490 NTIR images in the validation set for model evaluation.

## Brno
The Brno Urban Dataset was collected from multiple sensors in the city of Brno for the study of autonomous driving tasks. Sixty-seven video clips are included in the entire dataset. Three videos (`set2_1_4_1`, `set2_1_5_1` and `set2_1_8_1`) of sunny days in the DC domain, and two videos (`set2_1_10_1` and `set2_1_10_2`) of cloudy days and one video (`set3_1_4_1`) of rainy days in the NTIR domain were selected as training data. An additional cloudy video (`set2_1_10_3`) and a rainy video (`set3_1_4_2`) in the NTIR domain were selected as test data. To eliminate redundant data at adjacent times, we re-sampled the videos at one frame per second, and finally obtained 2574 DC images and 2383 NTIR images for model training. Similarly, we re-sampled the tested NTIR videos at an interval of 60 frames, and finally selected 500 NTIR images randomly as the test set.

