# Nighttime Thermal Infrared Image Colorization With Feedback-Based Object Appearance Learning
Pytorch implementation of the paper "Nighttime Thermal Infrared Image Colorization With Feedback-Based Object Appearance Learning".

![tease](https://github.com/FuyaLuo/FoalGAN/blob/main/docs/Model.PNG)

### [Paper](https://ieeexplore.ieee.org/abstract/document/10313314)

## Abstract
>Stable imaging in adverse environments (e.g., total darkness) makes thermal infrared (TIR) cameras a prevalent option for night scene perception. However, the low contrast and lack of chromaticity of TIR images are detrimental to human interpretation and subsequent deployment of RGB-based vision algorithms. Therefore, it makes sense to colorize the nighttime TIR images by translating them into the corresponding daytime color images (NTIR2DC). Despite the impressive progress made in the NTIR2DC task, how to improve the translation performance of small object classes is under-explored. To address this problem, we propose a generative adversarial network incorporating feedback-based object appearance learning (FoalGAN). Specifically, an occlusion-aware mixup module and corresponding appearance consistency loss are proposed to reduce the context dependence of object translation. As a representative example of small objects in nighttime street scenes, we illustrate how to enhance the realism of traffic light by designing a traffic light appearance loss. To further improve the appearance learning of small objects, we devise a dual feedback learning strategy to selectively adjust the learning frequency of different samples. In addition, we provide pixel-level annotation for a subset of the Brno dataset, which can facilitate the research of NTIR image understanding under multiple weather conditions. Extensive experiments illustrate that the proposed FoalGAN is not only effective for appearance learning of small objects, but also outperforms other image translation methods in terms of semantic preservation and edge consistency for the NTIR2DC task. Compared with the state-of-the-art NTIR2DC approach, FoalGAN achieves at least 5.4% improvement in semantic consistency and at least 2% lead in edge consistency. 

## Prerequisites
* Python 3.6 
* Pytorch 1.1.0 and torchvision 0.3.0 
* TensorboardX
* visdom
* dominate
* pytorch-msssim
* kmeans_pytorch
* CUDA 10.0.130, CuDNN 7.3, and Ubuntu 16.04.

## Data Preparation 
Download [FLIR](https://www.flir.co.uk/oem/adas/adas-dataset-form/) and [Brno](https://github.com/Robotics-BUT/Brno-Urban-Dataset). First, the corresponding training set and test set images are sampled according to the txt files in the `./img_list/` folder. Then, all images are first resized to 500x400, and then crop centrally to obtain images with a resolution of 360x288. Finally, place all images into the corresponding dataset folders. Domain A and domain B correspond to the daytime visible image and the nighttime TIR image, respectively. As an example, the corresponding folder structure for the FLIR dataset is:
 ```
mkdir FLIR_datasets
# The directory structure should be this:
FLIR_datasets
   ├── trainA (daytime RGB images)
       ├── FLIR_00002.png 
       └── ...
   ├── trainB (nighttime IR images)
       ├── FLIR_00135.png
       └── ...
   ├── testA (testing daytime RGB images)
       ├── FLIR_09112.png (The test image that you want)
       └── ... 
   ├── testB (testing nighttime IR images)
       ├── FLIR_08872.png (The test image that you want)
       └── ... 

mkdir FLIR_testsets
# The directory structure should be this:
FLIR_testsets
   ├── test0 (empty folder)
   ├── test1 (testing nighttime IR images)
       ├── FLIR_08863.png
       └── ...
```

We predict the edge maps of Nighttime TIR images and daytime color images using [MCI](https://drive.google.com/file/d/1Qf2wIyzr0J8nWSuc8d6bHyO2Mxzeuamv/view?usp=sharing) method and Canny edge detection method, respectively. Next, place all edge maps into the corresponding folders(e.g., `/FLIR_IR_edge_map/` and `/FLIR_Vis_edge_map/` for FLIR dataset).

For segmentation mask prediction of DC images, we first utilize [HMSANet](https://github.com/segcv/hierarchical-multi-scale-attention) and [Detectron2](https://github.com/facebookresearch/detectron2) models to obtain the initial mask. Then, the masks obtained from the predictions of the two models are fused and semantic denoising is performed, which can be realized by running the code `MaskFusedDenoised_demo.m`. You will need to modify the four paths in the code to suit your situation. The final masks for the DC images used for training of the FLIR and Brno datasets can be downloaded via [google drive](https://drive.google.com/file/d/1WQy8UZ1OfSwTP6Kp0MxuwCtla7aIbdi5/view?usp=sharing). Next, place all segmentation masks into the corresponding folders(i.e., `/FLIR_Vis_seg_mask/` and `/Brno_Vis_seg_mask/` for the FLIR and Brno dataset, respectively).

For mask prediction of object categories of NTIR images, we first introduce bias correction loss based on the MornGAN model, and the training process is consistent with the [MornGAN](https://github.com/FuyaLuo/MornGAN). Then, the weights obtained from the training are used to translate the NTIR images from the training sets into DC images. Afterwards, the segmentation masks of the translated DC images are predicted using the segmentation models (i.e., HMSANet) trained on the Cityscape and Mapillary datasets, respectively. Given that the automobile region is larger, the corresponding incorrectly predicted region will be more compared to other object categories. Therefore, we finally fused the masks predicted by the two segmentation models to extract masks for object categories other than the car category, and the MATLAB code for the mask fusion is the `mask_trainIR2Vis_obj.m` file in the directory `MatlabCode4NTIRObjSegMask`. The final masks can be downloaded via [google drive](https://drive.google.com/file/d/1WQy8UZ1OfSwTP6Kp0MxuwCtla7aIbdi5/view?usp=sharing).

After obtaining the semantic masks for the two domains in the training set, we can determine the image containing the SOC, and the maximum cropping region containing the SOC, based on the category index. After that, the corresponding image names with the upper left corner of the cropped region are written into a txt file, and the corresponding MATLAB code is the `Sample_SSC_patch.m` file in the directory `MatlabCode4NTIRObjSegMask`. The final txt files are located in the `Vis_FG_list.txt` and `IR_FG_list.txt` files under the directory `FLIR_txt_file` and the directory `Brno_txt_file`.

## Inference Using Pretrained Model

<details>
  <summary>
    <b>1) FLIR</b>
  </summary>
  
Download and unzip the [pretrained model](https://drive.google.com/file/d/1iRrP6wvaxvSR_6u2bMDT2MuhzSAh6qyC/view?usp=sharing) and save it in `./checkpoints/FoalGAN_FLIR/`. Place the test images of the FLIR dataset in `./FLIR_testsets/test1/`. Then run the command 
```bash
python test_output_only.py --phase test --serial_test --name FoalGAN_FLIR --dataroot ./FLIR_testsets/ --n_domains 2 --which_epoch 100 --results_dir ./res_FLIR/ --loadSize 288 --net_Gen_type gen_v1 --no_flip --gpu_ids 0
```
</details>

<details>
  <summary>
    <b>2) Brno</b>
  </summary>
  
Download and unzip the [pretrained model](https://drive.google.com/file/d/1VpqevcjQ7uXw6_hAiHjdDoG13jRVyCvW/view?usp=sharing) and save it in `./checkpoints/FoalGAN_Brno/`. Place the test images of the FLIR dataset in `./Brno_testsets/test1/`. Then run the command 
```bash
python test_output_only.py --phase test --serial_test --name FoalGAN_Brno --dataroot ./Brno_testsets/ --n_domains 2 --which_epoch 160 --results_dir ./res_Brno/ --loadSize 288 --net_Gen_type gen_v1 --no_flip --gpu_ids 0
```
</details>

## Training

To reproduce the performance, we recommend that users try multiple training sessions.
<details>
  <summary>
    <b>1) FLIR</b>
  </summary>
  
  Place the corresponding images in each subfolder of the folder `./FLIR_datasets/`. Then run the command
  ```bash
  bash ./train_FLIR.sh
  ```
</details>


<details>
  <summary>
    <b>2) Brno</b>
  </summary>
  
  Place the corresponding images in each subfolder of the folder `./Brno_datasets/`. Then run the command
   ```bash
   bash ./train_Brno.sh
   ```

</details>

## Labeled Segmentation Masks
We annotated a subset of [Brno](https://drive.google.com/file/d/18giAtQdYH_lwVPj7rYEfmJ7B-ljP6_eZ/view?usp=sharing) datasets with pixel-level category labels, which may catalyze research on the colorization and semantic segmentation of nighttime TIR images.

![Labeled Masks](https://github.com/FuyaLuo/FoalGAN/blob/main/docs/docs/Masks.png)

## Evaluation
<details>
  <summary>
    <b>1) Semantic segmenation</b>
  </summary>
  
   Download the code for the semantic segmentation model [HMSANet](https://github.com/segcv/hierarchical-multi-scale-attention) and then follow the instructions to install it. Next, download the pre-trained [model](https://drive.google.com/open?id=1fs-uLzXvmsISbS635eRZCc5uzQdBIZ_U) on the Cityscape dataset, and then change line 52 in the `config.py` to the path of the folder where these pre-training weights are located. After that, download the segmentation mask and code for both datasets via [google drive](https://drive.google.com/file/d/1SACn6rJm1Ry_2x-bfslA4Y52AE8N3wzE/view?usp=sharing). Put `misc.py` in folder `./utils/` and replace the original file, all other files are placed inside the directory `/semantic-segmentation-main/`. For the evaluation on FLIR dataset, run the command
   ```bash
   python -m torch.distributed.launch --nproc_per_node=1 eval_FLIR.py --dataset cityscapes --syncbn --apex --fp16 --eval_folder /Your_FLIR_Results_Path --snapshot /Your_Pretrained_Models_Path/cityscapes_ocrnet.HRNet_Mscale_outstanding-turtle.pth --dump_assets --dump_all_images --result_dir ./Your_FLIR_Mask_SavePath
   ```
   And for the evaluation on Brno dataset, run the command
   ```bash
   python -m torch.distributed.launch --nproc_per_node=1 eval_Brno.py --dataset cityscapes --syncbn --apex --fp16 --eval_folder /Your_Brno_Results_Path --snapshot /Your_Pretrained_Models_Path/cityscapes_ocrnet.HRNet_Mscale_outstanding-turtle.pth --dump_assets --dump_all_images --result_dir ./Your_Brno_Mask_SavePath
   ```
   
</details>

<details>
  <summary>
    <b>2) Object detection</b>
  </summary>
  
  Download the code for [YOLOv7](https://github.com/WongKinYiu/yolov7), then follow the instructions to install it. Next, download the YOLOv7 detection txt file we transformed from the FLIR and Brno datasets via [google drive](https://drive.google.com/file/d/1EHmVXn-t8om_74Ozi8Y_yAwAGxkV9qc5/view?usp=sharing). Once the unzip is complete, place all files in the `/yolov7-main/` folder. Note that the files `FLIR.yaml`, `FLIR_imglist.txt`, `Brno.yaml` and `Brno_imglist.txt` should be placed in the directory `/yolov7-main/data/`. Then, the translation results of FLIR and Brno should be placed inside the `/yolov7-main/FLIR_datasets/images/` and `/yolov7-main/Brno_datasets/images/` directories respectively. For the evaluation on FLIR dataset, run the command
  ```bash
  python test.py --data data/FLIR.yaml --img 640 --batch 32 --conf 0.001 --iou 0.65 --device 0 --weights pretrain_weights/yolov7.pt --name FLIR_640_val --verbose
  ```
       
</details>


<details>
  <summary>
    <b>3) Edge consistency</b>
  </summary>
  
   Please refer to the [PearlGAN](https://github.com/FuyaLuo/PearlGAN) repository.

    
</details>

## Downloading files using Baidu Cloud Drive
If the above Google Drive link is not available, you can try to download the relevant code and files through the [Baidu cloud link](https://pan.baidu.com/s/1QgIPiFFGOwBfNsPWYa0Mfg), extraction code: foal.

## Citation
If you like our work and use the code or models for your research, please cite our work as follows.
```
@article{luo2024nighttime,
  title={Nighttime thermal infrared image colorization with feedback-based object appearance learning}, 
  author={Luo, Fu-Ya and Liu, Shu-Lin and Cao, Yi-Jun and Yang, Kai-Fu and Xie, Chang-Yong and Liu, Yong and Li, Yong-Jie},
  journal={IEEE Transactions on Circuits and Systems for Video Technology}, 
  year={2024},
  volume={34},
  number={6},
  pages={4745-4761}
}
```

## License

The codes and the pretrained model in this repository are under the BSD 2-Clause "Simplified" license as specified by the LICENSE file. 

## Acknowledgments
This code is heavily borrowed from [ToDayGAN](https://github.com/AAnoosheh/ToDayGAN).  
Spectral Normalization code is borrowed from [BigGAN-PyTorch](https://github.com/ajbrock/BigGAN-PyTorch/blob/master/layers.py).  
