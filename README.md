# BWG: Learning Generalized Segmentation for Foggy-scenes by Bi-directional Wavelet Guidance

This is the official implementation of our work entitled as ```Learning Generalized Segmentation for Foggy-scenes by Bi-directional Wavelet Guidance```.

## Methodology Overview

Learning scene semantics that can be well generalized to foggy conditions is important for safety-crucial applications such as autonomous driving. 
Existing methods need both annotated clear images and foggy images to train a curriculum domain adaptation model.
Unfortunately, these methods can only generalize to the target foggy domain that has seen in the training stage, but the foggy domains vary a lot in both urban-scene styles and fog styles.

In this paper, we propose to learn scene segmentation well generalized to foggy-scenes under the domain generalization setting, which does not involve any foggy images in the training stage and can generalize to any arbitrary unseen foggy scenes. 
We argue that an ideal segmentation model that can be well generalized to foggy-scenes need to simultaneously enhance the content, de-correlate the urban-scene style and de-correlate the fog style. 
As the content (e.g., scene semantics) rest more in low-frequency features while the style of urban-scene and fog rest more in high-frequency features, we propose a novel ```bi-directional wavelet guidance``` (BWG) mechanism to realize the above three objectives in a divide-and-conquer manner. 
With the aid of Haar wavelet transformation,
the low frequency component is concentrated on the content enhancement self-attention, while the high frequency components are shifted to the style and fog self-attention for de-correlation purpose.
It is integrated into existing mask-level Transformer segmentation pipelines in a learnable fashion.

![avatar](/BWG.png)

## Environment Configuration
The development of ```BWG``` is largely based on Mask2Former [https://bowenc0221.github.io/mask2former/].

```Detectron2``` and ```PyTorch``` are required. Other packages include:
```
    ipython==7.30.1
    numpy==1.21.4
    torch==1.8.1
    torchvision==0.9.1
    opencv-python==4.5.5.62
    Shapely==1.8.0
    h5py==3.6.0
    scipy==1.7.3
    submitit==1.4.1
    scikit-image==0.19.1
    Cython==0.29.27
    timm==0.4.12
```

## Training on Source Domain
An example of training on ```CityScapes``` source domain is given below.

```
python train_net.py --num-gpus 2 --config-file configs/cityscapes/semantic-segmentation/swin/maskformer2_swin_base_IN21k_384_bs16_90k.yaml
```

## Evaluating mIoU on Unseen Target Domains

The below lines are the example code to infer on ```Foggy CityScapes``` and ```Foggy Zurich``` unseen target domains.
```
python train_net.py --config-file configs/cityscapes/semantic-segmentation/swin/maskformer2_swin_base_IN21k_384_bs16_90k.yaml --eval-only MODEL.WEIGHTS E:/DGtask/DGViT/Mask2Former-main/output_fc/model_final.pth
```
```
python train_net.py --config-file configs/cityscapes/semantic-segmentation/swin/maskformer2_swin_base_IN21k_384_bs16_90k.yaml --eval-only MODEL.WEIGHTS E:/DGtask/DGViT/Mask2Former-main/output_fz/model_final.pth
```
## Inferring predictions on Unseen Target Domains
The below line is an example code to infer visual prediction results on unseen target domains.
```
python demo.py --config-file ../configs/cityscapes/semantic-segmentation/swin/maskformer2_swin_base_IN21k_384_bs16_90k.yaml --input citys_test --output inf --opts MODEL.WEIGHTS E:/DGtask/DGViT/Mask2Former-main/output_citys/model_final.pth
```

## Cite the proposed BWG

If you find the proposed BWG is useful for your task please cite our work as follows:

```BibTeX

```

## Acknowledgement

The development of BWG is largely based on Mask2Former [https://bowenc0221.github.io/mask2former/].

The majority of Mask2Former is licensed under a [MIT License](LICENSE).

However portions of the project are available under separate license terms: Swin-Transformer-Semantic-Segmentation is licensed under the [MIT license](https://github.com/SwinTransformer/Swin-Transformer-Semantic-Segmentation/blob/main/LICENSE), Deformable-DETR is licensed under the [Apache-2.0 License](https://github.com/fundamentalvision/Deformable-DETR/blob/main/LICENSE).

If you find the proposed BWG is useful for domain-generalized urban-scene segmentation, please also cite the asserts from the orginal Mask2Former as follows:

```BibTeX
@inproceedings{cheng2021mask2former,
  title={Masked-attention Mask Transformer for Universal Image Segmentation},
  author={Bowen Cheng and Ishan Misra and Alexander G. Schwing and Alexander Kirillov and Rohit Girdhar},
  journal={CVPR},
  year={2022}
}
```

## Contact

For further information or questions, please contact Qi Bi via ```q.bi@uva.nl``` or ```2009biqi@163.com```.

