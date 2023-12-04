# BWG: Learning Generalized Segmentation for Foggy-scenes by Bi-directional Wavelet Guidance

This is the official implementation of our work entitled as ```Learning Generalized Segmentation for Foggy-scenes by Bi-directional Wavelet Guidance```.

## Methodology Overview

Recent work has shown that mask-level segmentation Transformer (e.g., Mask2Former) is a scalable learner for domain generalized semantic segmentation. Unfortunately, we empirically observed that, a mask-level representation is better at representing content but more sensitive to style variations; its low-resolution counterpart on the contrary is less capable to represent content but more robust to the style variations.

Overall, the mask representation and its down-sampled counterpart shows complementary properties when handling samples from different domains. Thus, it is natural to jointly
leverage both mask representation and its down-sampled counterparts, so as to at the same time stabilizing the content and be insensitive to the style variation.

## Environment Configuration
The development of CMFormer is largely based on Mask2Former [https://bowenc0221.github.io/mask2former/].

```Detectron2``` and ```PyTorch``` are required. Other packages include:
```
cython
scipy
shapely
timm
h5py
submitit
scikit-image
```

## Training on Source Domain
An example of training on ```CityScapes``` source domain is given below.

```
python train_net.py --num-gpus 2 --config-file configs/cityscapes/semantic-segmentation/swin/maskformer2_swin_base_IN21k_384_bs16_90k.yaml
```

## Inference on Unseen Target Domains

The below lines are the example code to infer on ```GTA``` and ```SYN``` unseen target domains.
```
python train_net.py --config-file configs/cityscapes/semantic-segmentation/swin/maskformer2_swin_base_IN21k_384_bs16_90k.yaml --eval-only MODEL.WEIGHTS E:/DGtask/DGViT/Mask2Former-main/output_gta/model_final.pth
```
```
python train_net.py --config-file configs/cityscapes/semantic-segmentation/swin/maskformer2_swin_base_IN21k_384_bs16_90k.yaml --eval-only MODEL.WEIGHTS E:/DGtask/DGViT/Mask2Former-main/output_syn/model_final.pth
```

## Cite the proposed CMFormer

If you find the proposed CMFormer is useful for domain-generalized urban-scene segmentation, please cite our work as follows:

```BibTeX
@article{bi2023learning,
  title={Learning Content-enhanced Mask Transformer for Domain Generalized Urban-Scene Segmentation},
  author={Bi, Qi and You, Shaodi and Gevers, Theo},
  journal={arXiv preprint arXiv:2307.00371},
  year={2023}
}
```

## Acknowledgement

The development of CMFormer is largely based on Mask2Former [https://bowenc0221.github.io/mask2former/].

The majority of Mask2Former is licensed under a [MIT License](LICENSE).

However portions of the project are available under separate license terms: Swin-Transformer-Semantic-Segmentation is licensed under the [MIT license](https://github.com/SwinTransformer/Swin-Transformer-Semantic-Segmentation/blob/main/LICENSE), Deformable-DETR is licensed under the [Apache-2.0 License](https://github.com/fundamentalvision/Deformable-DETR/blob/main/LICENSE).

If you find the proposed CMFormer is useful for domain-generalized urban-scene segmentation, please also cite the asserts from the orginal Mask2Former as follows:

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

