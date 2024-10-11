<div align="center">

  <h1 align="center">[NIPS2024]DC-Gaussian: Improving 3D Gaussian Splatting for Reflective Dash Cam Videos</h1>

### [Project Page](https://linhanwang.github.io/dcgaussian/)

</div>

## 🖼️ Demo
<div align="center">
<img width="800" alt="image" src="assets/teaser.png">
<p>Given a sequence of video captured by a dash cam that may contain obstructions like reflections and occlusions, DC-Gaussian achieves high-fidelity novel view synthesis getting rid of the obstructions. (a) dash cam; (b) original video frame; (c) novel view rendering with obstruction removal.</p>
</div>

## 📖 Abstract
We present DC-Gaussian, a new method for generating novel views from in-vehicle dash cam videos. 

While neural rendering techniques have made significant strides in driving scenarios, existing methods are primarily designed for videos collected by autonomous vehicles. However, these videos are limited in both quantity and diversity compared to dash cam videos, which are more widely used across various types of vehicles and capture a broader range of scenarios. Dash cam videos often suffer from severe obstructions such as reflections and occlusions on the windshields, which significantly impede the application of neural rendering techniques. To address this challenge, we develop DC-Gaussian based on the recent real-time neural rendering technique 3D Gaussian Splatting (3DGS). Our approach includes an adaptive image decomposition module to model reflections and occlusions in a unified manner. Additionally, we introduce illumination- aware obstruction modeling to manage reflections and occlusions under varying lighting conditions. Lastly, we employ a geometry-guided Gaussian enhancement strategy to improve rendering details by incorporating additional geometry priors.

Experiments on self-captured and public dash cam videos show that our method not only achieves state-of-the-art performance in novel view synthesis, but also accurately reconstructing captured scenes getting rid of obstructions.

## 🚀 Pipeline

<div align="center">
<img width="800" alt="image" src="assets/figure_pipeline.png">
</div>

## Environment setup

```
conda env create --file environment.yml
conda activate dcgaussian

conda install -c "nvidia/label/cuda-11.7.1" cuda-toolkit

pip install submodules/simple-knn/
pip install submodules/diff-gaussian-rasterization/
pip install ninja git+https://github.com/hturki/tiny-cuda-nn.git@ht/res-grid#subdirectory=bindings/torch
```

## Dataset 

Data used in our work can be downloaded from [here](https://zenodo.org/records/13916656). They are in colmap format.

## Train and evaluation

```
python train.py -s $dataset -m $output --eval -r 1 --win_type iom --position_lr_init 0.00008  --position_lr_final 0.000008 --scaling_lr 0.005 --percent_dense 0.0005 --densify_until_iter 20000
python render.py -s $dataset -m $output --skip_train
python metrics.py -m $output
```

The rendered images will be located in `$output/test/ours_30000/renders/`. In addition to the composed images, the rendering process will also produce obstruction, transmission, depth, and opacity maps.

## 🗓️ TODO
- [✔] Relase training code
- [✔] Environment setup
- [✔] Release training, rendering and eval scripts
- [✔] Release dataset

## Citation

If you find DC-Gaussian useful in your research or application, please cite using this BibTex:

```
@article{wang2024dc,
  title={DC-Gaussian: Improving 3D Gaussian Splatting for Reflective Dash Cam Videos},
  author={Wang, Linhan and Cheng, Kai and Lei, Shuo and Wang, Shengkun and Yin, Wei and Lei, Chenyang and Long, Xiaoxiao and Lu, Chang-Tien},
  journal={NeurIPS 2024},
  year={2024}
}
```

## Acknowledgements

We borrowed code from [3DGS](https://github.com/graphdeco-inria/gaussian-splatting), [tinycudann](https://github.com/NVlabs/tiny-cuda-nn) and [SUDS](https://github.com/hturki/suds). Thanks for their contribution to the community.
