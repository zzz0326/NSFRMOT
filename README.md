# Nuisance-Suppressed Feature Refinement for UAV Multi-Object Tracking

## Environment
```
conda create -n NSFRMOT python=3.8 -y && conda activate NSFRMOT
python -m pip install -r requirements.txt
python setup.py develop
python -m pip install cython
python -m pip install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
python -m pip install cython_bbox
```

## Dataset
Download the [VisDrone-MOT](https://github.com/VisDrone/VisDrone-Dataset) and [UAVDT](https://sites.google.com/view/grli-uavdt/%E9%A6%96%E9%A1%B5), and put them under the <DataPath> in the following structure:
```
DataPath   
   |——————VisDrone-MOT
   |        └——————VisDrone2019-MOT-train
   |        └——————VisDrone2019-MOT-val
   |        └——————VisDrone2019-MOT-test-dev
   └——————UAVDT
            └——————uavdt-test
            └——————uavdt-train
```

## Pre-trained weights
### Detector weights
The detector [weight](https://drive.google.com/drive/folders/1MTtu_gbvK7akKjr3cLNlX28L80fcSFQV) of VisDrone-MOT comes from [u2mot](https://github.com/alibaba/u2mot).

The weight corresponding to UAVDT is obtained from this [link](https://drive.google.com/file/d/19Uvi5d3qSkxmixIXPOcLAm-dh_2YjWo_/view?usp=sharing).
### Re-ID weights
| Benchmark |    Weights    |
|:---------:|:-------------:|
|  VisDrone | Released soon |
|   UAVDT   | Released soon |


## Training

1. Use any ReID model to extract features on the training set, and save the feature maps to the path `<npy_path>`.
2. Modify the paths at the bottom of the two Python files under the `train_reid` folder, then run:

   ```bash
   python BR_train.py
   python SGR_train.py
   ```

   After training, three model weights will be generated.
## Tracking
1. Download the full code of [u2mot](https://github.com/alibaba/u2mot), and move `Track/track_cluster.py` to the `tools` directory.
2. Move all other `.py` files from the `Track` folder to `yolox/tracker`.
3. Move the `UAVDT` folder to `yolox/tracker/GMC_files`.

Then, run the following commands to perform tracking:

```bash
# Tracking on VisDrone
python tools/track_cluster.py <VisDrone_testset> \
    --benchmark VisDrone \
    -f exps/example/NSFRMOT/yolox_x_NSFRMOT_visdrone.py \
    -c <VisDrone_Pretrain> \
    --device 0 --fp16 --fuse \
    --cmc-method file --cmc-file-dir VisDrone/test-dev

# Tracking on UAVDT
python tools/track_cluster.py <UAVDT_testset> \
    --benchmark UAVDT \
    -f exps/example/NSFRMOT/yolox_x_NSFRMOT_uavdt.py \
    -c <UAVDT_Pretrain> \
    --device 0 --fp16 --fuse \
    --cmc-method file --cmc-file-dir UAVDT
```
## Evaluation
* **Evaluate on VisDrone**
```
python tools/utils/eval.py <DataPath>/VisDrone2019-MOT-test-dev/annotations
```
* **Evaluate on UAVDT**
```
python tools/utils/eval_uavdt.py <DataPath>/uavdt_test/annotations
```
## Acknowledgement
A large part of the code is borrowed from [u2mot](https://github.com/alibaba/u2mot). Many thanks for their wonderful work.
