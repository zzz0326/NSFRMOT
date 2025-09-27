# CausalMOT: Learning Causal Invariant Representation for UAV Multi-Object Tracking

## Environment
```
conda create -n CausalMOT python=3.8 -y && conda activate CausalMOT
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

## Code
**All codes related to training, testing, and evaluation will be released soon.**

## Training
* **Train on VisDrone**
```
python Tracker_train/step_mamba_tracker.py <DataPath>/VisDrone2019-MOT-train/annotations --step 1 --benchmark VisDrone
python Tracker_train/step_mamba_tracker.py <DataPath>/VisDrone2019-MOT-train/annotations --step 2 --benchmark VisDrone
```
* **Train on UAVDT**
```
python Tracker_train/step_mamba_tracker.py <DataPath>/uavdt_train/annotations --step 1 --benchmark UAVDT
python Tracker_train/step_mamba_tracker.py <DataPath>/uavdt_train/annotations --step 2 --benchmark UAVDT
```
## Tracking
* **Test on VisDrone**
```
python tools/track_VisDrone.py <DataPath>/VisDrone2019-MOT-test-dev/sequences --benchmark VisDrone -f exps/example/CausalMOT/yolox_x_u2mot_uavdt.py -c <VisDrone-Detetcor-Weight> --device 1 --fp16 --fuse --cmc-method file --cmc-file-dir VisDrone/test-dev
```
* **Test on UAVDT**
```
python tools/track_UAVDT.py <DataPath>/uavdt_test/sequences --benchmark UAVDT -f exps/example/CausalMOT/yolox_x_u2mot_uavdt.py -c <UAVDT-Detetcor-Weight> --device 1 --fp16 --fuse --cmc-method file --cmc-file-dir UAVDT/test-dev
```
## Evaluation
* **Evaluate on VisDrone**
```
python tools/utils/eval.py <DataPath>/VisDrone2019-MOT-test-dev/annotations --benchmark VisDrone
```
* **Evaluate on UAVDT**
```
python tools/utils/eval.py <DataPath>/uavdt_test/annotations --benchmark UAVDT
```
## Acknowledgement
A large part of the code is borrowed from [u2mot](https://github.com/alibaba/u2mot). Many thanks for their wonderful work.
