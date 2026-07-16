

![Kherson_transfer](pic\Kherson_transfer.png)

# A variation attention-guided network for spatio-temporal fusion

## 1. Installation
Currently there is no pypi or conda package to install from. You therefore have to clone the package and install manually. 
osgeo/GDAL is needed for the installation, which is easiest to install via conda. Otherwise, you need to prepare the torch and CUDA package, refer [here](https://pytorch.org/get-started/locally/)

```
git clone https://github.com/DecaiJin/Attention-Guided-Spatiotemporal-Data-Fusion.git
cd <path where you want to keep the package>
python install requirements.txt
```

## 2. Training

### 2.1 Prepare training data
You can use simple scripts to make training set, step is following:

1) download modis and landsat image pair and make sure they have **same size**
2) make sure the name can be divided by "_", like followed:

 ```
 data_root/
|
|-COARSE_YYYYMMDD.tif
|
|-FINE_YYYYMMDD.tif
|
|...
 
 ```

`YYYYMMDD`means the acquired time(you can define your data name).  you can customize `xx` as suffix to distinguish images with same date

3) make training meta info

You can use the `scripts/make_train_meta.py` to produce random image pairs. The script will output a `.csv` file to recored paired info. You also can make yourself training meta, format like this:

| t1/lr_data            | t2/lr_data            | t1/hr_data          | t2/hr_data          |
| :-------------------- | --------------------- | ------------------- | ------------------- |
| `COARSE_20011007.tif` | `COARSE_20020111.tif` | `FINE_20011007.tif` | `FINE_20020111.tif` |
| `COARSE_20011016.tif` | `COARSE_20020426.tif` | `FINE_20011016.tif` | `FINE_20020426.tif` |

4. clip remote sensing image to sub-img

  you can use `scripts/clip_image.py` to clip the images in `data_root` based on training meta in `step 3`. For example:

`python scripts/clip_image.py --input data_root --meta_set -<path of train meta>` . And you will get the data structure , as followed:

```
data_root/
|
|--t1
|----lr_data
|----hr_data
|
|--t2
|----lr_data
|----hr_data
|
```




### 2.2 train or fine-tune model
We used torch accelerate to distribute model on multi-gpu. Directly train model with yourself data:

```python
accelerate launch train.py -i <your train data> --test_path <your test data> -ic 6 -oc 6 --max_value 10000 --channel_weight 0.23,0.23,0.23,0.05,0.05,0.2
```

Default value of `test_path` is `None`, which means that there is no validation data to show the effect of model training. `max_value` refers to the max value of image reflectance and paired images should have same max_value. `channel_weight` is the weight for each band. We suggest sum of channel_weight equal to 1.0. `ic` and `oc` are the input bands and output bands.

Train model with pretrained weights:

```python
accelerate launch train.py -i <train_dataroot> --test_path <val data set> --state <path of pretrained weight>
```

Our weight can be downloaded from [here](https://drive.google.com/file/d/1KErzj-u3pq7dnq50rBopV5-9ibd-0mvr/view?usp=drive_link).  Currently, it supports six channels (`B, G, R, NIR, Swir1, Swir2`). Future, we will consider to construct a model weight for `Modis-Sentinel` 、 `Sentinel3-Sentinel2`、`Sentinel2-GF` data fusion.

## 3. Inference

We designed two inference modes: `python script and C++ exe`. We suggest using `exe`, which don't need to prepare runtime environment and easy to use.

### 3.1 Using python script

If you have trained the model with your own data or modified the network structure, you may need to use the python script `test.py` for example:

```python
python test.py -test_path <root path to be predicted> --state <your model.pth path>
```

if you modify architecture, you must modify the responding architecture parameter, More parameters details can be seen:

`python test.py --help`

### 3.2 Using deployed exe (recommend)

Otherwise, we recommend you to use the `exe` version. The `exe` version does not require additional runtime configuration and is easy to use. 

Before you use the `exe` to predict your own images, you need to add the `. /bin` folder to the system environment variable.

We write `setupvars.bat` to execute it. You must run it as administrator. If the command fails, you can also manually add `bin` to the system variable.  And trained weight must be renamed to `agsdf_model.bin` and `agsdf_model.xml`

Next, you can use terminal to predict image:

```bash
./agsdf.exe c1_file_path f1_file_path c2_file_path out_file_path MaxValue

Usage:
    - c1_file_path: the coarse image path at time 1
    - f1_file_path: the fine image path at time 1
    - c2_file_path: the coarse image path at predicted time
    - out_file_path: output path + filename of predicted image
    - MaxValue: 1 or 10000, resonding to normal saving type(float32|int16)
```



## 4. Evaluation
| Metric index                            | Characteristics                      | Range   | Optimal  value |
| --------------------------------------- | ------------------------------------ | ------- | -------------- |
| RMSE (Root  mean square error)          | Evaluating  pixel spectral accuracy  | [0, 1]  | 0              |
| AD (Average  difference)                | Indicating  global spectral accuracy | [-1, 1] | 0              |
| EDGE  (Robert’s edge)                   | Evaluating  edge accuracy.           | [-1, 1] | 0              |
| LBP (Local  binary patterns)            | Measuring  texture matching          | [-1, 1] | 0              |
| SSIM (Structural  similarity)           | Evaluating structural  details       | [0, 1]  | 1              |
| ERGAS (Error  relative global accuracy) | Evaluating  the resolution quality   | [0, +∞] | 0              |

You can download `AD/RMSE/LBP/EDGE` evaluate code from [here](https://github.com/XZhu-lab/Fusion-accuracy-assessment)

## 5. Deploy

We use `openvino` to accelerate inference efficiency.  Currently, it only support cpu device. In future, we may develop the GPU device.

If you want to execute your own model in C++ `exe`, you just need to convert your weights. we also provide the conversion code in  `script/export_model.py` . Usage as follow:

```python
python scripts/export_model.py --state <your training weight path> --output <output path of *.onnx, *.bin, *.xml>  --patch_size <sliding window size>
```

 output `*.xml` and `*.bin`  save in root path. Copy and renamed these files to the `exe` root folder.

## 6. Development

1. We have developed six band pretrained model. Paper has only three band weight.
2. We will probably build `Modis-Sentinel` model and offer corresponding training weight.
3. `TensorRT`-based `GPU`-accelerated versions may also be released.

## Citations

