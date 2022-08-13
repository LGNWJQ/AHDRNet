# AHDRNet

This is an unofficial pytorch implementation of [AHDRNet](https://www.researchgate.net/publication/332603952_Attention-guided_Network_for_Ghost-free_High_Dynamic_Range_Imaging)

## Requirements
* python 3.8
* pytorch 1.11.0
* Use the **requirements.txt** file to install the necessary packages
  * `pip install -r requirements.txt`

## Usage
### Dataset Structure
* Download [here](https://cseweb.ucsd.edu/~viscomp/projects/SIG17HDR/) 
* Organize the dataset file as follows, and set the dataset path as the **Main Path** in args_file.py
* There are only 10 groups of test data under the original EXTRA folder, you can add the five groups of data under the PAPER folder to the EXTRA folder, rename it 011 - 015, and note that there are only five files under the folder to check each group of data: three tif format pictures, one hdr format picture, and one txt folder
```commandline
Main Path
├─Test
│  ├─EXTRA
│  │  ├─001
|  |  | ├── 1.tif
|  |  | ├── 2.tif
|  |  | ├── 3.tif
|  |  | ├── exposure.txt
|  |  | └── HDRImg.hdr (optional)
│  │  ├─002
│  │  ├─003
│  │  ├─004
│  │  ├─005
│  │  ├─006
│  │  ├─007
│  │  ├─008
│  │  ├─009
│  │  ├─010
│  │  ├─011
│  │  ├─012
│  │  ├─013
│  │  ├─014
│  │  └─015
│  └─PAPER
│      ├─BarbequeDay
│      ├─LadySitting
│      ├─ManStanding
│      ├─PeopleStanding
│      └─PeopleTalking
└─Training
    ├─001
    ├─002
    ├─003
    ├─004
    └─...

```


### Train
1. Open **args_file.py** file to set training related parameters
2. Run **train.py** to Train
3. Follow the prompts to open the tensorboard to observe the validation results during training, the loss function changes

### Test
1. Open **fusion_HDR.py** file, Set weight path, dataset path, path to save fusion results
2. Run **fusion_HDR.py** to get HDR image.

## Reference
* https://github.com/qingsenyangit/AHDRNet

## Some Results
The effect of opening only with Adobe Photoshop, without any processing

![](https://raw.githubusercontent.com/LGNWJQ/picgo/main/AHDR/R3.png)

![](https://raw.githubusercontent.com/LGNWJQ/picgo/main/AHDR/R5.png)

![](https://raw.githubusercontent.com/LGNWJQ/picgo/main/AHDR/R11.png)
