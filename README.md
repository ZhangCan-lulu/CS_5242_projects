# NUS-CS5242-projects(video classification)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/action-segmentation-with-joint-self/action-segmentation-on-breakfast)](https://paperswithcode.com/sota/action-segmentation-on-breakfast?p=action-segmentation-with-joint-self)

---
Implement based on the official PyTorch of the paper:

**Action Segmentation with Joint Self-Supervised Temporal Domain Adaptation**  
[__***Min-Hung Chen***__](https://www.linkedin.com/in/chensteven), [Baopu Li](https://www.linkedin.com/in/paul-lee-46b2382b/), [Yingze Bao](https://www.linkedin.com/in/yingze/), [Zsolt Kira](https://www.cc.gatech.edu/~zk15/), [Ghassan AlRegib (Advisor)](https://ghassanalregib.info/) <br>
[*IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2020*](http://cvpr2020.thecvf.com/)   
[[arXiv](https://arxiv.org/abs/2003.02824)]

<p align="center">
<img src="webpage/Overview.png?raw=true" width="70%">
</p>


---
## Requirements
Tested with:
* Ubuntu 18.04.2 LTS
* PyTorch 1.1.0
* Torchvision 0.3.0
* Python 3.7.3
* GeForce GTX 1080Ti
* CUDA 9.2.88
* CuDNN 7.14

Or you can directly use our environment file:
```
conda env create -f environment.yml
```

---
## Data Preparation
* Clone the this repository:
```
git clone https://github.com/lusindazc/CS_5242_projects.git
cd SSTDA-master
```
* Download the [Dataset](https://www.dropbox.com/s/yodx2dknti0ah2v/Datasets.zip?dl=0) folder, which contains the features and the ground truth labels. (~30GB)
* Extract it so that you have the `Datasets` folder.
* The default path for the dataset is `../../Datasets/action-segmentation/` if the current location is `./action-segmentation-DA/`. If you change the dataset path, you need to edit the scripts as well.

---
## Usage
#### Quick Run
* Since there are lots of arguments, we recommend to directly run the scripts.
* All the scripts are in the folder `scripts/` with the name `run_<dataset>_<method>.sh`.
* You can simply copy any script to the main folder (same location as all the `.py` files), and run the script as below:
```
# one example
bash run_breakfast_SSTDA_noatt_1.sh
```
The script will do training, predicting and evaluation for all the splits on the dataset (`<dataset>`) using the method (`<method>`).

#### More Details
* In each script, you may want to modify the following sections:
  * `# === Mode Switch On/Off === #`
    * `training`, `predict` and `eval` are the modes that can be switched on or off by set as `true` or `false`.
  * `# === Paths === #`
    * `path_data` needs to be the same as the location of the input data.
    * `path_model` and `path_result` are the path for output models and prediction. The folders will be created if not existing.
  * `# === Main Program === #`
    * You can run only the specific splits by editing `for split in 1 2 3 ...` (line 53).
* We DO NOT recommend to edit other parts (e.g. `# === Config & Setting === #
`); otherwise the implementation may be different.


