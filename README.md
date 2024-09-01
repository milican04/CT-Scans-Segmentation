# CT-Scans-Segmentation

The project was created as part of the Machine Learning course at the Faculty of Mathematics, University of Belgrade. It explores the implementation of the [U-Net](https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/) CNN architecture and its application in the problem of segmentation of biomedical images, such as locating tumors in patient CT scans.

The proposed U-Net based models are trained on the datasets used for the [Medical Segmentation Decathlon](http://medicaldecathlon.com/). Specifically, two tasks (organs) were explored from the challenge - the lung tumor segmentation task and the liver tumor segmentation task.

### Project organization

The project is split into multiple Jupyter notebooks. The root level notebooks (Data-Preprocessing.ipynb, U-Net.ipynb, Train-Eval-Utils.ipynb) contain code used by the notebooks in the liver_tumor_segmentation and lung_tumor_segmentation directories. The notebooks in the segmentation directories contain the training and evaluation code for different models trained on the respective dataset, as well as more detailed explanations behind the choices we've made during the process. After downloading the required datasets, running the cells sequentially in the segmentation directoriy notebooks should work smoothly.

### Project results

The trained models had very satisfactory performance for the liver tumor dataset, but performed poorly for the lung tumor dataset. However, this is not surprising since the models trained for the Medical Segmentation Decathlon consistently perform poorly on the lung tumor segmentation challenge. The reason behind this could be the relatively small dataset of only 63 lung CT scans and the extreme class imbalance, with most of the CT scan slices containing no tumor traces, or in very small amounts.

### Authors

- Milica Nikolić
- Aleksandar Stefanović

### Required data and dependencies

Use the following link to download data (Lungs and Liver datasets are used): http://medicaldecathlon.com/

Use pip install to install following libraries:
- json,
- numpy,
- nibabel,
- matplotlib
- torch

### Trained models

The trained models can be downloaded from [this location](https://drive.google.com/drive/folders/1I010v-5x9I9dxLBbR0NWgfkNG7hcfBUh). 
