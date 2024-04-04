
# Spatio-Temporal Image Encoding


This repository contains the implementation of the Enhanced Spatio-Temporal Image Encoding (ESTIE), in order to perform Online Human Activity Recognition using 3D skeletons. This method is based on the use of the Spatio-Temporal Image Encoding (STIE) and the motion energy in order to encode a sequence of 3D skeletons into an image, while preserving both spatial and temporal dependencies. 

Our paper can be found at:

[Enhanced Spatio- Temporal Image Encoding for Online Human Activity Recognition](https://ieeexplore.ieee.org/abstract/document/10459847)

If you use or build on our work, please consider citing us:

```
@INPROCEEDINGS{10459847,
  author={Mokhtari, Nassim and Fer, Vincent and Nédélec, Alexis and Gilles, Marlene and de Loor, Pierre},
  booktitle={2023 International Conference on Machine Learning and Applications (ICMLA)}, 
  title={Enhanced Spatio- Temporal Image Encoding for Online Human Activity Recognition}, 
  year={2023},
  volume={},
  number={},
  pages={884-889},
  keywords={Training;Image coding;Three-dimensional displays;Image recognition;Time series analysis;Focusing;Streaming media;3D Skeleton Data;Spatio-temporal Image En-coding;Motion Energy;Online Action Recognition;Human Activity Recognition;Deep learning},
  doi={10.1109/ICMLA58977.2023.00130}}

```


## Dataset
Before running our code, please unzip the archive **data.zip** provided in this repo. This archive contains skeleton data and sequence labels from the [Online Action Detection dataset](https://www.icst.pku.edu.cn/struct/Projects/OAD.html).

**note:** If you are using your own dataset, please consider adjusting the *load_data_file()* function.

## Usage
You can start the encoding using the default parameters by running the STIE.py from the command line :

	python ./ESTIE.py

Several parameters can be used to adapt the encoding according to your needs. You can find more details about these parameters using :

	python ./ESTIE.py --help

## Encoded Sequence

The ESTIE image is the combination of the [STIE](https://github.com/nassimmokhtari/Spatio-Temporal-Image-Encoding) and the motion energy proposed by [Liu et al.](https://www.sciencedirect.com/science/article/abs/pii/S0031320317300936)

### STIE image
![STIE example](https://github.com/nassimmokhtari/Enhanced-Spatio-Temporal-Image-Encoding/blob/main/images/STIE.png)

### Motion Energy
![motion energy example](https://github.com/nassimmokhtari/Enhanced-Spatio-Temporal-Image-Encoding/blob/main/images/MotionEnergy.png)

### ESTIE

![ESTIE example](https://github.com/nassimmokhtari/Enhanced-Spatio-Temporal-Image-Encoding/blob/main/images/ESTIE.png)


