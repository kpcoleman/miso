# Resolving tissue complexity by multi-modal spatial omics modeling with MISO

### Kyle Coleman*, Daiwei Zhang, Amelia Schroeder, Melanie Braisted, Niklas Blank, Alexis Jazmyn, Hanying Yan, Yanxiang Deng, Elizabeth F. Furth, Edward B. Lee, Christoph A. Thaiss, Jian Hu*, Mingyao Li*

MISO is a deep-learning based method developed for the integration and clustering of multi-modal spatial omics data. MISO requires minimal hyperparameter tuning, and can be applied to any number of 
omic and imaging data modalities from any multi-modal spatial omics experiment. MISO has been evaluated on datasets from experiements including spatial transcriptomics (transcriptomics and histology), 
spatial epigenome-transcriptome co-profiling (chromatin accessibility, histone modification, and transcriptomics), spatial CITE-seq (transcriptomics, 
proteomics, and histology), and spatial transcriptomics and metabolomics (transcriptomics, metabolomics, and histology)

![png](images/workflow.png)


## MISO Installation

MISO has been tested on the following operating systems: 
- macOS: Ventura (13.5.1)
- Linux: CentOS (7) 

MISO installation requires python version 3.7. The version of python can be checked by: 
```python
import platform
platform.python_version()
```

    '3.7.12'

We recommend creating and activating a new conda environment when installing the MISO package. For instance, 
```bash
conda create -n miso python=3.7.12
conda activate miso
```        

## Software Requirements  
scikit-learn==1.0.2  
torch  
torchvision  
numpy==1.21.6  
Pillow==6.1.0  
opencv-python=4.6.0  
scipy==1.7.3  
einops==0.6.0  
scanpy==1.9.1  

H&E image feature extraction code is based on HIPT and iSTAR. Pre-trained vision transformer models are from HIPT.

