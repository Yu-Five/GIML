# GIML

Here is the official code for **"General Incomplete Multimodal Learning via Dynamic Quality Perception"**, which proposes a general incomplete multimodal learning framework (GIML) that unifies intra-modality and inter-modality missing through dynamic quality perception, enabling their joint optimization within a single stage. Please refer to our [ECCV 2026 paper]() for more details.

## Main Dependencies

- Ubuntu 20.04  
- CUDA 11.3  
- PyTorch 1.11  
- Python 3.8.6  

## Data

We evaluate our method on five multimodal benchmarks across a range of tasks (e.g., sentiment analysis, action recognition), spanning AV, VT, AVT, and RGB-D. Detailed dataset configurations are given in `Section 4.1` of the paper. Additional information regarding dataset sources and preprocessing procedures is provided in the following.

### CREMA-D and KS

We follow the preprocessing pipeline from the [official repository of DMRNet (ECCV 2024)](https://github.com/shicaiwei123/ECCV2024-DMRNet/tree/main/audio-visual%20classification#readme) for the CREMA-D and KS datasets.  Please refer to `/dataset/data/CREMAD/video_preprocessing.py` and `/dataset/data/KineticSound/video_preprocessing.py` for the implementation. Before using the preprocessed datasets, please modify the `audio_path` and `visual_path` arguments in the `argparse` configuration, as well as `self.data_root` in `/dataset/CramedDataset.py`.

### MVSA-Single

MVSA-Single is a multimodal sentiment analysis dataset comprising image-text pairs. We download the data from the [official implementation of QMF (ICML 2023)](https://github.com/QingyangZhang/QMF). Please modify `self.data_root` in `/dataset/MVSADataset.py` according to the actual dataset download path. 

### MOSI

MOSI is a benchmark for multimodal sentiment analysis. We refer **to** the script provided by the [MMML repository](https://github.com/zehuiwu/MMML/blob/main/extract_audio.py) for data preprocessing.  See `/dataset/data/MOSI/mp4_to_wav_jpg.py`. Before using the preprocessed dataset, please modify `root_dir` in `/dataset/MOSIDataset.py`.

### NVGesture

NVGesture is a video dataset for dynamic hand gesture recognition. We adopt the dataset source and preprocessing pipeline from the [Real-time-GesRec repository](https://github.com/ahmetgunduz/Real-time-GesRec), which extracts key frames and motion features for action recognition using a sliding-window approach with 3D CNNs.  See `/dataset/data/NVGesture`. Before using the processed dataset, please modify the `annotation_path` and `root_path` in `/dataset/NVGestureDataset.py`. 

## Train 

During training, we load the pre-trained BERT model `bert-base-uncased` for the text modality. Please update the `bert_model_path` in `/dataset/MOSIDataset.py` and `/dataset/MVSADataset.py` with the actual path on your system. 

For more detailed training configurations, please see `Section 4.2` of the Paper and `/bash/run.sh`. 

For Five datasets, 

```
bash /bash/run.sh
```


## Test 

For Five datasets, 

```
bash /benchmark/test.sh
```


## Contact us 

If you have any detailed questions or suggestions, you can email us: fivemeng3@gmail.com or shicaiwei@uestc.edu.cn.










