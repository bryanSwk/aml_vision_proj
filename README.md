# CS610 Applied Machine Learning - Group 7 Project Submission

## Group Members

| Name               | Email Address                                |
|--------------------|----------------------------------------------|
| Bryan Siow Wei Kang | bryan.siow.2024@mitb.smu.edu.sg             |
| Li Gen              | gen.li.2024@mitb.smu.edu.sg                  |
| Poon Yu Hui         | yuhui.poon.2024@mitb.smu.edu.sg              |
| See Mei Fen         | meifen.see.2024@mitb.smu.edu.sg              |
| Soh Chee Wei        | cheewei.soh.2024@mitb.smu.edu.sg             |


## How to Run

Follow these steps to set up and run the project:

### 1. Clone the Repository

```
git clone https://github.com/bryanSwk/aml_vision_proj.git
cd aml_vision_proj
```

### 2. Install dependencies

```
pip install -r requirements.txt
```

### 3. Run Inference

```
python main.py \
  --source 0 \
  --model cnn \
  --mode mask \
  --output results/output_cnn_mask.mp4
```

### 4. Run Training and Evaluation

To reproduce training experiments or explore the model development process, refer to the Jupyter notebooks in the [`train/`](./train) directory:

- [`cs610_proj_yolov11s.ipynb`](./train/cs610_proj_yolov11s.ipynb) – YOLOv11 training and evaluation
- [`grounding_dino.ipynb`](./train/grounding_dino.ipynb) – GroundingDINO experiments
- [`rt_detrv2_training.ipynb`](./train/rt_detrv2_training.ipynb) – RT-DETRv2 training


### 5. Download Pretrained Model Weights

Pretrained model weights are available on Google Drive:

**[Download Model Weights](https://drive.google.com/drive/folders/1oqTcqCTsU_Nx0DYgjIq0pncZY9O-u_lj?usp=sharing)**
