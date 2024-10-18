# Handwritten Digit Recognition using PyTorch

This project trains a DL model on the MNIST dataset to recognize handwritten digits.

## Installation
```python
pip install -r requirements.txt
```
# ***Training & Evaluation***
```
📂 src
├── 📂 checkpoints
│   ├── 📂 checkpoint_1
│   └── 📂 checkpoint_2
├── 📂 data
│   ├── 📂 MNIST
│   └── 📄 dataset.py
├── 📂 models
│   └── 📄 model.py
├── 📂 test_images
│   ├── 📄 00.png
│   ├── 📄 01.png
│   └── ...
├── 📂 training
│   ├── 📄 evaluation.py
│   └── 📄 train.py
├── 📂 utils
│   └── 📄 utils.py
├── 📄 main.py
└── 📄 mnist.ipynb
```
```python
python main.py
```
# ***Deploy***
## API
```
📂 api_app
├── 📂 models
│   └── 📄 model.py
├── 📂 utils
│   ├── 📄 preprocess.py
│   └── 📄 utils.py
├── 📄 api.py
├── 📄 inference.py
└── 📄 run.py
```
```python
python run.py
```
## Streamlit
```
📂 streamlit_app
├── 📂 models
│   └── 📄 model.py
├── 📂 utils
│   ├── 📄 predict.py
│   ├── 📄 preprocess.py
│   └── 📄 utils.py
└── 📄 streamlit_app.py
```
```python
streamlit run streamlit_app/streamlit_app.py
```
