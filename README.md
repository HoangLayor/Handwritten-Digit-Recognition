># Handwritten Digit Recognition using PyTorch
>>This project trains a simple DL model on the MNIST dataset to recognize handwritten digits.

## Installation
```python
pip install -r requirements.txt
```
----
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
__Train & Eval__
```python
python src/main.py
```
----
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
__Run FastAPI__
```python
python api_app/run.py
```
- [127.0.0.1:8000](http//127.0.0.1:8000/docs)
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
__Run streamlit app__
```python
streamlit run streamlit_app/streamlit_app.py
```
- [https://hoanglayor-handwritten-digit--streamlit-appstreamlit-app-jmbim3.streamlit.app/](https://hoanglayor-handwritten-digit--streamlit-appstreamlit-app-jmbim3.streamlit.app/)
