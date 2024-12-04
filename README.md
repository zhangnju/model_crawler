# model_crawler

Verified docker: rocm/pytorch:rocm6.2.3_ubuntu22.04_py3.10_pytorch_release_2.3.0 

Enviroment settings: export HF_TOKEN= "your HF token"

Steps: 
1) pip install -U huggingface_hub onnx transformers

2) only download 10 models per model type: python3 model_crawler.py --model_dir="the path to store model files" --num=10

3) download all models: python3 model_crawler.py --model_dir="the path to store model files" --num=0

4) download new task models ,which is defined by HF: python3 model_crawler.py --model_dir="the path to store model files" --num=10 --tasks="task name", if you need to add the task model permanently, you could change the codes derectly 
          