import os
import argparse
import tempfile
import onnx
from huggingface_hub import HfApi
from huggingface_hub import snapshot_download
from huggingface_hub import hf_api, hf_hub_download
from tqdm import tqdm

def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
    else:
        print("file path has existed")

parser = argparse.ArgumentParser(description='Model Crawler')
parser.add_argument('--model_dir', type=str, default='/home/models',
                    help='the path to store the downloaded model files')
parser.add_argument('--num', default=10, type=int,
                    help='the number of downloaded models, if this option is 0, all models will be downloaded')
parser.add_argument('--tasks', default='', type=str,
                    help="the task type of downloaded models, if this option is not set, all models will be downloaded")                   


# here is just for demo, if needs more HF tasks to handle, please feel free to add them into the below list 
HF_tasks=["image-classification",
          "object-detection",
          "automatic-speech-recognition",
          "text-generation"]


def main():
    args = parser.parse_args()

    if not args.model_dir:
        print("please input the valid model dir oath")
    else:
        mkdir(args.model_dir)

    torch_list=[]   
    onnx_list=[]
        
    api = HfApi()
    if not args.tasks: 
        for task in HF_tasks:
            torch_models = api.list_models(
                task=task,
                library="pytorch"
            )
            torch_models = list(torch_models)
            torch_list.append(torch_models)
            
            onnx_models = api.list_models(
                task=task,
                library="onnx"
            )
            onnx_models = list(onnx_models)
            onnx_list.append(onnx_models)
    else:
        torch_models = api.list_models(
                task=args.tasks,
                library="pytorch"
        )
        torch_models = list(torch_models)

        onnx_models = api.list_models(
                task=args.tasks,
                library="onnx"
            )
        onnx_models = list(onnx_models)

    #print(len(torch_list[0]))
    #print(torch_list[0][0].modelId)
    #print(len(onnx_list[0]))
    #print(onnx_list[0][0].modelId)

    if not args.tasks: 
        for (task, torch_models, onnx_models) in zip(HF_tasks,torch_list,onnx_list):
            torch_path = os.path.join(args.model_dir,task,"torch")
            mkdir(torch_path)
            onnx_path = os.path.join(args.model_dir,task,"onnx")
            mkdir(onnx_path)
            if args.num == 0:
                for i in range(len(torch_models)):
                     model_path = os.path.join(torch_path,torch_models[i].modelId)
                     mkdir(model_path)
                     snapshot_download(repo_id = torch_models[i].modelId, local_dir=model_path)
                for i in range(len(onnx_models)):
                     model_path = os.path.join(onnx_path,onnx_models[i].modelId)
                     mkdir(model_path)
                     snapshot_download(repo_id = onnx_models[i].modelId, local_dir=model_path)
            else:
                for i in range(min(args.num,len(torch_models))):
                     model_path = os.path.join(torch_path,torch_models[i].modelId)
                     mkdir(model_path)
                     snapshot_download(repo_id = torch_models[i].modelId, local_dir=model_path)
                for i in range(min(args.num,len(onnx_models))):
                     model_path = os.path.join(onnx_path,onnx_models[i].modelId)
                     mkdir(model_path)
                     snapshot_download(repo_id = onnx_models[i].modelId, local_dir=model_path)
        
    else:
        torch_path = os.path.join(args.model_dir,args.tasks,"torch")
        mkdir(torch_path)
        onnx_path = os.path.join(args.model_dir,args.tasks,"onnx")
        mkdir(onnx_path)
       
        if args.num == 0:
                for i in range(len(torch_models)):
                     model_path = os.path.join(torch_path,torch_models[i].modelId)
                     mkdir(model_path)
                     snapshot_download(repo_id = torch_models[i].modelId, local_dir=model_path)
                for i in range(len(onnx_models)):
                     model_path = os.path.join(onnx_path,onnx_models[i].modelId)
                     mkdir(model_path)
                     snapshot_download(repo_id = onnx_models[i].modelId, local_dir=model_path)
        else:
                for i in range(min(args.num,len(torch_models))):
                     model_path = os.path.join(torch_path,torch_models[i].modelId)
                     mkdir(model_path)
                     snapshot_download(repo_id = torch_models[i].modelId, local_dir=torch_path)
                for i in range(min(args.num,len(onnx_models))):
                     model_path = os.path.join(onnx_path,onnx_models[i].modelId)
                     mkdir(model_path)
                     snapshot_download(repo_id = onnx_models[i].modelId, local_dir=onnx_path)

if __name__ == "__main__":
    main()