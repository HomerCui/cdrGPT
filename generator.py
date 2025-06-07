import torch
import gpt_model as gpt
import os
# import gpt_function as func
import try_m as func
import time

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"device is {device}")

model_path = "./use_model"
if os.path.exists(model_path):  # 检测是否存在可以加载模型的目录
    print(f"检测到模型加载目录{model_path}")
else:
    print(f"{model_path} 目录不存在，正在为你创建该目录，请将推理需要的模型放在该目录下......")
    os.makedirs(model_path)


def get_model(path=model_path):
    model_file = None
    for model_file in os.listdir(path):
        # 获取模型名称
        print(f"检测到可加载模型：{model_file}")
    return model_file


"""
my_gpt = gpt.GPT()
model_file = get_model(path=model_path)
my_gpt.load_state_dict(torch.load(f"{model_path}/{model_file}"))
my_gpt.eval()
"""

# 定义并加载模型
my_gpt = gpt.GPT().to(device)
model_file = get_model(path=model_path)
my_gpt.load_state_dict(torch.load(f"{model_path}/{model_file}"))
my_gpt.eval()
print("模型加载成功")

n = 1
s_time = time.time()
for i in range(50000):
    inputs = 'SW'
    s2 = func.rr_smiles(my_gpt, inputs, temperature=1.0)
    print(f"{n}: {s2}")
    n = n + 1
    with open('./5w_FT_03.txt', 'a') as file:
       file.write(s2 + '\n')
e_time = time.time()
print(f"end time: {e_time - s_time}")


