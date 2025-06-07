import torch
import data_processing as dp
import random

device = 'cuda' if torch.cuda.is_available() else 'cpu'
start_char = [1]  # 设置生成口令<sos>的序号“1”，模型接收指令后开始随机生成
dic_0, dic_c = dp.save_load_dictionary(name_1="dic_1_c.json", name_c="dic_c_1.json", model="load")


def rr_smiles(model, inputs, dictionary=dic_0, dictionary_c_0=dic_c, temperature=0.8):
    max_len = 100  # 设置了最大生成数，一旦生成的字符串超过这个数值，生成就会被强制终止
    dictionary = dictionary
    smiles_str = inputs
    start = [1]

    for i in range(len(smiles_str)):

        if smiles_str[i] in dictionary_c_0:
            i = smiles_str[i]
            temp = dictionary_c_0[i]
        else:
            temp = 2
        start.append(temp)
        # output.append(temp)

    start_char = start
    start_char_tensor = torch.LongTensor(start_char).unsqueeze(0)  # 改变形状使模型可以接收[1, 1](模仿[batch_size, smiles_len])
    start_char_tensor = start_char_tensor.to(device)

    for i in range(max_len):
        predict = model(start_char_tensor.to(device))
        predict = predict[-1]
        predict = torch.softmax(predict / temperature, dim=-1)
        predict = torch.multinomial(predict, num_samples=1).unsqueeze(0)

        if predict.item() == 3:
            break

        start_char_tensor = torch.cat([start_char_tensor, predict], dim=-1)

    start_char_tensor = torch.LongTensor(start_char_tensor.tolist()).cpu()

    return dp.output_to_smiles(start_char_tensor[0][1:], dictionary)




