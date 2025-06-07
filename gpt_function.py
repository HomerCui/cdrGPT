import torch
import data_processing as dp
import random

device = 'cuda' if torch.cuda.is_available() else 'cpu'
start_char = [1]  # 设置生成口令<sos>的序号“1”，模型接收指令后开始随机生成
dic_0, dic_c = dp.save_load_dictionary(name_1="dic_1_c.json", name_c="dic_c_1.json", model="load")


def rand_smiles(model, start=None, dictionary=dic_0, grab="random"):  # 生成随机分子
    """
    通过自回归生成一个分子的smiles
    :param grab: grab只有None、random两种模式，且默认为random
    :param model: 传入的模型（gpt）
    :param start: 默认生成指令（开始符号）<sos>（1）
    :param dictionary: 数字序号转字母和分子符号的字典
    :return: smiles字符串
    """
    if start is None:
        start = [1]
    max_len = 200  # 设置了最大生成数，一旦生成的字符串超过这个数值，生成就会被强制终止
    dictionary = dictionary
    output = []  # 准备了一个空列表用来装生成的数据

    for i in range(max_len):

        start_char = start
        start_char_tensor = torch.LongTensor(start_char).unsqueeze(0)  # 改变形状使模型可以接收[1, 1](模仿[batch_size, smiles_len])
        predict= model(start_char_tensor.to(device))

        if grab == "None":
            # 如果抓取方式是None，每次都抓取预测概率最大的
            predict = predict.max(dim=-1)[1]  # 这行代码可以换成其他搜索模式，比如随机或者树搜索，增加结果随机性
        elif grab == "random":
            # 如果是随机模式
            if len(output) <= 15:  # 至少生成长度
                # 如果是前几个字符，那么不能出现终止符
                judge = True  # 设置循环判断条件
                while judge:
                    # 由于judge是ture，会先执行一次循环体，如果循环体内满足条件就把judge设置成false
                    rand_pre_list = predict[-1].cpu().detach().numpy().tolist()
                    sorted_pre_list = sorted(rand_pre_list, reverse=True)

                    rn = 1
                    if len(output) <= 2:
                        rn = 6
                    elif 3 < len(output) <= 5:
                        rn = 3
                    else:
                        rn = 2

                    most_likely_list = sorted_pre_list[:rn]  # 这里的数字确定了选取可能性最大的前几个
                    rand_choice = random.choice(most_likely_list)
                    pre = rand_pre_list.index(rand_choice)  # 预测出来的数

                    judge = False  # 假设pre不是终止符，先给出循环中断的条件，如果后续的判断认为输出是终止符，那么条件将被剥夺

                    if pre == 3:  # 如果输出的真的是终止符，那么judge设置为ture，循环继续
                        judge = True

                predict = predict.max(dim=-1)[1]
                predict_cut1 = predict[:-1]
                pre = torch.Tensor([pre])
                pre = pre.to(torch.int64)
                predict = torch.cat((predict_cut1.to(device), pre.to(device)), dim=0)

            else:
                predict = predict.max(dim=-1)[1]  # 前几个字符已经提供了足够的随机性，后续只取概率最大

        get_next = predict[-1].item()  # 将预测得到的字符储存，下一步送入输出和自回归部分中

        if len(output) <= 15:
            # 分子太小就不终止
            pass
        else:
            # 分子满足一定长度后，出现终止符就停止生成
            if get_next == 3:
                break

        start_char.append(get_next)  # 将本次预测得到的数值进行自回归
        output.append(get_next)

    output = torch.LongTensor(output)
    smiles = dp.output_to_smiles(label_in=output, dictionary=dictionary)

    start.clear()

    return smiles


def create_smiles(model, inputs, start=None, dictionary=dic_0, dictionary_c_0=dic_c):  # 生成随机分子

    if start is None:
        start = [1]
    max_len = 200  # 设置了最大生成数，一旦生成的字符串超过这个数值，生成就会被强制终止
    dictionary = dictionary
    output = []  # 准备了一个空列表用来装生成的数据
    smiles_str = inputs
    start = start

    for i in range(len(smiles_str)):

        if smiles_str[i] in dictionary_c_0:
            i = smiles_str[i]
            temp = dictionary_c_0[i]
        else:
            temp = 2
        start.append(temp)
        output.append(temp)

    for i in range(max_len):

        start_char = start
        start_char_tensor = torch.LongTensor(start_char).unsqueeze(0)  # 改变形状使模型可以接收[1, 1](模仿[batch_size, smiles_len])
        predict= model(start_char_tensor)

        if len(output) > 30:
            predict = predict.max(dim=-1)[1]  # 这行代码可以换成其他搜索模式，比如随机或者树搜索，增加结果随机性
        else:
            judge = True  # 设置循环判断条件
            while judge:
                # 由于judge是ture，会先执行一次循环体，如果循环体内满足条件就把judge设置成false
                rand_pre_list = predict[-1].cpu().detach().numpy().tolist()
                sorted_pre_list = sorted(rand_pre_list, reverse=True)

                most_likely_list = sorted_pre_list[:3]  # 这里的数字确定了选取可能性最大的前几个
                rand_choice = random.choice(most_likely_list)
                pre = rand_pre_list.index(rand_choice)  # 预测出来的数

                judge = False  # 假设pre不是终止符，先给出循环中断的条件，如果后续的判断认为输出是终止符，那么条件将被剥夺

                if pre == 3:  # 如果输出的真的是终止符，那么judge设置为ture，循环继续
                    judge = True

            predict = predict.max(dim=-1)[1]
            predict_cut1 = predict[:-1]
            pre = torch.Tensor([pre])
            pre = pre.to(torch.int64)
            predict = torch.cat((predict_cut1.to(device), pre.to(device)), dim=0)

        get_next = predict[-1].item()

        if get_next == 3:
            #  遇到终止符<eos>自动终止
            break

        start_char.append(get_next)  # 将本次预测得到的数值进行自回归
        output.append(get_next)

    output = torch.LongTensor(output)
    smiles = dp.output_to_smiles(label_in=output, dictionary=dictionary)

    start.clear()

    return smiles
