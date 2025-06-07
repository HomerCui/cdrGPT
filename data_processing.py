import os
import random
import json
import torch
import torch.nn.functional as f

"""
使用说明：
    该程序包含了数据处理用到的方法，可以直接import调用

数据处理顺序：
    1. 将指定目录的训练集剪切成小数据集方便处理
    2. 生成字典
    3. 生成dataset
    4. 生成dataloader

实例：
    import data_processing as dp

    cut = dp.cut_text("./training_data", 20000)
    dic_0, dic_c = dp.make_dictionary(cut)
    data = dp.make_dataset(cut, "6.smi.cut.0.smi", shuffle=True)
    dataiter = dp.dataloader(data, 32, dic_c, 20)
"""

device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 512


def cut_text(data_path, large):
    """
    该函数作用是将指定文件夹中的文件切分成小份的文本文件，方便程序调用，加快加载速度
    裁剪后的文件放在dataset文件夹中
    :param data_path: 存储未切割训练集的位置
    :param large: 设置每个小文件的最大行数
    :return: 存储切分后数据集的路径
    """
    large = large  # 每个小文件的最大行数
    data_path = data_path
    new_path = "./dataset"
    # 判断存放数据的文件夹是否存在，不存在就创建一个并报错
    if os.path.exists(data_path):
        # 如果存在就开始剪切文件
        print("please wait......")
        # 如果接收小文件的目录不存在则创建
        if os.path.exists(new_path):
            print("new path exist")
        else:
            print("creating new path")
            os.makedirs(new_path)
        # 遍历\
        for file_name in os.listdir(data_path):
            print("cut ", file_name)
            with open(f"{data_path}/{file_name}", 'r', encoding='utf-8') as input_f:  # 此处注意，由于默认读取相对路径，所以需要手动补上路径位置
                count = 1
                n = 0
                for line in input_f:
                    if count <= large:
                        with open(f"{new_path}/{file_name}.cut.{n}.smi", 'a', encoding='utf-8') as output_f:
                            output_f.write(line)
                            count += 1
                    else:
                        count = 0
                        n += 1
                        with open(f"{new_path}/{file_name}.cut.{n}.smi", 'a', encoding='utf-8') as output_f:
                            output_f.write(line)
                            count += 1
                print(f"{new_path}/{file_name}.cut.{n}.smi")
            os.remove(f"{data_path}/{file_name}")
    else:
        os.makedirs(data_path)
        print("wrong: dir does not exist!")
    print("DONE")
    return new_path


def make_dictionary(data_path):
    """
    data_path 是存放切割后小数据集的目录
    返回两个字典用于输入和输出
    return: char_list, char_list_flipped
    """
    data_path = data_path
    # 预设标记符
    char_count = {'<pad>': 0, '<sos>': 1, '<unk>': 2, '<eos>': 3, '<sep>': 4, '<mask>': 5, '<bos>': 6}
    for filename in os.listdir(data_path):
        print("creating dictionary: ", filename)
        with open(f"{data_path}/{filename}", 'r') as file:
            # 逐个字符读取文件
            while True:
                char = file.read(1)
                if char in char_count:
                    # 如果该字符出现过，就不统计
                    pass
                else:
                    # 如果没出现过，放入字典
                    char_count[char] = 1
                if not char:
                    break  # 达到文件末尾时退出循环
        # 遍历字库，生成一个编号为键名，字符为键值的字典
    char_list = {}
    num_of_char = 0
    for i in char_count.keys():
        char_list[num_of_char] = i
        num_of_char += 1
    char_list_flipped = {v: k for k, v in char_list.items()}
    """
    char_list字典=序号：字母           0: 'N', 1: '(', 2: 'C'
    char_list_flipped=字母：序号    'N': 0, '(': 1, 'C': 2
    """
    return char_list, char_list_flipped


def get_smiles(path, filename, row):
    """
    输入文件名，地址，行数，得到该行的smiles
    :param path: 存储训练集的目录
    :param filename: 要读取的训练集名
    :param row: 指定的行数，通过外面的程序遍历得到
    :return: 一行字符串，内容是smiles
    """
    path = path
    filename = filename
    row = row
    with open(f"{path}/{filename}", 'r', encoding='utf-8') as file:
        lines = file.readlines()
        smiles = lines[row]
    return smiles.strip()  # 可以修改为 smiles.stripe()


def get_smiles_serial(smiles, dictionary):
    """
    输入训练集里的smiles字符串，把字符换成对应的序号
    :param smiles: smiles字符串
    :param dictionary:
    :return: 序号（tensor）如tensor([7., 7., 8., 7., 7., 7., 8., 7.])
    """
    char_list_flipped = dictionary
    smiles_str = "".join(smiles)  # 将ndarray连成字符串
    smiles_serial = []
    # 遍历得到的字符串
    for i in smiles_str:
        # 此处i储存着字符串里的字符（单个）
        # 将字符串的字符逐一对应字典，变成序号，为转变为embedding做准备
        if i in char_list_flipped:
            temp = char_list_flipped[i]
        else:
            temp = 2  # 2是<unk>
        smiles_serial.append(temp)
    return torch.LongTensor(smiles_serial)[:-1]  # 这里需要测试一下去掉最后一位是去掉的什么


def padding(smiles, size):
    """
    用于padding
    :param smiles: tensor, smiles转换来的序号
    :param size: 统一padding的尺寸
    :return: padding后的tensor
    """
    smiles = smiles
    size = size
    pad = f.pad(smiles, (0, (size - len(smiles))), "constant", 0)
    return pad


def output_to_smiles(label_in, dictionary):
    """
    将序号（输出）转化成smiles
    :param label_in: tensor，对应smiles的序号
    :param dictionary: char_list字典=序号：字母  0: 'N', 1: '(', 2: 'C'
    :return: 字符串，分子的smiles
    """
    label_smiles = []
    dictionary = dictionary
    for i in (label_in.numpy()):
        i = f'{i}'
        label_smiles.append(dictionary[i])
    label_smiles = "".join(label_smiles)
    return label_smiles


def sos_eos(smiles):
    """
    将分子的开头和结尾处加入标记指明开头和结尾的位置
    :param smiles:
    :return:
    """
    dic = [1]
    for i in range(len(smiles)):
        dic.append(int(smiles[i]))
    dic.append(3)
    return torch.LongTensor(dic)


def get_input(smiles):
    """
    去除最后一位生成input
    :param smiles: 数组
    :return:
    """
    return smiles[:-1]


def get_label(smiles):
    """
    去除第一位生成label
    :param smiles: 数组
    :return:
    """
    return smiles[1:]


def make_dataset(data_path, filename, shuffle):
    """
    准备训练的数据集，可选择是否打乱顺序
    :param data_path: 准备处理的分割后数据集目录
    :param filename: 准备处理的分割后数据集
    :param shuffle: 是否打乱。TRUE是打乱
    :return: 返回处理后文件的位置，方便dataloader处理
    """
    data_path = data_path
    filename = filename
    new_path = "./tem_operate"

    if os.path.exists(data_path):
        # 如果数据集目录存在就继续
        if os.path.exists(new_path):
            # 如果临时目录存在就继续，跳过创建新目录这一步
            if os.path.exists(f"{data_path}/{filename}"):
                # 如果要处理的文件存在就跳过，否则报错
                pass
            else:
                print("wrong: no data!")
        else:
            os.makedirs(new_path)
    else:
        print("wrong: no data dir called 'dataset' !")

    # 开始制作dataset的临时文件
    with open(f"{data_path}/{filename}", 'r') as file:
        lines = file.readlines()

    if shuffle:
        # 如果设置了打乱，则数据集将被打乱后进行训练
        random.shuffle(lines)
    else:
        pass

    with open(f"{new_path}/{filename}.tem", 'w') as file:
        for line in lines:
            file.write(line.strip() + '\n')

    return f"{new_path}/{filename}.tem"


def dataloader(dataset, batch_size, dictionary, padding_size):
    """
    通过dataset建立的临时文件，将smiles字符串按bitch_size的大小读入变量并且转换序号，添加标记，padding，最后创建一个可迭代对象
    下一步直接可以embedding了
    :param dataset: 函数make_dataset的返回值，当前要使用的小数据集
    :param batch_size: 同时输入网络的数据量（每次读取多少个分子进行训练）
    :param dictionary: 将smiles字符串翻译成序号的字典
    :param padding_size: padding大小
    :return: 返回两个tensor作为可迭代对象，每次可以读出batch size个分子，一个是input，一个是label
    """
    dataset = dataset
    batch_size = batch_size
    dictionary = dictionary
    size = padding_size + 1

    with open(dataset, 'r') as file:
        lines = file.readlines()

    if len(lines) % batch_size != 0:
        # 判断内容是否能被完整分割，不能则复制原有部分增加行数使之满足条件
        add_lines = batch_size - (len(lines) % batch_size)
        to_add = lines[:add_lines]

        with open(dataset, 'a') as file:
            for line in to_add:
                file.write(line)

        with open(dataset, 'r') as file:
            lines = file.readlines()

    # 创建可迭代对象，用嵌套的列表实现
    temp = []
    with open(dataset, 'r') as file:
        for line in file:
            smiles = get_smiles_serial(line, dictionary)  # 得到了一行的smiles序号
            # 此处应当考虑后续其他任务的训练，加入判断，如果由其他任务就调用一个写好的函数，将返回值与smiles拼接
            mark = sos_eos(smiles)  # 加入<eos><sos>标记
            pad = padding(mark, size).tolist()  # 进行padding
            # pad: 格式list
            temp.append(pad)  # 把处理好的数据放到一个临时列表里，用来准备制作嵌套列表

    temp_input = []
    temp_label = []
    for i in temp:
        temp_input.append(get_input(i))
        temp_label.append(get_label(i))

    temp_batch_input = []  # 每个装batch size个list
    dataiter_input = []  # 最后返回的嵌套列表，每个元素是一个装有batch size个list的列表

    for i in range(int(len(lines) / batch_size)):
        temp_batch_input.append(temp_input[:batch_size])
        del temp_input[:batch_size]
        dataiter_input.append(temp_batch_input[0])
        temp_batch_input = []

    temp_batch_label = []  # 每个装batch size个list
    dataiter_label = []  # 最后返回的嵌套列表，每个元素是一个装有batch size个list的列表

    for i in range(int(len(lines) / batch_size)):
        temp_batch_label.append(temp_label[:batch_size])
        del temp_label[:batch_size]
        dataiter_label.append(temp_batch_label[0])
        temp_batch_label = []

    os.remove(dataset)  # 删除临时文件，节省内存

    return torch.LongTensor(dataiter_input), torch.LongTensor(dataiter_label)


def save_load_dictionary(dic_1_c=None, dic_c_1=None, name_1="", name_c="", model=""):
    """
    保存或导入字典，
    如果是save模式，就接收两个字典的参数用于保存；
    如果是load模式，就接收字典的位置用于载入字典并返回两个可以调用字典的变量
    :param dic_1_c:序号转换字母的字典
    :param dic_c_1:字母转序号的字典
    :param name_1:序号转换字母的字典的文件名
    :param name_c:字母转序号的字典的文件名
    :param model:字符串，分save和load两种模式
    :return:char_list（序号转换字母的字典）, char_list_flipped（字母转序号的字典）
    """
    char_list, char_list_flipped = "", ""  # 先准备两个空值避免返回时报错
    dic_1_c = dic_1_c
    dic_c_1 = dic_c_1
    path_1 = name_1
    path_c = name_c

    s_l_path = "./dictionary"

    if model == "save":
        # 如果需要保存，就需要用到前两个参数：dic_1_c, dic_c_1，和默认保存地址“./dictionary/”
        if os.path.exists(s_l_path):
            # 如果路径存在就开始保存字典
            pass
        else:
            os.makedirs(s_l_path)

        with open(f"{s_l_path}/dic_1_c.json", 'w') as file:
            json.dump(dic_1_c, file)

        with open(f"{s_l_path}/dic_c_1.json", 'w') as file:
            json.dump(dic_c_1, file)

    elif model == "load":
        # 如果是load模式，则需要path_1="", path_c=""两个参数，并返回两个可以调用字典的变量
        if os.path.exists(s_l_path):
            pass
        else:
            print("wrong: no dictionary can be load!")

        with open(f"{s_l_path}/{path_1}", 'r') as file:
            char_list = json.load(file)

        with open(f"{s_l_path}/{path_c}", 'r') as file:
            char_list_flipped = json.load(file)

    else:
        # 如果不是save和load就报错
        print("wrong: model only can be 'save' or 'load'! ")

    return char_list, char_list_flipped


def first_time_or_not():

    # 要检查的地址
    data_path = "./training_data"
    dataset_path = "./dataset"
    dictionary_path = "./dictionary"

    # 一个数组表示状态，0-5位分别表示0：training_data，1：training_data的文件， 2：dataset， 3.dataset文件
    # 4.dictionary， 5.字典文件
    # 如果状态是1就说明正确，0代表缺失
    state = [0, 0, 0, 0, 0, 0]

    # 判断training_data
    if os.path.exists(data_path):
        print("training_data文件夹存在")
        state[0] = 1
    else:
        print("training_data文件夹不存在！建议终止程序！！！")

    # 判断training_data文件
    if len(os.listdir(data_path)) != 0:
        print("存在文件")
        state[1] = 1
    else:
        print("training_data文件不存在！建议终止程序！！！")

    # 判断dataset
    if os.path.exists(dataset_path):
        state[2] = 1
    else:
        pass

    # dataset文件
    if len(os.listdir(dataset_path)) != 0:
        state[3] = 1
    else:
        pass

    # 字典目录
    if os.path.exists(dictionary_path):
        state[4] = 1
    else:
        pass

    # 字典文件
    if len(os.listdir(dictionary_path)) != 0:
        state[5] = 1
    else:
        pass

    return state


def add_file(file_dir, out_file_path):
    for file_name in os.listdir(file_dir):  # 遍历要合并的所有文件
        file_path = f"{file_dir}/{file_name}"

        with open(file_path, 'r') as file:  # 打开当前文件
            readall = file.readlines()  # 读取当前文件

        with open(out_file_path, 'a') as out_file:  # 打开结果输出文件
            out_file.writelines(readall)

        os.remove(file_path)  # 删除读取万的文件

    return out_file_path


"""
成于2024.4.6 清明假期 国科大雁栖湖 2公寓
cwr
"""
