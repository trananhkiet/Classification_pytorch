import os

def choose_class_weight(data_path, a):
    list_len = []
    for i in range(len(a)):
        len_class = len(os.listdir(os.path.join(data_path, a[i])))    
        list_len.append(len_class)

    return [sum(list_len)/(list_len[0]*4),sum(list_len)/(list_len[1]*4), sum(list_len)/(list_len[2]*4), sum(list_len)/(list_len[3]*4)]