import os
import numpy as np

def LabelCreate(path1):
    files1 = os.listdir(path1)
    num1 = len(files1)
    num2 = []
    for i in range(num1):
        path2 = path1 + '//' + files1[i]
        files2 = os.listdir(path2)
        num2.append(len(files2))

    print("all files name:")
    print(files1)
    print("every file numbers:")
    print(num2)
    print(len(num2))



    for i in range(len(num2)):
        if i == 0:
            labels = np.full((num2[i]), 1)
        else:
            label = np.full((num2[i]),i+1)
            labels = np.concatenate((labels,label))
    xinhua = dict(zip(files1, num2))

    valueall = 0

    for key, value in xinhua.items():
        #print('{key}:{value}'.format(key=key, value=value))
        new_value = int(value)
        valueall += new_value
    return labels,valueall