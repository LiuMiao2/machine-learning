import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("./西瓜数据集4.0.csv", index_col='number')
data = data.values.tolist()

# 画出原始图像
fig, ax = plt.subplots()
plt.scatter([i[0] for i in data], [i[1] for i in data])
plt.show()

'''
计算欧氏距离
'''


def calDist(a, b):
    a = np.array(a)
    b = np.array(b)
    dist = np.sqrt(np.dot((a - b), (a - b).T))

    return dist


'''
计算簇之间的最小距离
'''


def calClusterMinDist(c1, c2):
    minDist = 1e5
    for vec1 in c1:
        for vec2 in c2:
            dist = calDist(vec1, vec2)
            if dist < minDist:
                minDist = dist

    return minDist


'''
计算簇之间的平均距离
'''


def calClusterAvgDist(c1, c2):
    num = len(c1) * len(c2)
    sum_dist = 0
    for vec1 in c1:
        for vec2 in c2:
            dist = calDist(vec1, vec2)
            sum_dist += dist
    return sum_dist


'''
获取最小距离列表
'''


def getMinDistList(data, method):
    cluster_num = len(data)
    #     print("cluster_num",cluster_num)
    minDistList = [[0 for i in range(cluster_num)] for j in range(cluster_num)]
    for i in range(cluster_num):
        j = i + 1
        while j < cluster_num:
            #             print("data[i]:",data[i])
            #             print("data[j]:",data[j])
            if method == "minDist":  # 使用最小距离计算
                minDistList[i][j] = calClusterMinDist(data[i], data[j])
                minDistList[j][i] = minDistList[i][j]
            elif method == "avgDist":  # 使用平均距离计算
                minDistList[i][j] = calClusterAvgDist(data[i], data[j])
                minDistList[j][i] = minDistList[i][j]
            j += 1

    return minDistList


'''
寻找距离列表中的最小值，用于合并簇以及删除
'''


def findMin(minDistList):
    row = len(minDistList)
    minDist = 1e5
    min_i = 0
    min_j = 0
    for i in range(row):
        for j in range(row):
            dist = minDistList[i][j]
            if dist < minDist and dist != 0:
                minDist = minDistList[i][j]
                min_i = i
                min_j = j

    return min_i, min_j, minDist


'''
AGNES算法实现
'''


def AGNES(data, k, method):
    cluster_num = len(data)
    C = []
    for i in data:  # 添加数据
        tmp = [i]
        C.append(tmp)
    minDistList = getMinDistList(C, method)
    while cluster_num > k:
        i, j, minDist = findMin(minDistList)
        #         print(len(minDistList))
        #         print(i,j,minDist)
        C[i].extend(C[j])  # 合并
        del C[j]  # 删除
        minDistList = getMinDistList(C, method)
        cluster_num -= 1

    return C


'''
程序入口
'''
if __name__ == "__main__":
    C_min = AGNES(data, 3, 'minDist')
    C_avg = AGNES(data, 3, 'avgDist')
    fig, ax = plt.subplots(nrows=2, ncols=1)
    ax[0].scatter([i[0] for i in C_min[0]], [i[1] for i in C_min[0]], c='r')
    ax[0].scatter([i[0] for i in C_min[1]], [i[1] for i in C_min[1]], c='g')
    ax[0].scatter([i[0] for i in C_min[2]], [i[1] for i in C_min[2]], c='b')
    ax[0].set_title("使用最小距离进行聚类")

    ax[1].scatter([i[0] for i in C_avg[0]], [i[1] for i in C_avg[0]], c='r')
    ax[1].scatter([i[0] for i in C_avg[1]], [i[1] for i in C_avg[1]], c='g')
    ax[1].scatter([i[0] for i in C_avg[2]], [i[1] for i in C_avg[2]], c='b')
    ax[1].set_title("使用平均距离进行聚类")

    fig.tight_layout()
    plt.show()
