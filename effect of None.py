# None的作用主要是在使用None的位置新增一个维度
import numpy as np
a = np.arange(25).reshape(5, 5)
print(a)
print(a[:, None])
