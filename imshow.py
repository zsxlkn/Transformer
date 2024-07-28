# ax1.imshow 是 Matplotlib 中的一个方法，用于在 Axes 对象上显示图像数据。
# 它通常用于显示二维数组或图像，并且提供了丰富的参数来定制图像的显示方式。
# ax1.imshow(X, cmap=None, norm=None, aspect=None, interpolation=None,
# alpha=None, vmin=None, vmax=None, origin=None, extent=None, **kwargs)
# X：要显示的数据（二维数组或图像）
# cmap：颜色映射（colormap），用于将数据值映射到颜色。可以是一个字符串（如 'viridis',
# 'plasma', 'inferno', 'magma', 'cividis', 'Greys',
# 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds', 'YlOrBr', 'YlOrRd',
# 'OrRd', 'PuRd', 'RdPu', 'BuPu', 'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn',
# 'BuGn', 'YlGn', 'binary', 'gist_yarg', 'gist_gray', 'gray', 'bone',
# 'pink', 'spring', 'summer', 'autumn', 'winter', 'cool', 'Wistia',
# 'hot', 'afmhot', 'gist_heat', 'copper']) 或者是一个 Colormap 实例。
# norm：用于缩放数据到 [0, 1] 的 Normalize 实例。如果未提供，默认使用线性缩放。


import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # 使用 'Agg' 后端进行非GUI渲染
data = np.random.rand(10, 10)
print(data)
for i in range(10):
    for j in range(10):
        data[i][j] = i*10+j
fig, ax = plt.subplots()
cax = ax.imshow(data, cmap='viridis')
# 添加颜色条
fig.colorbar(cax)
plt.show()
