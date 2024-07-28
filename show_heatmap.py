# 这个函数的作用是以网格形式显示多个矩阵的热图，并根据输入参数设置标签和标题
# Axes 对象是所有绘图元素的容器。它包含了坐标系、刻度、标签、图例、标题等内容。
# Axes 对象提供了丰富的属性和方法，用于定制和操作图表。
# fig 是一个 Figure 对象。Figure 对象是整个图形的容器，
# 而 Axes 对象是图形中的子区，即绘图区域。一个 Figure 对象可以包含多个 Axes 对象
import torch
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('TkAgg')  # 使用 'Agg' 后端进行非GUI渲染


# gridspec_kw：字典类型，用于设置网格属性。
# 例如，gridspec_kw={'wspace': 0.5, 'hspace': 0.5} 用于调整子图之间的间距。
# 输入matrices的形状是 （要显示的行数，要显示的列数，查询的数目，键的数目）
# cmap：颜色映射，默认为 'Reds'
# matrices：要显示的矩阵列表
def show_heatmaps(matrices, xlabel, ylabel, titles=None, figsize=(8, 8), cmap='viridis'):
    """显示矩阵热图"""
    num_rows, num_cols, rows, clos = matrices.shape[0], matrices.shape[1], matrices.shape[2], matrices.shape[3]
    print(num_rows, num_cols)
    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize, sharex=True, sharey=True, squeeze=False)
    # 返回的 Axes 对象始终是包含 Axes 实例的二维数组
    print(axes.shape)
    fig.suptitle("attention map")
    for i, (row_axes, row_matrices) in enumerate(zip(axes, matrices)):
        for j, (ax, matrix) in enumerate(zip(row_axes, row_matrices)):
            pcm = ax.imshow(matrix.detach().numpy(), cmap=cmap)
            if i == num_rows - 1:
                ax.set_xlabel(xlabel)
            if j == 0:
                ax.set_ylabel(ylabel)
                # ax1.set_facecolor('lightyellow')
            if titles:
                ax.set_title(titles[j])
            for n in range(rows):
                for m in range(clos):
                    text = ax.text(m, n, round(float(matrices[i, j, n, m]), 2), ha="center", va="center", color="w",
                                   rotation=0)
    fig.colorbar(pcm, ax=axes, shrink=0.6)
    fig.patch.set_facecolor('lightblue')
    fig.savefig('my_figure.png')


if __name__ == '__main__':
    attention_weights = torch.rand(150)
    attention_weights = attention_weights.reshape((2, 3, 5, 5))
    show_heatmaps(attention_weights, xlabel='Keys', ylabel='Queries')
    plt.show()
