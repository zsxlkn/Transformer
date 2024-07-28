import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # 使用 'Agg' 后端进行非GUI渲染


def heatmap(data, row_labels, col_labels, ax=None, cbar_kw=None, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.
    Parameters
    ----------
    data
        A 2D numpy array of shape (M, N).
    row_labels
        A list or array of length M with the labels for the rows.
    col_labels
        A list or array of length N with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current Axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if ax is None:
        ax = plt.gca()
# 获取当前轴对象：如果当前的绘图区域中已经有一个轴对象（例如，你已经调用过绘图函数），plt.gca() 会返回这个轴对象。
# 创建新的轴对象：如果当前没有任何轴对象存在，plt.gca() 会创建一个新的轴对象并返回它。
    if cbar_kw is None:
        cbar_kw = {}

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)
    # Show all ticks and label them with the respective list entries.
    ax.set_xticks(np.arange(data.shape[1]), labels=col_labels)
    ax.set_yticks(np.arange(data.shape[0]), labels=row_labels)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # Let the horizontal axes labeling appear on top.
    # ax1.tick_params 是 Matplotlib 中用于自定义坐标轴刻度（包括主刻度和次刻度）的函数。
    # 它允许你设置刻度的样式、方向、长度、颜色等各种属性
    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right", rotation_mode="anchor")

    # # Turn spines off and create white grid.
    # 用于隐藏绘图的坐标轴脊线（spines）。
    # 脊线是指围绕绘图区域的边框线，通常有四条：左（left）、右（right）、上（top）和下（bottom）。
    print(ax.spines)
    ax.spines[:].set_visible(False)
    # ax1.set_xticks 是设置 x 轴刻度位置的函数。
    # minor=True 表示这些刻度是次刻度（minor ticks），
    # 与主要刻度（major ticks）相区别。次刻度通常用于更精细的刻度标记。
    # 这段代码的作用是在 x 轴上设置次刻度线的位置，使这些次刻度线位于每个单元格的中间位置。
    # 设置次刻度线的原因通常是为了在绘制热图或其他网格图时，更清晰地显示单元格边界。
    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)

    # 具体配置了次要网格线（minor grid lines）的颜色、样式和宽度
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}", textcolors=("black", "white"),
                     threshold=None, **textkw):
    # 热图对象（通常是由 ax1.imshow() 创建的图像对象）
    # 用于注释文本颜色的元组。第一个颜色用于低于阈值的数值，第二个颜色用于高于阈值的数值。
    # threshold：决定文本颜色的阈值。如果未提供，将使用数据的最大值的一半。
    # **textkw：传递给 text 函数的其他文本属性关键字参数。
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    # 用于注释的数据。如果为None，则使用图像的数据。
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
        # 热图中注释的格式。
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
        一对颜色。第一个用于低于阈值的值，第二个是上面提到的。
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    # 检查 data 是否是 list 或 np.ndarray 类型。
    # 如果 data 不是这两种类型之一，它会从图像对象 im 中获取数组数据
    # 如果 im 是一个热图或其他图像对象，im.get_array() 会返回用于创建该图像的底层数据数组
    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    # 根据给定的 threshold 值或数据中的最大值来规范化阈值。
    # 这在数据可视化过程中，尤其是在处理图像或热图时，常用于确定某些颜色映射或标注的阈值。
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    # 创建一个字典 kw，包含默认的文本对齐方式
    # 使用 textkw 更新 kw，以便允许用户传递其他文本属性。
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    # 如果 valfmt 是一个字符串，则将其转换为一个格式化函数
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    # 遍历 data 的每个元素，并根据其位置 i 和 j 在热图上添加注释
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            # 键值被更新
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts



if __name__ == "__main__":
    fig, ax = plt.subplots()
    vegetables = ["cucumber", "tomato", "lettuce", "asparagus", "potato", "wheat", "barley"]
    farmers = ["Farmer Joe", "Upland Bros.", "Smith Gardening", "Agrifun", "Organiculture", "BioGoods Ltd.",
               "Cornylee Corp."]
    harvest = np.array([[0.8, 2.4, 2.5, 3.9, 0.0, 4.0, 0.0],
                        [2.4, 0.0, 4.0, 1.0, 2.7, 0.0, 0.0],
                        [1.1, 2.4, 0.8, 4.3, 1.9, 4.4, 0.0],
                        [0.6, 0.0, 0.3, 0.0, 3.1, 0.0, 0.0],
                        [0.7, 1.7, 0.6, 2.6, 2.2, 6.2, 0.0],
                        [1.3, 1.2, 0.0, 0.0, 0.0, 3.2, 5.1],
                        [0.1, 2.0, 0.0, 1.4, 0.0, 1.9, 6.3]])
    im, cbar = heatmap(harvest, vegetables, farmers, ax=ax, cmap="YlGn", cbarlabel="harvest [t/year]")
    texts = annotate_heatmap(im, valfmt="{x:.1f} t", rotation=20)
    fig.tight_layout()
    plt.show()
