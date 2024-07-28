# zip 函数是 Python 内置的一个函数，
# 用于将多个可迭代对象（如列表、元组等）打包成一个迭代器，生成对应元素的元组
# zip(*iterables)  iterables：可以是任意数量的可迭代对象。
# zip 函数返回一个迭代器，其内容是由传入的可迭代对象的元素配对而成的元组。
# 当可迭代对象的长度不同时，zip 函数会以最短的可迭代对象为准，截断其他可迭代对象。
list1 = [[1,2,3], [4,5,6], [7,8,9]]
list2 = [['a', 'b', 'c'], ['d', 'e', 'f'], ['g', 'h', 'i']]
zipped = zip(list1, list2)
print(zipped)  # 输出: [(1, 'a'), (2, 'b'), (3, 'c')]
# for i in zipped:
#     print(i)
for (row_axes, row_matrices) in zipped:
    print(row_axes,row_matrices)
    for (ax, matrix) in zip(row_axes, row_matrices):
        print(ax,matrix)


