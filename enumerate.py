# enumerate 函数是 Python 内置的一个函数，
# 用于在遍历可迭代对象时生成一个计数器。它通常用于在循环中需要既获取元素值又获取元素索引的情况。
# enumerate(iterable, start=0)
# start：计数的起始值，默认从 0 开始。
# enumerate 返回一个迭代器，其内容是一个包含计数和对应元素值的元组。
list1 = ['a', 'b', 'c']
for index, value in enumerate(list1, start=1):
    print(index, value)



