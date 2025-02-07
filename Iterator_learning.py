class Reverse:
    """Iterator for looping over a sequence backwards."""
    def __init__(self, data):
        self.data = data
        self.index = len(data)

    def __iter__(self):
        return self

    def __next__(self):
        if self.index == 0:
            # self.index = len(self.data) 观察有没有这句代码对第二个for循环输出结果的影响
            raise StopIteration
        self.index = self.index - 1
        return self.data[self.index]

rev = Reverse('spam')
print(iter(rev)==rev.__iter__())
for char in rev:
    print(char)
for char in rev:
    print(char)
# If the class defines __next__(), then __iter__() can just return self
# 执行iter(rev)和rev.__iter__()返回结果相同，都是rev
# 迭代器每次执行__next__()方法，返回一个值
"""
The for statement calls iter(rev),
then the function returns an iterator object that defines the method __next__(),
in this case, the iterator object is rev itself.
When there are no more elements, 
__next__() raises a StopIteration exception which tells the for loop to terminate.
"""
def reverse(data):
    for index in range(len(data)-1, -1, -1):
        yield data[index]

for char in reverse('golf'):
    print(char)

# 函数reverse(data)是一个iterator，yield输出iterator的下一个值
# 因此reverse(data)也可以用于for循环