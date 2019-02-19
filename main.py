import random
from itertools import islice


def sign(num):
    if num > 0:
        return 1
    elif num < 0:
        return -1
    else:
        return 0


class GradientDescent:

    def __init__(self, lr=1):
        self.a = random.random()
        self.b = random.random()
        self.c = random.random()
        self.lr = lr

    def predict(self, x, y):
        return self.a * x + self.b * y + self.c

    def loss(self, x, y, z):
        return abs(z - self.predict(x, y))

    def gd(self, x, y, z):
        pred = self.predict(x, y)
        factor = sign(z - pred)

        self.a -= self.lr * factor * -x
        self.b -= self.lr * factor * -y
        self.c -= self.lr * factor * -1

    def __str__(self):
        return (f'a: {self.a}, b: {self.b}, '
                f'c: {self.c}, lr: {self.lr}')


def data_generator():
    while True:
        x = random.random() * 100
        y = random.random() * 100

        z = 3*x - 5*y + 1

        yield x, y, z


def normalize(feature):
    return [f/100 for f in feature]


gd = GradientDescent(lr=1)
ds = list(islice(data_generator(), 100))
xl, yl, zl = zip(*ds)
xl = normalize(xl)
yl = normalize(yl)
ds = list(zip(xl, yl, zl))

for _ in range(100):
    for (x, y, z) in ds:
        gd.gd(x, y, z)
        print(gd)
        print(f'loss: {gd.loss(x, y, z)}')





