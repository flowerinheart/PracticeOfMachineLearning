# coding:utf-8
import math as math
import numpy as np
import random


def action_func_1(x):
    return 1 / (1 + math.exp(-x))


def action_func_2(x):
    if x < 0:
        return -1.0
    else:
        return 1.0


class Perceptron:
    def __init__(self):
        self.w = np.array([random.uniform(-1.0, 1.0), random.uniform(-1.0, 1.0), random.uniform(-1.0, 1.0)])
        self.trainset = []
        self.label = []
        self.n = random.uniform(0.0, 1.0)
        self.num = 5000

    def update(self, i):
        """
        [w,b] = [x,1] * y * n
        """
        factor = self.n * self.label[i]
        self.w += factor * self.trainset[i]

    def train(self):
        number = self.num
        trainset = self.trainset
        length = len(trainset)
        w = self.w
        label = self.label

        count = 0

        over = False
        while not over:
            is_change = False
            for i in range(length):
                predicted_kind = action_func_2(w.dot(trainset[i]))
                if predicted_kind != label[i]:
                    self.update(i)
                    is_change = True
                count += 1
                if count > number:
                    over = True
                    break
            if not is_change:
                break

    def test(self):
        error_count = 0
        w = self.w
        trainset = self.trainset
        length = len(trainset)
        label = self.label
        for i in range(length):
            instance = trainset[i]
            predicted_kind = action_func_2(w.dot(instance))
            # print str(i) + " result: " + str(action_func_2(w.dot(instance))) + str(label[i])
            if predicted_kind != label[i]:
                error_count += 1

        #print str(float(error_count) / float(length))
        return str(float(error_count) / float(length))

    def read_from_file(self, filename):
        self.__init__()
        datafile = open(filename)
        for line in datafile:
            chunk = line.strip().split(",")
            self.trainset.append(np.array([float(chunk[0]), float(chunk[1]), float(1.0)]))
            kind = float(chunk[2])
            if kind == 1:
                kind = -1
            else:
                kind = 1
            self.label.append(kind)
        datafile.close()

    def run(self, filename):
        self.read_from_file(filename)
        for i in range(10):
            self.w = np.array([random.uniform(-1.0, 1.0), random.uniform(-1.0, 1.0), random.uniform(-1.0, 1.0)])
            self.n = random.uniform(0.0, 1.0)
            self.train()
            ER = self.test()
            print "%s:第%d次测试, 初始w = %s, 初始学习率=%s, 错误率:%s" % (filename, i, str(self.w), str(self.n), str(ER))


if __name__ == "__main__":
    perceptron = Perceptron()
    perceptron.run("ls.csv")
    perceptron.run("xor.csv")
    #perceptron.read_from_file("ls.csv")
    #perceptron.train()
    #perceptron.test()


