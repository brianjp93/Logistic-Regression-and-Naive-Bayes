from __future__ import division, print_function
import sys
import csv
from math import sqrt, exp


class Logistic():

    def __init__(self, train=None, test=None, eta=.01, sigma=20, model=None):
        self.headers = None
        self.train = self.getTrainingSet(train) if train is not None else None
        self.test = self.getTestSet(test) if test is not None else None
        self.eta = eta
        self.lam = 1 / (sigma**2)
        self.model = model
        self.num = len(self.headers) - 1
        self.w = [0] * self.num

    def getTrainingSet(self, train):
        trainset = []
        with open(train, "r") as f:
            reader = csv.reader(f)
            for row in reader:
                trainset.append(row)
        self.headers = trainset[0][:-1] + ["bias"] + [trainset[0][-1]]
        trainset = [t[:-1] + [1] + [t[-1]] for t in trainset]
        return trainset[1:]

    def getTestSet(self, test):
        testset = []
        with open(test, "r") as f:
            reader = csv.reader(f)
            for row in reader:
                testset.append(row)
        testset = [t[:-1] + [1] + [t[-1]] for t in testset]
        testset[0][-2] = "bias"
        return testset[1:]

    def getP1(self, sample):
        wx = sum([self.w[i] * x for i, x in enumerate(sample)])
        # print("wx: {}".format(wx))
        return (exp(wx)/(1 + exp(wx)))

    def getDelta(self, dataset):
        delta = [0] * len(self.w)
        for sample in dataset:
            y = int(sample[-1])
            sample = [int(x) for x in sample[:-1]]
            p1 = self.getP1(sample)
            # print(p1)
            term2 = y - p1
            # print(term2)
            delta = [delta[i] + (x * term2 * self.eta) for i, x in enumerate(sample)]
        return delta

    def getMagnitude(self, grad):
        return sqrt(sum([g**2 for g in grad]))

    def learn(self, dataset):
        while True:
            delta = self.getDelta(dataset)
            if not delta:
                continue
            # print(delta)
            mag = self.getMagnitude(delta)
            print(mag)
            if mag < 1:
                return self.w
            for i in range(len(self.w)):
                self.w[i] += delta[i] - (self.eta * self.lam * self.w[i])
            # print(self.w)
            # input()
            # print(self.w)

    def testData(self, dataset):
        correct = 0
        for sample in dataset:
            sample = [int(x) for x in sample]
            noclass = sample[:-1]
            y = sample[-1]
            p1 = self.getP1(noclass)
            print(p1)
            # p1 = self.yhat(sample)
            # print(self.yhat(sample))
            # print(p1)
            if p1 >= .5 and y == 1:
                correct += 1
            elif p1 < .5 and y == 0:
                correct += 1
        print("Got {}".format(correct/len(dataset)))

    def writeModel(self):
        with open(self.model, "w") as f:
            f.write("{}\n".format(self.w[-1]))
            for i, w in enumerate(self.w[:-1]):
                f.write("{}{}{}\n".format(self.headers[i], " " * (30 - len(self.headers[i])), w))


def main():
    train = sys.argv[1]
    test = sys.argv[2]
    eta = float(sys.argv[3])
    sigma = float(sys.argv[4])
    model = sys.argv[5]
    l = Logistic(train, test, eta, sigma, model)
    # print(l.headers)
    l.learn(l.train)
    l.testData(l.test)
    pos = 0
    for sample in l.train:
        if sample[-1] == "1":
            pos += 1
    print(pos, len(l.train))
    l.writeModel()

if __name__ == '__main__':
    main()
