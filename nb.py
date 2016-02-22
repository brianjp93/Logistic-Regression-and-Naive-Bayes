from __future__ import division
import sys
import csv
from math import log


class Nb:

    def __init__(self, train=None, test=None, beta=5, model=None):
        self.headers = None
        self.train = self.getTrainingSet(train) if train is not None else None
        self.test = self.getTestSet(test) if test is not None else None
        self.beta = int(beta)
        self.model = model
        self.num = len(self.headers) - 1
        self.w = [0] * self.num
        self.prob1, self.pos, self.neg = self.countPositives()

    def countPositives(self):
        prob1 = 0
        pos = [self.beta-1] * self.num
        neg = [self.beta-1] * self.num
        for sample in self.train:
            sample = [int(x) for x in sample]
            y = sample[-1]
            if y == 1:
                prob1 += 1
            for i, x in enumerate(sample[:-1]):
                if y == 1 and x == 1:
                    pos[i] += 1
                elif y == 0 and x == 1:
                    neg[i] += 1
        prob1 = prob1/len(self.train)
        return prob1, pos, neg

    def getTrainingSet(self, train):
        trainset = []
        with open(train, "r") as f:
            reader = csv.reader(f)
            for row in reader:
                trainset.append(row)
        self.headers = trainset[0]
        return trainset[1:]

    def getTestSet(self, test):
        testset = []
        with open(test, "r") as f:
            reader = csv.reader(f)
            for row in reader:
                testset.append(row)
        return testset[1:]

    def testData(self):
        correct = 0
        totalpos = sum(self.pos)
        totalneg = sum(self.neg)
        for sample in self.test:
            y0prob = log(1-self.prob1)
            y1prob = log(self.prob1)
            y = int(sample[-1])
            sample = [int(x) for x in sample[:-1]]
            for i, x in enumerate(sample):
                if x == 0:
                    # print(self.pos[-1] - self.pos[i])
                    y0prob += log((totalneg - self.neg[i])/totalneg)
                    y1prob += log((totalpos - self.pos[i])/totalpos)
                elif x == 1:
                    y0prob += log(self.neg[i]/totalneg)
                    y1prob += log(self.pos[i]/totalpos)
            if y1prob >= y0prob and y == 1 or y1prob < y0prob and y == 0:
                correct += 1
            # print("P0: {}, P1: {}".format(y0prob, y1prob))
        print(correct/len(self.test))

    def testSample(self, sample):
        y0prob = log(1-self.prob1)
        y1prob = log(self.prob1)
        totalpos = sum(self.pos)
        totalneg = sum(self.neg)
        for i, x in enumerate(sample):
            if x == 0:
                # print(self.pos[-1] - self.pos[i])
                y0prob += log((totalneg - self.neg[i])/totalneg)
                y1prob += log((totalpos - self.pos[i])/totalpos)
            elif x == 1:
                y0prob += log(self.neg[i]/totalneg)
                y1prob += log(self.pos[i]/totalpos)
        return y1prob, y0prob

    def writeModel(self):
        biasvector = [0] * self.num
        y1, y0 = self.testSample(biasvector)
        with open(self.model, "w") as f:
            f.write("{}\n".format(y1 - y0))
            for i in range(self.num):
                vec = [0] * self.num
                vec[i] = 1
                y1, y0 = self.testSample(vec)
                f.write("{}{}{}\n".format(self.headers[i], " " * (40 - len(self.headers[i])), y1 - y0))

def main():
    train = sys.argv[1]
    test = sys.argv[2]
    beta = sys.argv[3]
    model = sys.argv[4]
    nb = Nb(train, test, beta, model)
    # print(nb.prob1)
    # print(nb.pos)
    # print(nb.neg)
    nb.testData()
    nb.writeModel()


if __name__ == '__main__':
    main()
