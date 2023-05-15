import pandas as p
import numpy as np
import random

class Data:
    #values
    #class_name
    def __init__(self, vals, name):
        self.values = vals
        self.class_name = name

    def __str__(self):
        return "Class: " + str(self.class_name) + ", Values: " +str(self.values)

    def __repr__(self):
        return "Class: " + str(self.class_name) + ", Values: " +str(self.values)

    def dif (self, vals):
        val = 0
        for i in range(len(self.values)):
            val = (self.values[i] - vals[i]) ** 2
        return val

    def normalize(self, max):
        self.values = [self.values[i]/max[i] for i in range(len(max))]
        return self

    def getgreater(self, val):
        val = [max(self.values[i], val[i]) for i in range(len(val))]
        return val


class DataML:
    #values
    #values_valid
    #values_test
    #info_classes
    #info_max

    def __init__(self, type):
        self.values = []
        self.values_valid = []
        self.values_test = []
        if type == "iris":
            # Additional Data
            with open("Data/irisInfo.data") as f:
                line = f.readline()
                line = line.replace("\n", "")
                vals_str = line.split(",")
                self.info_classes = [val for val in vals_str]
                line = f.readline()
                line = line.replace("\n", "")
                vals_str = line.split(",")
                self.info_max = [float(val) for val in vals_str]
                # self.info_max = [0 for val in range(int(vals_str[0]))]
            with open("Data/irisTr.data") as f:
                for line in f:
                    line = line.replace("\n","")
                    vals_str = line.split(",")
                    vals = [float(val) for val in vals_str[:-1]]
                    self.values.append(Data(vals, vals_str[-1]))
            with open("Data/irisVa.data") as f:
                for line in f:
                    line = line.replace("\n","")
                    vals_str = line.split(",")
                    vals = [float(val) for val in vals_str[:-1]]
                    self.values_valid.append(Data(vals, vals_str[-1]))
            with open("Data/irisTe.data") as f:
                for line in f:
                    line = line.replace("\n","")
                    vals_str = line.split(",")
                    vals = [float(val) for val in vals_str[:-1]]
                    self.values_test.append(Data(vals, vals_str[-1]))

        elif type == "wine":
            #Additional Data
            with open("Data/wineInfo.data") as f:
                line = f.readline()
                line = line.replace("\n","")
                vals_str = line.split(",")
                self.info_classes = [val for val in vals_str]
                line = f.readline()
                line = line.replace("\n", "")
                vals_str = line.split(",")
                self.info_max = [float(val) for val in vals_str]
                #self.info_max = [0 for val in range(int(vals_str[0]))]
            #DATA#
            with open("Data/wineTr.data") as f:
                for line in f:
                    line = line.replace("\n","")
                    vals_str = line.split(",")
                    vals = [float(val) for val in vals_str[1:]]
                    self.values.append(Data(vals, vals_str[0]).normalize(self.info_max))
            #print(self.values)
            with open("Data/wineW.data") as f:
                for line in f:
                    line = line.replace("\n","")
                    vals_str = line.split(",")
                    vals = [float(val) for val in vals_str[1:]]
                    self.values_valid.append(Data(vals, vals_str[0]).normalize(self.info_max))
            #print(self.values_valid)
            with open("Data/wineTe.data") as f:
                for line in f:
                    line = line.replace("\n","")
                    vals_str = line.split(",")
                    vals = [float(val) for val in vals_str[1:]]
                    self.values_test.append(Data(vals, vals_str[0]).normalize(self.info_max))
        elif type == "banknote":
            with open("Data/data_banknote_authentication.txt") as f:
                for line in f:
                    line = line.replace("\n","")
                    vals_str = line.split(",")
                    vals = [float(val) for val in vals_str[:4]]
                    self.values.append(Data(vals, np.sign(int(vals_str[4])-0.5)))
            with open("Data/banknote_test.txt") as f:
                for line in f:
                    line = line.replace("\n","")
                    vals_str = line.split(",")
                    vals = [float(val) for val in vals_str[:4]]
                    self.values_test.append(Data(vals, np.sign(int(vals_str[4])-0.5)))
            with open("Data/banknote_info.txt") as f:
                line = f.readline()
                line = line.replace("\n","")
                vals_str = line.split(",")
                self.info_classes = [val for val in vals_str]


            #for val in self.values:
            #    self.info_max = val.getGreater(self.info_max)
            #for val in self.values_valid:
            #    self.info_max = val.getGreater(self.info_max)
            #for val in self.values_test:
            #    self.info_max = val.getGreater(self.info_max)

            #print(self.info_max)
            #print(self.values_test)
            #print(self.info_classes)

    def knn(self, vals, k):
        values = []
        for dat in self.values:
            val = dat.dif(vals)
            values.append((val, dat.class_name))
        values = sorted(values, key=lambda value: value[0])[:k]

        #print(values)

        class_names = [val[1] for val in values]
        class_name = p.Series(class_names).value_counts().index.tolist()[0]
        #print(p.Series(class_names).value_counts())

        return Data(vals, class_name)

    def matrixknn(self, k):
        #Confusion Matrix
        list = [(class1, class2) for class1 in self.info_classes for class2 in self.info_classes]
        dict_matrix = dict((value, 0) for value in list)

        for val in self.values_test:
            data = self.knn(val.values, k)
            # if data.class_name == val.class_name:
            #     print("true")
            # else:
            #     print("false")
            dict_matrix[(data.class_name, val.class_name)] += 1

        #data.class_name == iris_data.values_test[i].class_name:
        #print(dictMatrix)

        for i,ele in enumerate(list):
            print(dict_matrix[ele], end="")
            if i%len(self.info_classes) == len(self.info_classes)-1:
                print()
            else:
                print(" | ", end="")

        #Precision
        correct = 0
        wrong = 0
        for ele in list:
            if ele[0] == ele[1]:
                correct += dict_matrix[ele]
            else:
                wrong += dict_matrix[ele]
        print(int(correct*10000/(wrong+correct))/100, end="%\n")


    def perceptron_learn(self, learning_rate = 0.1, iteration = 10):
        weight = [0 for ele in range(4+1)]
        for epoch in range(iteration):
            change = False
            for val in self.values:
                predicted_output = np.sign(np.sign(sum([weight[i] * val.values[i] for i in range(len(weight)-1)])+ weight[-1])+0.1)
                #for i, w_ in enumerate(weight):
                for i in range(len(weight)):
                    if i == len(weight) - 1:
                        weight[i] = weight[i] + learning_rate * (int(val.class_name) - predicted_output) * 1
                    else:
                        weight[i] = weight[i] + learning_rate * (int(val.class_name) - predicted_output) * val.values[i]
                if int(val.class_name) != predicted_output:
                    change = True

            if not change:
                print("Break: " + epoch)
                break

        print(weight)
        return weight


    def perceptron(self, values, weight):
        predicted_output = np.sign(np.sign(sum([weight[i] * values[i] for i in range(len(weight) - 1)]) + weight[-1]) + 0.1)
        return Data(values, predicted_output)


    def matrix_perceptron(self, lr = 0.1, iter = 10):
        #Confusion Matrix
        list = [(int(class1), int(class2)) for class1 in self.info_classes for class2 in self.info_classes]
        dict_matrix = dict((value, 0) for value in list)

        weight = self.perceptron_learn(lr, iter)
        for val in self.values_test:
            data = self.perceptron(val.values, weight)
            dict_matrix[(data.class_name, val.class_name)] += 1

        #data.class_name == iris_data.values_test[i].class_name:
        #print(dictMatrix)
        for i,ele in enumerate(list):
            print(dict_matrix[ele], end="")
            if i%len(self.info_classes) == len(self.info_classes)-1:
                print()
            else:
                print(" | ", end="")

        #Precision
        correct = 0
        wrong = 0
        for ele in list:
            if ele[0] == ele[1]:
                correct += dict_matrix[ele]
            else:
                wrong += dict_matrix[ele]
        print(int(correct*10000/(wrong+correct))/100, end="%\n")


    def matrix_one_vs_all(self):
        weights = [(class_name, self.perceptron_learn_2(class_name)) for class_name in self.info_classes]
        for weight in weights:
            print(weight)

        correct = 0
        wrong = 0

        list = [(class1, class2) for class1 in self.info_classes for class2 in self.info_classes]
        dict_matrix = dict((value, 0) for value in list)
        for val in self.values_test:
            data = self.perceptron_2(val.values, weights)
            if data.class_name != -1:
                dict_matrix[(data.class_name, val.class_name)] += 1
            else:
                wrong += 1

        for i,ele in enumerate(list):
            print(dict_matrix[ele], end="")
            if i%len(self.info_classes) == len(self.info_classes)-1:
                print()
            else:
                print(" | ", end="")

        #Precision
        for ele in list:
            if ele[0] == ele[1]:
                correct += dict_matrix[ele]
            else:
                wrong += dict_matrix[ele]
        print(int(correct*10000/(wrong+correct))/100, end="%\n")

    def perceptron_learn_2(self, the_one, learning_rate=1, iteration=500):
        weights = [0 for ele in range(len(self.info_max) + 1)]
        for epoch in range(iteration):
            change = False
            for val in self.values:
                predicted_output = np.sign(np.sign(sum([weights[i] * val.values[i] for i in range(len(weights) - 1)]) + weights[-1]) + 0.1)
                for i in range(len(weights)):
                    if predicted_output < 0 and val.class_name == the_one:
                        correct_value = 1
                        change = True
                    elif predicted_output > 0 and val.class_name != the_one:
                        correct_value = -1
                        change = True
                    else:
                        correct_value = 0

                    if i == len(weights) - 1:
                        weights[i] = weights[i] + learning_rate * correct_value * 1
                    else:
                        weights[i] = weights[i] + learning_rate * correct_value * val.values[i]
            if not change:
                print("Break")
                break
        #print(weights)
        return weights

    def perceptron_2(self, values, weights):
        predict = []
        for weight in weights:
            predicted_output = np.sign(np.sign(sum([weight[1][i] * values[i] for i in range(len(weight[1]) - 1)]) + weight[1][-1]))
            predict.append((weight[0], predicted_output))
        #print(predict)

        output = []

        counter = 0
        for ele in predict:
            if ele[1] > 0:
                output.append(ele)
                counter += 1

        if counter > 1:
            #output = [-1]
            print("ERROR: Double")
            return Data(values, random.choice(output)[0])
        elif counter == 0:
            #output = [-1]
            print("ERROR: Out of bounds")
            return Data(values, random.choice(self.info_classes)[0])

        return Data(values, output[0][0])