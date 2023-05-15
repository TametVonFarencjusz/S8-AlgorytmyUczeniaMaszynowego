import data as dt
import classification as cf

# for i in range(len(iris_data.values_valid)):
#     data = iris_data.kNN(iris_data.values_valid[i].values, 10)
#     if data.class_name == iris_data.values_valid[i].class_name:
#         print("true")
#     else:
#         print(i)

# print("Iris")
# iris_data = dt.DataML("iris")
# iris_data.matrixknn(10)
#
# print("")

print("Wine")
wine_data = dt.DataML("wine")
wine_data.matrixknn(10)

print("")

print("Banknote")
banknote_data = dt.DataML("banknote")
banknote_data.matrix_perceptron(0.1, 10)

print("")

print("Wine")
one_wine_data = dt.DataML("wine")
one_wine_data.matrix_one_vs_all()
"""

wine_data = dt.dataML("iris")
#data = iris_data.kNN([5,3,4,1], 100)
#data = wine_data.kNN(wine_data.values_test[1].values, 100)
#print(data)

for i in range(len(wine_data.values)):
     print(i+1, end=": ")#
     wine_data.matrixKNN(i+1)

#wine_data.matrixKNN(2)

"""