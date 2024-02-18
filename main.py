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


"""

#file = open('Knn3.txt', 'w')
print("Wine")
wine_data = dt.DataML("wine")
for k in [36, 35]:
    prec = wine_data.matrixknn(k, True)
    #file.write(f"{prec}\n")
    print(f"{k}: {prec}")
#file.close()
print("")




print("Banknote")
banknote_data = dt.DataML("banknote")
banknote_data.matrix_perceptron(0.1, 10)

print("")
print("Wine")
one_wine_data = dt.DataML("wine")
one_wine_data.matrix_one_vs_all()

print("Wine")
one_wine_data = dt.DataML("wine")
one_wine_data.matrix_one_vs_all(1)
"""
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
"""
file = open('ML_alpha.txt', 'w')
print("Wine")
wine_data = dt.DataML("wine")
for alpha in [0, 1e-7, 5e-7, 1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 0.1, 0.5, 1, 5, 10, 50, 100]:
    prec = wine_data.scikit_multilayer_perceptron(show=False, alpha=alpha)
    file.write(f"{prec}\n")
    print(f"{alpha}: {prec}")
file.close()
print("")

file = open('ML_iter.txt', 'w')
print("Wine")
wine_data = dt.DataML("wine")
for i in range(50):
    prec = wine_data.scikit_multilayer_perceptron(show=False, valide=True, iter=i+1)
    file.write(f"{prec}\n")
    print(f"{i}: {prec}")
file.close()
print("")

file = open('ML_hidden.txt', 'w')
print("Wine")
wine_data = dt.DataML("wine")
for i in range(10):
    for j in range(10):
        prec = wine_data.scikit_multilayer_perceptron(show=False, valide=True, hidden_layer_sizes=(i+1,j+1))
        file.write(f"{prec}\n")
        print(f"{i+1,j+1}: {prec}")
file.close()
print("")


#file = open('ML.txt', 'w')
print("Wine")
wine_data = dt.DataML("wine")
for solver in ['lbfgs', 'sgd', 'adam']:
    prec = wine_data.scikit_multilayer_perceptron(show=False, valide=True, solver=solver)
    #file.write(f"{prec}\n")
    print(f"{solver}: {prec}")
#file.close()
print("")


file = open('Forest_max_depth.txt', 'w')
print("Wine")
wine_data = dt.DataML("wine")
for max_depth in range(20):
    prec = wine_data.scikit_random_forest(show=False, valide=True, max_depth=max_depth+1)
    file.write(f"{prec}\n")
    print(f"{max_depth}: {prec}")
file.close()
print("")

file = open('Forest_trees.txt', 'w')
print("Wine")
wine_data = dt.DataML("wine")
for trees in range(100):
    prec = wine_data.scikit_random_forest(show=False, valide=True, n_estimators=trees+1)
    file.write(f"{prec}\n")
    print(f"{trees}: {prec}")
file.close()
print("")
file = open('Gauss.txt', 'w')
print("Wine")
wine_data = dt.DataML("wine")
for var_smoothing in [0, 1e-10, 5e-10, 1e-9, 5e-9, 1e-8, 5e-8, 1e-7, 5e-7, 1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 0.1, 0.5, 1, 5, 10, 50, 100]:
    prec = wine_data.scikit_bayes(show=False, valide=True, var_smoothing=var_smoothing+1)
    file.write(f"{prec}\n")
    print(f"{var_smoothing}: {prec}")
file.close()
print("")
"""

wine_data = dt.DataML("wine")
wine_data.scikit_multilayer_perceptron()
print("")

wine_data = dt.DataML("wine")
wine_data.scikit_random_forest()
print("")

wine_data = dt.DataML("wine")
wine_data.scikit_bayes()
print("")
