import random

import numpy as np  # Importowanie biblioteki NumPy do operacji na macierzach i wektorach
import matplotlib.pyplot as plt  # Importowanie biblioteki Matplotlib do tworzenia wykresów
from perceptron_mmajew import Perceptron  # Importowanie zaimplementowanej klasy Perceptron

# Generowanie danych
np.random.seed(0)  # Ustawienie ziarna losowości dla powtarzalności wyników
size_of_data = 50  # Określenie liczby punktów danych w każdej klasie
X = np.array([
    np.random.normal(loc=[1, 1], scale=[1, 1], size=(size_of_data, 2)),  # Generowanie punktów dla pierwszej klasy
    np.random.normal(loc=[10, 10], scale=[1, 2], size=(size_of_data, 2)),  # Generowanie punktów dla drugiej klasy
    np.random.normal(loc=[1, 10], scale=[1, 1], size=(size_of_data, 2)),  # Generowanie punktów dla trzeciej klasy
    np.random.normal(loc=[10, 1], scale=[1, 1], size=(size_of_data, 2))  # Generowanie punktów dla czwartej klasy
])

# Podział danych na zbiór treningowy i testowy
train_split = int(0.8 * size_of_data)  # Określenie rozmiaru zbioru treningowego na podstawie proporcji
test_split = int(0.2 * size_of_data)
X_train = np.array([X[_, :train_split] for _ in range(len(X))])  # Podział danych na zbiór treningowy
X_test = np.array([X[_, train_split:] for _ in range(len(X))])  # Podział danych na zbiór testowy

models = []

X_train13v24 =  np.vstack((X_train[0],X_train[2],X_train[1],X_train[3]))
X_train14v32 =  np.vstack((X_train[0],X_train[3],X_train[2],X_train[1]))

standard_answers_training = ([1] * train_split * 2 + [-1]* train_split * 2 )
standard_answers_testing = ([1] * test_split * 2 + [-1]* test_split *2 )

p13v24 = Perceptron()
p13v24.train(X_train13v24, standard_answers_training)
models.append(p13v24)

p14v32 = Perceptron()
p14v32.train(X_train14v32, standard_answers_training)
models.append(p14v32)

X_test13v24 =  np.vstack((X_test[0],X_test[2],X_test[1],X_test[3]))
X_test14v32 =  np.vstack((X_test[0],X_test[3],X_test[2],X_test[1]))
#
# print("===========================")
# print(X_test[0])
# print("---------------------------")
# print(X_test[1])
# print("---------------------------")
# print(X_test1v2)
# print("===========================")
#
# # Przewidywanie klas dla danych testowych i obliczanie dokładności
# #indywidualne
# print(X_test1v2)
# print(standard_answers_testing)
# print("==============================")
print("Skuteczność perceptrona 1v2 red-green: "+(str)(p13v24.accuracy(X_test13v24,standard_answers_testing)))
print("Skuteczność perceptrona 3v4: blue-magenta:  "+(str)(p14v32.accuracy(X_test14v32,standard_answers_testing)))



#Łączone
X_test_flatened = X_test.reshape(-1, 2)

hit = 0
for x, y in zip(X_test_flatened, np.array([1] * test_split + [2] * test_split + [3] * test_split + [4] * test_split)):

    result = 0
    if(p13v24.predict(x)==1):
        if (p14v32.predict(x) == 1):
            result=1
        else:
            result=3
    else:
        if (p14v32.predict(x) == 1):
            result=4
        else:
            result=2


    if(result==y):
        hit+=1

print("Skuteczność przewidywania globalnie: "+(str)(hit/len(X_test_flatened)))



# Wyświetlanie punktów danych treningowych i testowych
colors = ['red', 'green', 'blue', 'magenta']  # Lista kolorów dla różnych klas
for _ in range(4):
    plt.scatter(X_train[_][:, 0], X_train[_][:, 1], label=f'Class {_}', color=colors[_], marker='o')  # Wyświetlenie punktów treningowych
    plt.scatter(X_test[_][:, 0], X_test[_][:, 1], color=colors[_], marker='x')  # Wyświetlenie punktów testowych

# Rysowanie granic decyzyjnych modeli perceptronów
min_x1 = np.min(X[:,:,0])
max_x1 = np.max(X[:,:,0])
min_x2 = np.min(X[:,:,1])
max_x2 = np.max(X[:,:,1])

#1v2 red green
[c, a, b] = models[0].weights  # Współczynniki prostej decyzyjnej
x_range = np.array([min_x1, max_x1])
y_range = (-a * x_range - c) / b
plt.plot(x_range-0.1, y_range, color=colors[0])  # Rysowanie granic decyzyjnych dla każdej klasy
plt.plot(x_range+0.1, y_range, color=colors[1])  # Rysowanie granic decyzyjnych dla każdej klasy

#3v4 blue magenta
[c, a, b] = models[1].weights  # Współczynniki prostej decyzyjnej
x_range = np.array([min_x1, max_x1])
y_range = (-a * x_range - c) / b
plt.plot(x_range, y_range-0.1, color=colors[3])  # Rysowanie granic decyzyjnych dla każdej klasy
plt.plot(x_range, y_range+0.1, color=colors[2])  # Rysowanie granic decyzyjnych dla każdej klasy

# Ustawienie limitów dla osi x i y
plt.xlim(min_x1, max_x1)  # Ustawienie limitów dla osi x
plt.ylim(min_x2, max_x2)  # Ustawienie limitów dla osi y

# Zapisanie i wyświetlenie wykresu
plt.savefig('06b perceptron multiclass_new.png')
plt.show()  # Wyświetlenie wszystkich punktów danych oraz granic decyzyjnych