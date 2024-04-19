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
    np.random.normal(loc=[1, 10], scale=[1, 2], size=(size_of_data, 2)),  # Generowanie punktów dla trzeciej klasy
    np.random.normal(loc=[10, 1], scale=[1, 1], size=(size_of_data, 2))  # Generowanie punktów dla czwartej klasy
])

# Podział danych na zbiór treningowy i testowy
train_split = int(0.8 * size_of_data)  # Określenie rozmiaru zbioru treningowego na podstawie proporcji
test_split = int(0.2 * size_of_data)
X_train = np.array([X[_, :train_split] for _ in range(len(X))])  # Podział danych na zbiór treningowy
X_test = np.array([X[_, train_split:] for _ in range(len(X))])  # Podział danych na zbiór testowy


models = []
X_train_flatened = X_train.reshape(-1, 2)

p1 = Perceptron()
p1.train(X_train_flatened, np.array([1] * train_split + [-1] * train_split * 3))
models.append(p1)

p2 = Perceptron()
p2.train(X_train_flatened, np.array([-1] * train_split + [1] * train_split + [-1] * train_split * 2))
models.append(p2)

p3 = Perceptron()
p3.train(X_train_flatened, np.array([-1] * train_split * 2 + [1] * train_split + [-1] * train_split * 1))
models.append(p3)

p4 = Perceptron()
p4.train(X_train_flatened, np.array([-1] * train_split * 3 + [1] * train_split))
models.append(p4)


X_test_flatened = X_test.reshape(-1, 2)

# Przewidywanie klas dla danych testowych i obliczanie dokładności
#indywidualne
print("PIERWSZY PERCEPTRON SKUTECZNOŚĆ: "+(str)(p1.accuracy(X_test_flatened,np.array([1] * test_split + [-1] * test_split * 3))))
print("DRUGI PERCEPTRON SKUTECZNOŚĆ: "+(str)(p2.accuracy(X_test_flatened,np.array([-1] * test_split + [1] * test_split + [-1] * test_split * 2))))
print("TRZECI PERCEPTRON SKUTECZNOŚĆ: "+(str)(p3.accuracy(X_test_flatened,[-1] * test_split * 2 + [1] * test_split + [-1] * test_split * 1)))
print("CZWARY PERCEPTRON SKUTECZNOŚĆ: "+(str)(p4.accuracy(X_test_flatened,np.array([-1] * test_split * 3 + [1] * test_split))))

#Łączone
hit = 0
for x, y in zip(X_test_flatened, np.array([1] * test_split + [2] * test_split + [3] * test_split + [4] * test_split)):
    results = []
    if(p1.predict(x)==1):
        results.append(1)
    if (p2.predict(x)==1):
        results.append(2)
    if (p3.predict(x)==1):
        results.append(3)
    if (p4.predict(x)==1):
        results.append(4)
    if(len(results)==0):
        results.append(1)
        results.append(2)
        results.append(3)
        results.append(4)

    result = random.choice(results)
    if(result==y):
        hit+=1
print("Skuteczność przewidywania globalnie"+(str)(hit/len(X_test_flatened)))

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

for _ in range(4):
    [c, a, b] = models[_].weights  # Współczynniki prostej decyzyjnej
    # Zakres dla zmiennej x
    x_range = np.array([min_x1, max_x1])
    # Obliczenie wartości zmiennej y na podstawie równania prostej
    y_range = (-a * x_range - c) / b
    # Tworzenie wykresu
    plt.plot(x_range, y_range, color=colors[_])  # Rysowanie granic decyzyjnych dla każdej klasy

# Ustawienie limitów dla osi x i y
plt.xlim(min_x1, max_x1)  # Ustawienie limitów dla osi x
plt.ylim(min_x2, max_x2)  # Ustawienie limitów dla osi y

# Zapisanie i wyświetlenie wykresu
plt.savefig('06b perceptron multiclass_new.png')
plt.show()  # Wyświetlenie wszystkich punktów danych oraz granic decyzyjnych