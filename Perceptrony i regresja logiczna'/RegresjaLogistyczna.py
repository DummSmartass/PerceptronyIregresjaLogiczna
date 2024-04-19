import numpy as np
import matplotlib.pyplot as plt
from logreg_mmajew import LogisticRegression

def softmax(x):
    exp_x = np.exp(x)
    sum_exp_x = np.sum(exp_x)
    softmax_values = exp_x / sum_exp_x
    return softmax_values

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
split = int(0.8 * size_of_data)
X_train = np.array([X[_, :split] for _ in range(len(X))])
X_test = np.array([X[_, split:] for _ in range(len(X))])

# Inicjalizacja i trenowanie modeli regresji logistycznej
models = []  # Inicjalizacja listy do przechowywania modeli
'''do uzupelnienia start1'''
#Bo nie mogędotykać X_train vo jest użyte do wyświetlania
test_split = int(size_of_data-split)

toTrainX = np.vstack(X_train)
toTrainY1 = np.array([1] * split + [0] * split * 3)
toTrainY2 = np.array([0] * split + [1] * split + [0] * split * 2)
toTrainY3 = np.array([0] * split * 2 + [1] * split + [0] * split)
toTrainY4 = np.array([0] * split * 3 + [1] * split)

toTestX = np.vstack(X_test)
toTestY1 = np.array([1] * test_split + [0] * test_split * 3)
toTestY2 = np.array([0] * test_split + [1] * test_split + [0] * test_split * 2)
toTestY3 = np.array([0] * test_split * 2 + [1] * test_split + [0] * test_split)
toTestY4 = np.array([0] * test_split * 3 + [1] * test_split)

model1 = LogisticRegression()  # Inicjalizuje LogisticRegression z określoną liczbą iteracji
model1.train(toTrainX, toTrainY1)  # Trenuje logisticregression na danych treningowych
models.append(model1)

model2 = LogisticRegression()  # Inicjalizuje LogisticRegression z określoną liczbą iteracji
model2.train(toTrainX, toTrainY2)  # Trenuje logisticregression na danych treningowych
models.append(model2)

model3 = LogisticRegression()  # Inicjalizuje LogisticRegression z określoną liczbą iteracji
model3.train(toTrainX, toTrainY3)  # Trenuje logisticregression na danych treningowych
models.append(model3)

model4 = LogisticRegression()  # Inicjalizuje LogisticRegression z określoną liczbą iteracji
model4.train(toTrainX, toTrainY4)  # Trenuje logisticregression na danych treningowych
models.append(model4)
'''do uzupelnienia end1'''

# Przewidywanie klas dla danych testowych i obliczanie dokładności
'''do uzupelnienia start2'''
#dokładności indywidualne
accuracym1 = model1.accuracy(toTestX, toTestY1)  # Oblicza dokładność na danych testowych
accuracym2 = model2.accuracy(toTestX, toTestY2)  # Oblicza dokładność na danych testowych
accuracym3 = model3.accuracy(toTestX, toTestY3)  # Oblicza dokładność na danych testowych
accuracym4 = model4.accuracy(toTestX, toTestY4)  # Oblicza dokładność na danych testowych
print("dokładnosć modelu1 (czerwony)"+ ((str)(accuracym1)))
print("dokładnosć modelu2 (zielony)"+ ((str)(accuracym2)))
print("dokładnosć modelu3 (niebieski)"+ ((str)(accuracym3)))
print("dokładnosć modelu4 (magentowy)"+ ((str)(accuracym4)))

#Łączone
hit = 0
# print(len(toTestX))
# print(len(np.array([1] * test_split + [2] * test_split + [3] * test_split + [4] * test_split)))
# print(",,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,")
for x, y in zip(toTestX, np.array([1] * test_split + [2] * test_split + [3] * test_split + [4] * test_split)):
    probabilities = []
    for m in models:
        probabilities.append((m.predict_probability(x)))
    probabilities=softmax(probabilities)

    result = np.argmax(probabilities) + 1

    if(result==y):
        hit+=1
print("Skuteczność przewidywania globalnie: "+(str)(hit/len(toTestX)))


'''do uzupelnienia end2'''


# Wyświetlanie punktów danych treningowych i testowych
colors = ['red', 'green', 'blue', 'magenta']
for _ in range(4):
    plt.scatter(X_train[_][:, 0], X_train[_][:, 1], label=f'Class {_}', color=colors[_], marker='o')
    plt.scatter(X_test[_][:, 0], X_test[_][:, 1], color=colors[_], marker='x')

min_x1 = np.min(X[:,:,0])
max_x1 = np.max(X[:,:,0])
min_x2 = np.min(X[:,:,1])
max_x2 = np.max(X[:,:,1])

for _ in range(4):
    [a, b] = models[_].weights
    c = models[_].bias
    # Zakres dla zmiennej x
    x_range = np.array([min_x1, max_x1])
    # Obliczenie wartości zmiennej y na podstawie równania prostej
    y_range = (-a * x_range - c) / b
    # Tworzenie wykresu
    plt.plot(x_range, y_range, color=colors[_])

# Ustawienie limitów dla osi x i y
plt.xlim(min_x1, max_x1)  # Ustawienie limitów dla osi x
plt.ylim(min_x2, max_x2)  # Ustawienie limitów dla osi y

# Wyświetlenie wykresu
plt.savefig('07b logistic regression multiclass_new.png')
plt.show()  # Wyświetlenie wykresu

# Obliczanie softmax dla środkowego punktu wykresu
x1_middle = (min_x1 + max_x1) / 2
x2_middle = (min_x2 + max_x2) / 2
y_prob = np.array([models[_].predict_probability([[x1_middle, x2_middle]]) for _ in range(len(X))]).flatten()
print(f"Prawdopodobieństwa przynależności {x1_middle, x2_middle} reg. log.: {y_prob}")
print(f"Wektor softmax {x1_middle, x2_middle}: {softmax(y_prob)}")

# Obliczanie softmax dla punktu (0,0) na wykresie
y_prob = np.array([models[_].predict_probability([[0, 0]]) for _ in range(len(X))]).flatten()
print(f"Prawdopodobieństwa przynależności {0, 0} reg. log.: {y_prob}")
print(f"Wektor softmax {0, 0}: {softmax(y_prob)}")