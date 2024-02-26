import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# membaca data dari file csv
dataku = pd.read_csv('cuaca100.csv')

# menampilkan beberapa baris pertama data
print("Beberapa baris pertama dari data:")
print(dataku.head())

# menampilkan ukuran data dan nama fitur
print("Ukuran:", dataku.shape)
print("Fitur:", dataku.columns)

# mengubah data kategorikal menjadi numerik menggunakan LabelEncoder
le = LabelEncoder()
dataku['cuaca'] = le.fit_transform(dataku['cuaca'])
dataku['suhu'] = le.fit_transform(dataku['suhu'])
dataku['kelembaban'] = le.fit_transform(dataku['kelembaban'])
dataku['angin'] = le.fit_transform(dataku['angin'])
dataku['main'] = le.fit_transform(dataku['main'])

# memisahkan fitur dan target
x = dataku[dataku.columns[:-1]]
y = dataku['main']

# membagi data menjadi set pelatihan dan pengujian
tes_sizes = [0.1, 0.3, 0.5]
for t in tes_sizes:
    print(f"ukuran data tes : {t}")
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=t, random_state=42)

    # melakukan percobaan dengan nilai k yang berbeda
    k_values = [3, 5, 7, 9]
    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(x_train, y_train)
        y_pred = knn.predict(x_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Akurasi KNN dengan k={k}: {accuracy}")