import pandas as pd

# membaca data dari file csv ya bolo...
dataku = pd.read_csv('cuaca.csv')

# menampilkan beberapa baris pertama data ya bolo...
print("Beberapa baris pertama dari data:")
print(dataku.head())

# menampilkan ukuran data dan nama fitur ya bolo...
print("Ukuran:", dataku.shape)
print("Fitur:", dataku.columns)

# memisahkan fitur dan target ya bolo...
x = dataku[dataku.columns[:-1]]
y = dataku[dataku.columns[-1]]

# menampilkan semua isi dataset ya bolo...
print("Feature matrix:", x)
print("Response vector:", y)

# menampilkan 7 baris terakhir ya bolo...
print("Feature matrix:", x.tail(7))
print("Response vector:", y.tail(7))