# memanggil dataset iris
from sklearn.datasets import load_iris
iris_ku = load_iris()

# simpan data fitur/kolom (x) dan target (y)
x = iris_ku.data
y = iris_ku.target

# simpan nama fitur/kolom (x) dan target (y)
feature_names = iris_ku.feature_names
target_names = iris_ku.target_names

# tampil nama fitur dan target dataset
print("Feature names:", feature_names)
print("Target names:", target_names)

# x dan y adalah numpy arrays
print("\nType of X is:", type(x))

# tampilkan 5 baris pertama
print("\nFirst 5 rows of X:\n", x[:5])

# fitur (x) dan target (y)
x = iris_ku.data
y = iris_ku.target

# splitting X dan y untuk data latih dan uji
from sklearn.model_selection import train_test_split
x_latih, x_tes, y_latih, y_tes = train_test_split(x, y, test_size=0.4, random_state=1)

# tampilkan data fitur latih dan uji
print(x_latih.shape)
print(x_tes.shape)

# tampilkan data target latih dan uji
print(y_latih.shape)
print(y_tes.shape)

# pelatihan pada data latih menggunakan KNN (k=3)
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x_latih, y_latih)

# melakukan prediksi pada data uji
y_prediksi = knn.predict(x_tes)

# perbandingan nilai aktual (y_tes) dengan nilai prediksi (y_prediksi)
from sklearn import metrics
print("Akurasi model KNN:", metrics.accuracy_score(y_tes, y_prediksi))

# prediksi menggunakan data sampel yang dibuat sendiri
contoh = [[3, 5, 4, 2], [2, 3, 5, 4]]

preds = knn.predict(contoh)
pred_species = [iris_ku.target_names[p] for p in preds]

print("Prediksi:", pred_species)

# saving the model
import joblib
joblib.dump(knn, 'iris_knn.pkl')