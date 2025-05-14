# train_models.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

# Membuat direktori untuk menyimpan model jika belum ada
if not os.path.exists('models'):
    os.makedirs('models')

# Membuat direktori untuk dataset jika belum ada
if not os.path.exists('dataset'):
    os.makedirs('dataset')

# Memuat data
print("Memuat dataset...")
try:
    df = pd.read_csv('dataset/Dataset_Kelompok_10D.csv')
except FileNotFoundError:
    try:
        df = pd.read_csv('Dataset_Kelompok_10D.csv')
        # Salin ke direktori dataset
        df.to_csv('dataset/Dataset_Kelompok_10D.csv', index=False)
    except FileNotFoundError:
        print("Error: File Dataset_Kelompok_10D.csv tidak ditemukan.")
        exit(1)

print(f"Dataset dimuat dengan {df.shape[0]} baris dan {df.shape[1]} kolom")

# Fitur untuk clustering dan rekomendasi
features = ['Peminat 2024', 'Rasio Keketatan', 'Tingkat Kelulusan (%)', 
            'Maks. Waktu Tunggu Kerja (Bulan)', 'Gaji Awal Min', 'Gaji Awal Max']

X = df[features].copy()

# Standarisasi data
print("Melakukan standarisasi data...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA untuk visualisasi
print("Melakukan reduksi dimensi dengan PCA...")
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
print(f"Variance explained oleh 2 komponen pertama: {pca.explained_variance_ratio_.sum():.2f}")

# Menentukan jumlah cluster optimal dengan metode Elbow
inertia = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

# Plot Elbow Method
plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), inertia, marker='o')
plt.title('Elbow Method untuk Menentukan Jumlah Cluster Optimal')
plt.xlabel('Jumlah Cluster')
plt.ylabel('Inertia')
plt.savefig('elbow_method.png')
plt.close()

# Memilih jumlah cluster
optimal_clusters = 4  # Bisa disesuaikan berdasarkan plot elbow method
print(f"Melakukan clustering dengan {optimal_clusters} cluster...")

# Clustering dengan K-Means
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(X_scaled)

# Menambahkan label cluster ke dataframe
df['Cluster'] = cluster_labels

# Analisis karakteristik cluster
cluster_means = df.groupby('Cluster')[features].mean()
print("\nKarakteristik rata-rata setiap cluster:")
print(cluster_means)

# Interpretasi cluster
print("\nInterpretasi Cluster:")
for cluster in range(optimal_clusters):
    cluster_data = df[df['Cluster'] == cluster]
    avg_peminat = cluster_data['Peminat 2024'].mean()
    avg_gaji = cluster_data['Gaji Awal Max'].mean()
    
    if avg_peminat < df['Peminat 2024'].mean():
        peminat_status = "Sepi Peminat"
    else:
        peminat_status = "Banyak Peminat"
    
    if avg_gaji > df['Gaji Awal Max'].mean():
        gaji_status = "Prospek Bagus"
    else:
        gaji_status = "Prospek Sedang"
    
    print(f"Cluster {cluster}: {peminat_status}, {gaji_status}")
    print(f"  - Jumlah jurusan: {len(cluster_data)}")
    print(f"  - Rata-rata peminat: {avg_peminat:.2f}")
    print(f"  - Rata-rata gaji max: Rp {avg_gaji:,.2f}")

# Melatih model Random Forest untuk memprediksi gaji berdasarkan fitur-fitur lain
print("\nMelatih model Random Forest untuk prediksi gaji...")

# Fitur untuk prediksi gaji
X_rf = df[['Peminat 2024', 'Rasio Keketatan', 'Tingkat Kelulusan (%)', 'Maks. Waktu Tunggu Kerja (Bulan)']]
y_rf = df['Gaji Awal Max']  # Target: prediksi gaji maksimal

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_rf, y_rf, test_size=0.2, random_state=42)

# Train Random Forest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Evaluasi model
y_pred = rf_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Model Random Forest untuk prediksi gaji:")
print(f"Mean Squared Error: {mse:.2f}")
print(f"R² Score: {r2:.2f}")

# Tambahkan data koordinat untuk peta
print("\nMenyiapkan data untuk visualisasi geografis...")
# Koordinat kota-kota di Indonesia (perlu ditambahkan untuk peta yang benar)
kota_coords = {
    'Jakarta': [-6.2088, 106.8456],
    'Bandung': [-6.9175, 107.6191],
    'Surabaya': [-7.2575, 112.7521],
    'Yogyakarta': [-7.7971, 110.3688],
    'Makassar': [-5.1477, 119.4327],
    'Semarang': [-7.0051, 110.4381],
    'Medan': [3.5896, 98.6739],
    'Malang': [-7.9797, 112.6304],
    'Padang': [-0.9198, 100.3531],
    'Denpasar': [-8.6705, 115.2126],
    'Aceh': [4.6951, 96.7494],
    'Palembang': [-2.9761, 104.7754],
    'Banjarmasin': [-3.3186, 114.5944],
    'Manado': [1.4748, 124.8420],
    'Lampung': [-5.4531, 105.2522],
    'Jember': [-8.1690, 113.7007],
    'Samarinda': [-0.5022, 117.1536],
    'Purwokerto': [-7.4249, 109.2353],
    'Solo': [-7.5695, 110.8274],
    'Bogor': [-6.5971, 106.8060],
    'Depok': [-6.4025, 106.7942],
    'Mataram': [-8.5833, 116.1167],
    'Pekanbaru': [0.5103, 101.4478],
    'Pontianak': [-0.0263, 109.3425],
    'Jayapura': [-2.5916, 140.6690],
    'Kupang': [-10.1771, 123.6070],
    'Ambon': [-3.6554, 128.1908],
    'Gorontalo': [0.5387, 123.0622],
    'Bengkulu': [-3.7928, 102.2608],
    'Jambi': [-1.6101, 103.6131],
    'Palangkaraya': [-2.2136, 113.9108],
    'Kendari': [-3.9985, 122.5127],
    'Palu': [-0.9003, 119.8779],
    'Ternate': [0.7833, 127.3833],
    'Sorong': [-0.8663, 131.2507]
}

# Simpan data koordinat untuk digunakan di aplikasi utama
joblib.dump(kota_coords, 'models/kota_coords.pkl')

# Menyimpan model
print("\nMenyimpan model...")
joblib.dump(kmeans, 'models/kmeans_model.pkl')
joblib.dump(scaler, 'models/scaler.pkl')
joblib.dump(pca, 'models/pca_model.pkl')
joblib.dump(rf_model, 'models/random_forest_model.pkl')

# Tambahan: simpan cluster_names untuk interpreatsi di aplikasi utama
cluster_names = {}
for cluster in range(optimal_clusters):
    cluster_data = df[df['Cluster'] == cluster]
    avg_peminat = cluster_data['Peminat 2024'].mean()
    avg_gaji = cluster_data['Gaji Awal Max'].mean()
    
    if avg_peminat < df['Peminat 2024'].mean():
        peminat_status = "Sepi Peminat"
    else:
        peminat_status = "Banyak Peminat"
    
    if avg_gaji > df['Gaji Awal Max'].mean():
        gaji_status = "Prospek Bagus"
    else:
        gaji_status = "Prospek Sedang"
    
    emoji = "🟢" if peminat_status == "Sepi Peminat" and gaji_status == "Prospek Bagus" else \
            "🟡" if peminat_status == "Sepi Peminat" and gaji_status == "Prospek Sedang" else \
            "🔵" if peminat_status == "Banyak Peminat" and gaji_status == "Prospek Bagus" else "🟠"
    
    cluster_names[cluster] = f"{emoji} {peminat_status}, {gaji_status}"

joblib.dump(cluster_names, 'models/cluster_names.pkl')

print("Semua model berhasil disimpan di direktori 'models'")
print("Training selesai!")