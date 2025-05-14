import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import os

# Konfigurasi halaman
st.set_page_config(
    page_title="Insight Jurusan IPS di PTN Indonesia",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
def load_css():
    st.markdown("""
    <style>
        /* Base styles */
        body {
            background-color: #121212;
            color: white;
        }
        .main {
            background-color: #121212;
            color: white;
        }
        
        /* Metric value styling */
        div[data-testid="stMetricValue"] {
            font-size: 24px;
            font-weight: 700;
            color: #4A6FE3;
        }
        
        /* Metric label styling */
        [data-testid="stMetricLabel"] {
            font-size: 14px;
            font-weight: 500;
            color: #DDDDDD;
        }
        
        /* Header styling */
        h1, h2, h3, h4, h5, h6 {
            color: white;
        }
        
        /* Highlight text */
        .highlight {
            color: #4A6FE3;
            font-weight: 600;
        }
        
        /* Button styling */
        .stButton>button {
            background-color: #4A6FE3;
            color: white;
            border-radius: 8px;
            padding: 10px 20px;
            font-weight: 500;
            border: none;
            transition: all 0.3s ease;
        }
        .stButton>button:hover {
            background-color: #3A5FD3;
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
            transform: translateY(-2px);
        }
        
        /* Tabs styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 10px;
            background-color: #1E1E1E;
            padding: 10px;
            border-radius: 10px;
        }
        .stTabs [data-baseweb="tab"] {
            background-color: #333333;
            border-radius: 8px;
            padding: 10px 20px;
            color: white;
        }
        .stTabs [aria-selected="true"] {
            background-color: #4A6FE3 !important;
            color: white !important;
            box-shadow: 0 2px 5px rgba(74, 111, 227, 0.3) !important;
        }
        
        /* Table styling */
        .dataframe {
            background-color: #1E1E1E !important;
            color: white !important;
        }
        
        /* Radio buttons */
        .st-cc {
            color: white;
        }

        /* Fix any white backgrounds */
        .element-container, .stDataFrame, .stPlotlyChart, .stText {
            color: white !important;
            background-color: transparent !important;
        }
        
        /* Badge styles */
        .badge {
            display: inline-block;
            padding: 5px 10px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: 600;
            margin-right: 8px;
        }
        .badge-primary {
            background-color: #4A6FE3;
            color: white;
        }
        .badge-success {
            background-color: #2BBD7E;
            color: white;
        }
        .badge-warning {
            background-color: #FFC107;
            color: #333;
        }
        .badge-danger {
            background-color: #E74C3C;
            color: white;
        }
        
        /* Credit footer */
        .footer {
            position: fixed;
            bottom: 0;
            left: 0;
            width: 335px;
            padding: 10px;
            background-color: #1E1E1E;
            color: #DDDDDD;
            text-align: center;
            font-size: 12px;
        }
    </style>
    """, unsafe_allow_html=True)

# Fungsi untuk memuat data
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('dataset/Dataset_Kelompok_10D.csv')
        return df
    except FileNotFoundError:
        try:
            df = pd.read_csv('Dataset_Kelompok_10D.csv')
            return df
        except FileNotFoundError:
            st.error("❌ File data tidak ditemukan. Harap pastikan file Dataset_Kelompok_10D.csv tersedia.")
            return None

# Fungsi untuk memuat model
@st.cache_resource
def load_models():
    models = {}
    try:
        models['kmeans'] = joblib.load('models/kmeans_model.pkl')
        models['scaler'] = joblib.load('models/scaler.pkl')
        models['pca'] = joblib.load('models/pca_model.pkl')
        models['rf'] = joblib.load('models/random_forest_model.pkl')
        return models
    except FileNotFoundError:
        st.warning("⚠️ Model belum tersedia. Jalankan script train_models.py terlebih dahulu untuk melatih model.")
        return None

# Fungsi untuk sistem rekomendasi
def get_recommendations(df, preferences, n=5):
    # Fitur untuk perbandingan
    features = ['Rasio Keketatan', 'Tingkat Kelulusan (%)', 'Maks. Waktu Tunggu Kerja (Bulan)', 
                'Gaji Awal Min', 'Gaji Awal Max']
    
    # Membuat DataFrame preferensi
    user_pref = pd.DataFrame([preferences], columns=features)
    
    # Gabungkan preferensi pengguna dengan data jurusan untuk perhitungan similarity
    combined_data = pd.concat([df[features], user_pref])
    
    # Normalisasi data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(combined_data)
    
    # Hitung similarity
    similarity = cosine_similarity(scaled_data[-1].reshape(1, -1), scaled_data[:-1])
    
    # Dapatkan indeks jurusan dengan similarity tertinggi
    similar_indices = similarity[0].argsort()[::-1][:n]
    
    # Kembalikan jurusan yang direkomendasikan
    return df.iloc[similar_indices]

# Fungsi untuk visualisasi
def create_histogram(df, column, title, color):
    fig = px.histogram(df, x=column, title=title, color_discrete_sequence=[color])
    fig.update_layout(
        xaxis_title=column,
        yaxis_title="Jumlah",
        bargap=0.2,
        showlegend=False,
        plot_bgcolor='rgba(30, 30, 30, 0.8)',
        paper_bgcolor='rgba(30, 30, 30, 0.8)',
        font=dict(color="white")
    )
    return fig

def create_scatter(df, x, y, color, title):
    fig = px.scatter(df, x=x, y=y, color=color, title=title,
                    hover_data=['Nama Jurusan', 'Nama PTN', 'Fakultas'])
    fig.update_layout(
        xaxis_title=x,
        yaxis_title=y,
        plot_bgcolor='rgba(30, 30, 30, 0.8)',
        paper_bgcolor='rgba(30, 30, 30, 0.8)',
        font=dict(color="white")
    )
    return fig

# Fungsi untuk membuat peta Indonesia dengan data jurusan
def create_indonesia_map(df):
    # Koordinat kota-kota di Indonesia
    # Sumber: Google Maps atau sumber terbuka lainnya
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
    
    # Menghitung jumlah jurusan per lokasi
    lokasi_counts = df['Lokasi'].value_counts().reset_index()
    lokasi_counts.columns = ['Lokasi', 'Jumlah Jurusan']
    
    # Menambahkan rata-rata gaji per lokasi
    lokasi_gaji = df.groupby('Lokasi')['Gaji Awal Max'].mean().reset_index()
    lokasi_gaji.columns = ['Lokasi', 'Rata-rata Gaji Max']
    
    # Menggabungkan informasi
    lokasi_info = pd.merge(lokasi_counts, lokasi_gaji, on='Lokasi')
    
    # Menambahkan koordinat ke dataframe
    lokasi_info['lat'] = lokasi_info['Lokasi'].map(lambda x: kota_coords.get(x, [0, 0])[0])
    lokasi_info['lon'] = lokasi_info['Lokasi'].map(lambda x: kota_coords.get(x, [0, 0])[1])
    
    # Hanya ambil data dengan koordinat valid
    lokasi_info = lokasi_info[(lokasi_info['lat'] != 0) & (lokasi_info['lon'] != 0)]
    
    # Buat peta menggunakan px.scatter_mapbox
    fig = px.scatter_mapbox(
        lokasi_info,
        lat='lat',
        lon='lon',
        color='Jumlah Jurusan',
        size='Jumlah Jurusan',
        hover_name='Lokasi',
        hover_data=['Jumlah Jurusan', 'Rata-rata Gaji Max'],
        color_continuous_scale='viridis',
        size_max=25,
        zoom=4,
        title='🗺️ Distribusi Jurusan IPS di Indonesia',
        center={"lat": -2.5, "lon": 118.0},  # Tengah Indonesia
        mapbox_style="carto-darkmatter"  # Pilihan: "open-street-map", "carto-positron", "carto-darkmatter", "stamen-terrain", "stamen-toner", "stamen-watercolor"
    )
    
    fig.update_layout(
        height=600,
        margin={"r": 0, "t": 30, "l": 0, "b": 0},
        plot_bgcolor='rgba(30, 30, 30, 0.8)',
        paper_bgcolor='rgba(30, 30, 30, 0.8)',
        font=dict(color="white")
    )
    
    return fig, lokasi_info

def create_box_plot(df, y, title):
    fig = px.box(df, y=y, title=title)
    fig.update_layout(
        yaxis_title=y,
        plot_bgcolor='rgba(30, 30, 30, 0.8)',
        paper_bgcolor='rgba(30, 30, 30, 0.8)',
        font=dict(color="white")
    )
    return fig

def create_violin_plot(df, x, y, title):
    fig = px.violin(df, x=x, y=y, box=True, title=title)
    fig.update_layout(
        xaxis_title=x,
        yaxis_title=y,
        plot_bgcolor='rgba(30, 30, 30, 0.8)',
        paper_bgcolor='rgba(30, 30, 30, 0.8)',
        font=dict(color="white")
    )
    return fig

def create_bubble_chart(df, x, y, size, color, title):
    fig = px.scatter(
        df, 
        x=x, 
        y=y, 
        size=size, 
        color=color,
        hover_name='Nama Jurusan',
        hover_data=['Nama PTN', 'Fakultas'],
        title=title,
        size_max=30
    )
    fig.update_layout(
        xaxis_title=x,
        yaxis_title=y,
        plot_bgcolor='rgba(30, 30, 30, 0.8)',
        paper_bgcolor='rgba(30, 30, 30, 0.8)',
        font=dict(color="white")
    )
    return fig

def create_sunburst(df, title):
    fig = px.sunburst(
        df, 
        path=['Fakultas', 'Tingkat Kesulitan', 'Tingkat Persaingan Kerja'],
        values='Peminat 2024',
        color='Kebutuhan Industri',
        title=title,
        color_discrete_sequence=px.colors.qualitative.Bold
    )
    fig.update_layout(
        plot_bgcolor='rgba(30, 30, 30, 0.8)',
        paper_bgcolor='rgba(30, 30, 30, 0.8)',
        font=dict(color="white")
    )
    return fig

def create_ridgeline_plot(df, selected_columns, title):
    fig = go.Figure()
    
    y_offset = 0
    for column in selected_columns:
        kde = sns.kdeplot(df[column], bw_adjust=0.5).get_lines()[0].get_data()
        x_kde, y_kde = kde
        
        # Normalisasi KDE untuk skala yang sama
        y_kde = y_kde / np.max(y_kde) * 0.9 + y_offset
        
        # Tambahkan outline untuk KDE
        fig.add_trace(go.Scatter(
            x=x_kde, 
            y=y_kde,
            mode='lines',
            line=dict(color='rgba(255, 255, 255, 0.8)', width=2),
            name=column
        ))
        
        # Tambahkan area di bawah KDE
        fig.add_trace(go.Scatter(
            x=np.concatenate([x_kde, [x_kde[-1], x_kde[0]]]),
            y=np.concatenate([y_kde, [y_offset, y_offset]]),
            fill='toself',
            mode='none',
            name=column,
            showlegend=False,
            fillcolor='rgba(74, 111, 227, 0.5)'
        ))
        
        y_offset += 1
    
    # Tambahkan label teks
    for i, column in enumerate(selected_columns):
        fig.add_annotation(
            x=df[column].min(), 
            y=i + 0.45, 
            text=column,
            showarrow=False,
            font=dict(color="white", size=14)
        )
    
    fig.update_layout(
        title=title,
        showlegend=False,
        plot_bgcolor='rgba(30, 30, 30, 0.8)',
        paper_bgcolor='rgba(30, 30, 30, 0.8)',
        font=dict(color="white"),
        height=600,
        xaxis=dict(
            showgrid=False,
            title="Nilai",
            color="white"
        ),
        yaxis=dict(
            showticklabels=False,
            showgrid=False,
            zeroline=False,
            color="white"
        )
    )
    
    return fig

# Memuat CSS
load_css()

# Memuat data
df = load_data()

# Menambahkan sidebar
st.sidebar.image("image1.webp", use_container_width=True)
st.sidebar.title("📋 Navigasi")

# Menu navigasi
menu = st.sidebar.radio(
    "Pilih Halaman:",
    ["🏠 Beranda", "📊 Visualisasi Data", "🧩 Analisis Cluster", "🔍 Sistem Rekomendasi", "ℹ️ Tentang Aplikasi"]
)

# Credit di sidebar
st.sidebar.markdown("<br><br><br>", unsafe_allow_html=True)
st.sidebar.markdown("""
<div class='footer'>
    <p>© 2025 Kelompok 10 Mini Tim D</p>
    <p>Universitas Udayana</p>
</div>
""", unsafe_allow_html=True)

# Konten berdasarkan menu
if menu == "🏠 Beranda":
    # Placeholder untuk gambar landscape di atas
    st.image("image2.webp", use_container_width=True)
    
    # Judul Aplikasi
    st.markdown("<h1 style='text-align: center;'>🎓 Insight 4: Jurusan IPS di Perguruan Tinggi Negeri di Indonesia</h1>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center;'>Sepi Peminat Namun Memiliki Prospek Kerja Bagus</h2>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center;'>Oleh Kelompok 10 Mini Tim D</h3>", unsafe_allow_html=True)
    
    st.markdown("""
    <div style='text-align: center;'>
    1. Made Pranajaya Dibyacita (549) (2208561122)<br>
    2. Maedelien Tiffany Kariesta Simatupang (550) (2208561065)<br>
    3. Merry Royanti Manalu (551) (2208561069)<br>
    4. Mochamad Abra Ibnu Rais (552) (2201561012)
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Ringkasan dan statistik dasar
    st.markdown("<h2 style='text-align: left;'>📊 Ringkasan Dataset</h2>", unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📋 Total Record", df.shape[0])
    
    with col2:
        st.metric("🏫 Jumlah Jurusan", df['Nama Jurusan'].nunique())
    
    with col3:
        st.metric("🏢 Jumlah PTN", df['Nama PTN'].nunique())
    
    with col4:
        avg_peminat = int(df['Peminat 2024'].mean())
        st.metric("👨‍🎓 Rata-rata Peminat 2024", f"{avg_peminat:,}")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Informasi tentang dataset
    st.markdown("## 📖 Tentang Dataset")
    st.write("""
    Dataset ini berisi informasi tentang jurusan IPS di Perguruan Tinggi Negeri (PTN) di Indonesia yang sepi peminat 
    namun memiliki prospek kerja yang bagus. Dataset mencakup 411 program studi dari 62 PTN di seluruh Indonesia.
    
    Meskipun jurusan-jurusan ini relatif sepi peminat, mereka menawarkan prospek kerja yang menjanjikan dengan 
    gaji awal yang kompetitif dan tingkat persaingan kerja yang beragam.
    
    Insight ini bertujuan untuk memberikan informasi kepada calon mahasiswa tentang pilihan jurusan yang mungkin 
    kurang populer tetapi memiliki peluang karir yang baik di masa depan.
    """)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Visualisasi Ringkasan
    st.markdown("## 📈 Visualisasi Ringkasan")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Distribusi Tingkat Kesulitan
        kesulitan_counts = df['Tingkat Kesulitan'].value_counts().reset_index()
        kesulitan_counts.columns = ['Tingkat Kesulitan', 'Jumlah']
        
        fig_kesulitan = px.pie(
            kesulitan_counts, 
            values='Jumlah', 
            names='Tingkat Kesulitan', 
            title='📚 Distribusi Tingkat Kesulitan Jurusan',
            color_discrete_sequence=px.colors.qualitative.Set3,
            hole=0.4
        )
        fig_kesulitan.update_traces(textposition='inside', textinfo='percent+label')
        fig_kesulitan.update_layout(
            plot_bgcolor='rgba(30, 30, 30, 0.8)',
            paper_bgcolor='rgba(30, 30, 30, 0.8)',
            font=dict(color="white")
        )
        st.plotly_chart(fig_kesulitan, use_container_width=True)
    
    with col2:
        # Distribusi Kebutuhan Industri
        kebutuhan_counts = df['Kebutuhan Industri'].value_counts().reset_index()
        kebutuhan_counts.columns = ['Kebutuhan Industri', 'Jumlah']
        
        fig_kebutuhan = px.pie(
            kebutuhan_counts, 
            values='Jumlah', 
            names='Kebutuhan Industri', 
            title='🏭 Distribusi Kebutuhan Industri',
            color_discrete_sequence=px.colors.qualitative.Pastel,
            hole=0.4
        )
        fig_kebutuhan.update_traces(textposition='inside', textinfo='percent+label')
        fig_kebutuhan.update_layout(
            plot_bgcolor='rgba(30, 30, 30, 0.8)',
            paper_bgcolor='rgba(30, 30, 30, 0.8)',
            font=dict(color="white")
        )
        st.plotly_chart(fig_kebutuhan, use_container_width=True)
    
    # Sunburst chart untuk hubungan fakultas-tingkat kesulitan-persaingan
    st.markdown("<br>", unsafe_allow_html=True)
    fig_sunburst = create_sunburst(df, "🧩 Hierarki Jurusan berdasarkan Fakultas, Tingkat Kesulitan, dan Persaingan Kerja")
    st.plotly_chart(fig_sunburst, use_container_width=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Grafik Scatter Peminat vs Gaji
    fig_scatter = px.scatter(
        df, 
        x='Peminat 2024', 
        y='Gaji Awal Max',
        color='Tingkat Persaingan Kerja',
        size='Rasio Keketatan',
        hover_name='Nama Jurusan',
        hover_data=['Nama PTN', 'Fakultas'],
        title='🔍 Hubungan antara Peminat dan Gaji Maksimum',
        color_discrete_map={
            'Tinggi': '#E3754A', 
            'Menengah': '#66C7F4', 
            'Rendah': '#4A6FE3'
        }
    )
    fig_scatter.update_layout(
        height=500,
        plot_bgcolor='rgba(30, 30, 30, 0.8)',
        paper_bgcolor='rgba(30, 30, 30, 0.8)',
        font=dict(color="white"),
        xaxis=dict(title='Jumlah Peminat 2024', color="white"),
        yaxis=dict(title='Gaji Awal Maksimum (Rp)', color="white")
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

elif menu == "📊 Visualisasi Data":
    st.markdown("## 📊 Visualisasi Data")
    
    # Tab untuk visualisasi yang berbeda
    tabs = st.tabs(["📈 Persebaran Data", "🔄 Hubungan Antar Variabel", "🗺️ Distribusi Geografis", "🏆 Top Jurusan"])
    
    with tabs[0]:
        st.markdown("### 📈 Persebaran Data")
        
        metric_options = [
            'Peminat 2024', 
            'Rasio Keketatan', 
            'Gaji Awal Min', 
            'Gaji Awal Max', 
            'Tingkat Kelulusan (%)',
            'Lama Studi Rata-rata (Bulan)',
            'Maks. Waktu Tunggu Kerja (Bulan)'
        ]
        
        col1, col2 = st.columns(2)
        
        with col1:
            selected_metric1 = st.selectbox("Pilih Metrik 1:", metric_options, index=0)
            
        with col2:
            selected_metric2 = st.selectbox("Pilih Metrik 2:", metric_options, index=2)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = create_histogram(df, selected_metric1, f"Distribusi {selected_metric1}", "#1f77b4")
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            fig = create_histogram(df, selected_metric2, f"Distribusi {selected_metric2}", "#ff7f0e")
            st.plotly_chart(fig, use_container_width=True)
        
        # Box plot untuk melihat outlier
        col1, col2 = st.columns(2)
        
        with col1:
            fig = create_box_plot(df, selected_metric1, f"Box Plot {selected_metric1}")
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            fig = create_box_plot(df, selected_metric2, f"Box Plot {selected_metric2}")
            st.plotly_chart(fig, use_container_width=True)
    
    with tabs[1]:
        st.markdown("### 🔄 Hubungan Antar Variabel")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            x_var = st.selectbox("Pilih Variabel X:", metric_options, index=0)
        
        with col2:
            y_var = st.selectbox("Pilih Variabel Y:", metric_options, index=2)
        
        with col3:
            color_var = st.selectbox("Pilih Variabel Warna:", 
                                     ['Tingkat Kesulitan', 'Tingkat Persaingan Kerja', 'Kebutuhan Industri', 'Akreditasi'], 
                                     index=0)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        fig = create_scatter(df, x_var, y_var, color_var, f"Hubungan antara {x_var} dan {y_var}")
        st.plotly_chart(fig, use_container_width=True)
        
        # Heatmap korelasi
        numeric_cols = ['Peminat 2024', 'Daya Tampung SNBP 2025', 'Daya Tampung SNBT 2025', 
                         'Rasio Keketatan', 'Lama Studi Rata-rata (Bulan)', 'Tingkat Kelulusan (%)',
                         'Maks. Waktu Tunggu Kerja (Bulan)', 'Gaji Awal Min', 'Gaji Awal Max']
        
        corr = df[numeric_cols].corr()
        
        fig = px.imshow(
            corr, 
            text_auto=True, 
            aspect="auto",
            color_continuous_scale='RdBu_r',
            title="Matriks Korelasi Antar Variabel Numerik"
        )
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)
    
    with tabs[2]:
        st.markdown("### 🗺️ Distribusi Geografis")
        
        # Koordinat kota-kota di Indonesia
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
        
        try:
            # Menghitung jumlah jurusan per lokasi
            lokasi_counts = df['Lokasi'].value_counts().reset_index()
            lokasi_counts.columns = ['Lokasi', 'Jumlah Jurusan']
            
            # Menambahkan rata-rata gaji per lokasi
            lokasi_gaji = df.groupby('Lokasi')['Gaji Awal Max'].mean().reset_index()
            lokasi_gaji.columns = ['Lokasi', 'Rata-rata Gaji Max']
            
            # Menggabungkan informasi
            lokasi_info = pd.merge(lokasi_counts, lokasi_gaji, on='Lokasi')
            
            # Menambahkan koordinat ke dataframe
            lokasi_info['lat'] = lokasi_info['Lokasi'].map(lambda x: kota_coords.get(x, [0, 0])[0])
            lokasi_info['lon'] = lokasi_info['Lokasi'].map(lambda x: kota_coords.get(x, [0, 0])[1])
            
            # Hanya ambil data dengan koordinat valid
            lokasi_info = lokasi_info[(lokasi_info['lat'] != 0) & (lokasi_info['lon'] != 0)]
            
            # Buat peta menggunakan px.scatter_mapbox
            fig = px.scatter_mapbox(
                lokasi_info,
                lat='lat',
                lon='lon',
                color='Jumlah Jurusan',
                size='Jumlah Jurusan',
                hover_name='Lokasi',
                hover_data=['Jumlah Jurusan', 'Rata-rata Gaji Max'],
                color_continuous_scale='viridis',
                size_max=25,
                zoom=4,
                title='🗺️ Distribusi Jurusan IPS di Indonesia',
                center={"lat": -2.5, "lon": 118.0},  # Tengah Indonesia
                mapbox_style="carto-darkmatter"  # Pilihan: "open-street-map", "carto-positron", "carto-darkmatter", "stamen-terrain", "stamen-toner", "stamen-watercolor"
            )
            
            fig.update_layout(
                height=600,
                margin={"r": 0, "t": 30, "l": 0, "b": 0},
                plot_bgcolor='rgba(30, 30, 30, 0.8)',
                paper_bgcolor='rgba(30, 30, 30, 0.8)',
                font=dict(color="white")
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("""
            #### 🗺️ Tentang Visualisasi Geografis
            
            Peta di atas menunjukkan distribusi jurusan IPS di berbagai kota di Indonesia. 
            - Ukuran lingkaran menunjukkan jumlah jurusan di lokasi tersebut
            - Warna menunjukkan jumlah jurusan (dari rendah ke tinggi)
            - Hover untuk melihat detail jumlah jurusan dan rata-rata gaji maksimum di lokasi tersebut
            
            Visualisasi ini membantu melihat konsentrasi jurusan IPS di berbagai wilayah Indonesia.
            """)
            
        except Exception as e:
            st.error(f"❌ Error dalam membuat peta: {e}")
            
            # Fallback untuk visualisasi geografis
            lokasi_counts = df['Lokasi'].value_counts().reset_index()
            lokasi_counts.columns = ['Lokasi', 'Jumlah Jurusan']
            
            fig = px.bar(
                lokasi_counts.head(15), 
                x='Lokasi', 
                y='Jumlah Jurusan',
                title='📍 15 Lokasi Teratas berdasarkan Jumlah Jurusan',
                color='Jumlah Jurusan',
                color_continuous_scale='Viridis'
            )
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Visualisasi fakultas
        fakultas_counts = df['Fakultas'].value_counts().reset_index()
        fakultas_counts.columns = ['Fakultas', 'Jumlah Jurusan']
        
        fig = px.bar(
            fakultas_counts.head(10), 
            x='Fakultas', 
            y='Jumlah Jurusan',
            title='🏫 10 Fakultas Teratas berdasarkan Jumlah Jurusan',
            color='Jumlah Jurusan',
            color_continuous_scale='Viridis'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tabs[3]:
        st.markdown("### 🏆 Top Jurusan")
        
        # Pilihan metrik untuk mengurutkan
        sort_metric = st.selectbox(
            "Urutkan Berdasarkan:",
            ['Peminat 2024', 'Gaji Awal Max', 'Rasio Keketatan', 'Tingkat Kelulusan (%)'],
            index=1
        )
        
        asc_order = st.checkbox("Urutkan dari Terkecil", value=False)
        
        # Jumlah jurusan yang ditampilkan
        top_n = st.slider("Jumlah Jurusan yang Ditampilkan:", 5, 50, 10)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Mengurutkan dan mengambil top-n jurusan
        if asc_order:
            top_jurusan = df.sort_values(by=sort_metric).head(top_n)
        else:
            top_jurusan = df.sort_values(by=sort_metric, ascending=False).head(top_n)
        
        # Visualisasi bar chart
        fig = px.bar(
            top_jurusan,
            x='Nama Jurusan',
            y=sort_metric,
            color='Nama PTN',
            hover_data=['Fakultas', 'Lokasi', 'Akreditasi'],
            title=f"Top {top_n} Jurusan berdasarkan {sort_metric}"
        )
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
        
        # Tampilkan tabel
        st.write("Detail Top Jurusan:")
        columns_to_show = ['Nama Jurusan', 'Nama PTN', 'Fakultas', 'Lokasi', 'Akreditasi', 
                           'Peminat 2024', 'Gaji Awal Min', 'Gaji Awal Max', 'Tingkat Kelulusan (%)', 
                           'Prospek Kerja Utama']
        st.dataframe(top_jurusan[columns_to_show], use_container_width=True)

elif menu == "🧩 Analisis Cluster":
    st.markdown("## 🧩 Analisis Cluster")
    
    st.markdown("""
    ### 📖 Tentang Analisis Cluster
    
    Analisis cluster dilakukan untuk mengelompokkan jurusan IPS berdasarkan karakteristik umum menggunakan algoritma K-Means. 
    Pengelompokan dilakukan berdasarkan beberapa fitur utama:
    
    - 👥 Jumlah Peminat 2024
    - 🔥 Rasio Keketatan
    - 💰 Gaji Awal Minimum dan Maksimum
    - ⏱️ Maksimum Waktu Tunggu Kerja
    - 🎓 Tingkat Kelulusan
    
    Hasil clustering menghasilkan beberapa kelompok jurusan dengan karakteristik serupa, yang dapat membantu mengidentifikasi:
    
    1. 🟢 Jurusan sepi peminat dengan prospek kerja bagus (gaji tinggi)
    2. 🟡 Jurusan sepi peminat dengan prospek kerja sedang
    3. 🔵 Jurusan banyak peminat dengan prospek kerja bagus
    4. 🟠 Jurusan banyak peminat dengan prospek kerja sedang
    
    Cluster ini membantu calon mahasiswa menemukan jurusan yang sesuai dengan preferensi mereka,
    terutama bagi yang ingin menghindari persaingan masuk yang ketat tetapi tetap mendapatkan
    prospek karir yang menjanjikan.
    """)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    try:
        # Memuat model
        models = load_models()
        
        # Membuat fitur untuk clustering
        features = ['Peminat 2024', 'Rasio Keketatan', 'Tingkat Kelulusan (%)', 
                    'Maks. Waktu Tunggu Kerja (Bulan)', 'Gaji Awal Min', 'Gaji Awal Max']
        
        X = df[features].copy()
        
        # Transformasi data
        X_scaled = models['scaler'].transform(X)
        
        # Prediksi cluster
        df['Cluster'] = models['kmeans'].predict(X_scaled)
        
        # Mapping nama cluster yang lebih informatif
        cluster_names = {
            0: "🟢 Sepi Peminat, Prospek Bagus",
            1: "🟡 Sepi Peminat, Prospek Sedang",
            2: "🔵 Banyak Peminat, Prospek Bagus",
            3: "🟠 Banyak Peminat, Prospek Sedang"
        }
        
        # Tambahkan nama cluster
        df['Nama Cluster'] = df['Cluster'].map(lambda x: cluster_names.get(x, f"Cluster {x}"))
        
        # Menampilkan jumlah jurusan per cluster
        cluster_counts = df['Nama Cluster'].value_counts().reset_index()
        cluster_counts.columns = ['Nama Cluster', 'Jumlah Jurusan']
        
        # Visualisasi jumlah jurusan per cluster
        fig = px.bar(
            cluster_counts, 
            x='Nama Cluster', 
            y='Jumlah Jurusan',
            color='Nama Cluster',
            title='📊 Jumlah Jurusan per Cluster'
        )
        fig.update_layout(
            plot_bgcolor='rgba(30, 30, 30, 0.8)',
            paper_bgcolor='rgba(30, 30, 30, 0.8)',
            font=dict(color="white"),
            xaxis=dict(title="Cluster", color="white"),
            yaxis=dict(title="Jumlah Jurusan", color="white")
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # PCA untuk visualisasi
        col1, col2 = st.columns(2)
        
        with col1:
            X_pca = models['pca'].transform(X_scaled)
            
            pca_df = pd.DataFrame(
                data=X_pca, 
                columns=['PC1', 'PC2']
            )
            pca_df['Cluster'] = df['Cluster']
            pca_df['Nama Cluster'] = df['Nama Cluster']
            pca_df['Nama Jurusan'] = df['Nama Jurusan']
            pca_df['Nama PTN'] = df['Nama PTN']
            pca_df['Fakultas'] = df['Fakultas']
            pca_df['Peminat 2024'] = df['Peminat 2024']
            pca_df['Gaji Awal Max'] = df['Gaji Awal Max']
            
            # Visualisasi scatter plot PCA
            fig = px.scatter(
                pca_df, 
                x='PC1', 
                y='PC2', 
                color='Nama Cluster',
                hover_name='Nama Jurusan',
                hover_data=['Nama PTN', 'Fakultas', 'Peminat 2024', 'Gaji Awal Max'],
                title='🔍 Visualisasi Cluster menggunakan PCA'
            )
            fig.update_layout(
                plot_bgcolor='rgba(30, 30, 30, 0.8)',
                paper_bgcolor='rgba(30, 30, 30, 0.8)',
                font=dict(color="white"),
                xaxis=dict(title="PC1", color="white"),
                yaxis=dict(title="PC2", color="white")
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("""
            ##### 🔍 Cara Membaca PCA Plot
            PCA (Principal Component Analysis) adalah teknik untuk mereduksi dimensi data. 
            Pada visualisasi ini:
            - Setiap titik adalah satu jurusan
            - Warna menunjukkan cluster yang sama
            - Jurusan yang berdekatan memiliki karakteristik mirip
            - PC1 dan PC2 adalah dua komponen utama yang menangkap sebagian besar variasi dalam data
            """)
        
        with col2:
            # Scatter plot peminat vs gaji berdasarkan cluster
            fig = px.scatter(
                df, 
                x='Peminat 2024', 
                y='Gaji Awal Max',
                color='Nama Cluster',
                hover_name='Nama Jurusan',
                hover_data=['Nama PTN', 'Fakultas', 'Rasio Keketatan'],
                title='💰 Cluster berdasarkan Peminat vs Gaji'
            )
            fig.update_layout(
                plot_bgcolor='rgba(30, 30, 30, 0.8)',
                paper_bgcolor='rgba(30, 30, 30, 0.8)',
                font=dict(color="white"),
                xaxis=dict(title="Peminat 2024", color="white"),
                yaxis=dict(title="Gaji Awal Max", color="white")
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("""
            ##### 💰 Cara Membaca Scatter Plot Cluster
            Plot ini menunjukkan hubungan antara jumlah peminat dan gaji maksimum untuk setiap cluster:
            - 🟢 Cluster hijau: Jurusan sepi peminat dengan gaji tinggi (ideal untuk yang mencari peluang masuk lebih mudah dengan prospek kerja bagus)
            - 🟡 Cluster kuning: Jurusan sepi peminat dengan gaji sedang
            - 🔵 Cluster biru: Jurusan banyak peminat dengan gaji tinggi (kompetitif tapi bernilai)
            - 🟠 Cluster orange: Jurusan banyak peminat dengan gaji sedang (paling kompetitif)
            """)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Analisis karakteristik cluster
        st.markdown("<h3 style='text-align: center;'>📊 Karakteristik Setiap Cluster</h3>", unsafe_allow_html=True)
        
        # Mendapatkan rata-rata fitur per cluster
        cluster_means = df.groupby('Nama Cluster')[features].mean().reset_index()
        
        # Membuat radar chart untuk setiap cluster
        fig = go.Figure()
        
        for cluster in cluster_means['Nama Cluster'].unique():
            cluster_data = cluster_means[cluster_means['Nama Cluster'] == cluster]
            
            # Normalisasi nilai untuk radar chart
            values = []
            for feature in features:
                min_val = df[feature].min()
                max_val = df[feature].max()
                
                # Inverse untuk variabel dimana nilai rendah lebih baik
                if feature == 'Maks. Waktu Tunggu Kerja (Bulan)':
                    norm_val = 1 - ((cluster_data[feature].values[0] - min_val) / (max_val - min_val))
                else:
                    norm_val = (cluster_data[feature].values[0] - min_val) / (max_val - min_val)
                
                values.append(norm_val)
            
            # Tambahkan titik pertama ke akhir untuk menutup poligon
            features_radar = features + [features[0]]
            values = values + [values[0]]
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=features_radar,
                fill='toself',
                name=cluster
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1],
                    color="white"
                ),
                bgcolor="rgba(30, 30, 30, 0.8)"
            ),
            title="📊 Karakteristik Rata-Rata Setiap Cluster (Nilai Ternormalisasi)",
            showlegend=True,
            plot_bgcolor='rgba(30, 30, 30, 0.8)',
            paper_bgcolor='rgba(30, 30, 30, 0.8)',
            font=dict(color="white")
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        ##### 📊 Interpretasi Radar Chart
        
        Chart ini menunjukkan karakteristik rata-rata setiap cluster:
        - Nilai mendekati 1 (tepi luar) menunjukkan nilai tinggi dibandingkan cluster lain
        - Untuk Waktu Tunggu Kerja, nilai dibalik sehingga mendekati 1 berarti waktu tunggu LEBIH PENDEK
        
        **Rekomendasi Berdasarkan Profil Anda:**
        - Jika Anda ingin gaji tinggi dan peluang kerja cepat: Fokus pada cluster 🟢 dan 🔵
        - Jika Anda ingin persaingan masuk yang rendah: Fokus pada cluster 🟢 dan 🟡
        - Jika Anda mencari keseimbangan terbaik: Cluster 🟢 (Sepi Peminat, Prospek Bagus) adalah pilihan optimal
        """)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Tabel karakteristik cluster
        st.write("📋 Nilai Rata-Rata Fitur per Cluster:")
        st.dataframe(cluster_means, use_container_width=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Analisis cluster berdasarkan variabel kategorikal
        st.markdown("<h3 style='text-align: center;'>📊 Distribusi Variabel Kategorikal dalam Cluster</h3>", unsafe_allow_html=True)
        
        categorical_vars = ['Tingkat Kesulitan', 'Tingkat Persaingan Kerja', 'Kebutuhan Industri', 'Akreditasi']
        selected_cat = st.selectbox("🔍 Pilih Variabel Kategorikal:", categorical_vars)
        
        # Buat crosstab
        cross_tab = pd.crosstab(df['Nama Cluster'], df[selected_cat])
        cross_tab_norm = cross_tab.div(cross_tab.sum(axis=1), axis=0)
        
        # Visualisasi heatmap
        fig = px.imshow(
            cross_tab_norm,
            text_auto=True,
            aspect="auto",
            color_continuous_scale='Blues',
            title=f"📊 Proporsi {selected_cat} dalam Setiap Cluster"
        )
        fig.update_layout(
            plot_bgcolor='rgba(30, 30, 30, 0.8)',
            paper_bgcolor='rgba(30, 30, 30, 0.8)',
            font=dict(color="white")
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown(f"""
        ##### 📊 Interpretasi Distribusi {selected_cat}
        
        Heatmap ini menunjukkan proporsi nilai {selected_cat} dalam setiap cluster.
        - Warna lebih gelap menunjukkan proporsi lebih tinggi
        - Angka dalam sel adalah persentase (0-1)
        
        Ini membantu memahami karakteristik dominan setiap cluster dari segi {selected_cat}.
        """)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Tabel jurusan per cluster
        st.markdown("<h3 style='text-align: center;'>📋 Daftar Jurusan dalam Cluster</h3>", unsafe_allow_html=True)
        
        selected_cluster = st.selectbox("🔍 Pilih Cluster:", sorted(df['Nama Cluster'].unique()))
        
        cluster_jurusan = df[df['Nama Cluster'] == selected_cluster]
        st.write(f"Jumlah jurusan dalam {selected_cluster}: {len(cluster_jurusan)}")
        
        columns_to_show = ['Nama Jurusan', 'Nama PTN', 'Fakultas', 'Peminat 2024', 
                           'Gaji Awal Min', 'Gaji Awal Max', 'Tingkat Kelulusan (%)',
                           'Tingkat Kesulitan', 'Tingkat Persaingan Kerja']
        
        st.dataframe(cluster_jurusan[columns_to_show], use_container_width=True)
        
        st.markdown(f"""
        ##### 💡 Rekomendasi untuk Cluster {selected_cluster}
        
        **Karakteristik Utama:**
        {"✅ Prospek kerja bagus dengan gaji relatif tinggi" if "Prospek Bagus" in selected_cluster else "⚠️ Prospek kerja sedang dengan gaji relatif lebih rendah"}
        {"✅ Kompetisi masuk relatif lebih rendah" if "Sepi Peminat" in selected_cluster else "⚠️ Kompetisi masuk relatif tinggi"}
        
        **Cocok untuk calon mahasiswa yang:**
        {"- Mencari peluang karir yang baik dengan persaingan masuk lebih rendah" if "Sepi Peminat, Prospek Bagus" in selected_cluster else ""}
        {"- Menginginkan prestige jurusan populer dengan prospek karir baik" if "Banyak Peminat, Prospek Bagus" in selected_cluster else ""}
        {"- Mencari jalur masuk lebih mudah dan bersedia menerima prospek karir sedang" if "Sepi Peminat, Prospek Sedang" in selected_cluster else ""}
        {"- Tertarik pada jurusan populer meskipun prospek karir lebih moderat" if "Banyak Peminat, Prospek Sedang" in selected_cluster else ""}
        
        **Strategi pemilihan jurusan:**
        - Filter berdasarkan lokasi dan fakultas yang diminati
        - Pertimbangkan tingkat kesulitan dan kebutuhan industri
        - Bandingkan rasio keketatan untuk melihat peluang masuk
        """)
    
    except Exception as e:
        st.error(f"❌ Error dalam analisis cluster: {e}")
        st.info("⚠️ Jalankan script train_models.py terlebih dahulu untuk membuat model clustering.")

elif menu == "🔍 Sistem Rekomendasi":
    st.markdown("## 🔍 Sistem Rekomendasi Jurusan")
    
    st.markdown("""
    ### 📝 Tentang Sistem Rekomendasi
    
    Sistem rekomendasi ini membantu calon mahasiswa menemukan jurusan IPS yang sesuai dengan preferensi mereka.
    Rekomendasi diberikan berdasarkan kriteria yang dipilih oleh pengguna, seperti:
    
    * 💰 Preferensi gaji
    * 🎓 Tingkat kelulusan
    * ⏱️ Waktu tunggu kerja
    * 🔥 Rasio keketatan
    * 📍 Lokasi yang diinginkan
    * 📚 Tingkat kesulitan
    
    Sistem akan memberikan daftar jurusan yang paling cocok dengan kriteria yang dipilih, 
    serta memberikan informasi tambahan tentang jurusan tersebut untuk membantu pengambilan keputusan.
    """)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    st.markdown("### ⚙️ Masukkan Preferensi Anda")
    
    col1, col2 = st.columns(2)
    
    with col1:
        tingkat_kelulusan = st.slider(
            "🎓 Tingkat Kelulusan (%) Minimal:", 
            min_value=int(df['Tingkat Kelulusan (%)'].min()),
            max_value=int(df['Tingkat Kelulusan (%)'].max()),
            value=85
        )
        
        waktu_tunggu = st.slider(
            "⏱️ Maksimum Waktu Tunggu Kerja (Bulan):", 
            min_value=int(df['Maks. Waktu Tunggu Kerja (Bulan)'].min()),
            max_value=int(df['Maks. Waktu Tunggu Kerja (Bulan)'].max()),
            value=36
        )
        
        gaji_min = st.slider(
            "💰 Gaji Awal Minimum (Rp):", 
            min_value=int(df['Gaji Awal Min'].min()),
            max_value=int(df['Gaji Awal Min'].max()),
            value=4500000,
            step=500000,
            format="%d"
        )
    
    with col2:
        keketatan = st.slider(
            "🔥 Rasio Keketatan Maksimal:", 
            min_value=float(df['Rasio Keketatan'].min()),
            max_value=float(df['Rasio Keketatan'].max()),
            value=7.0,
            step=0.5
        )
        
        gaji_max = st.slider(
            "💰 Gaji Awal Maksimum (Rp):", 
            min_value=int(df['Gaji Awal Max'].min()),
            max_value=int(df['Gaji Awal Max'].max()),
            value=8500000,
            step=500000,
            format="%d"
        )
    
    # Filter berdasarkan lokasi dan tingkat kesulitan
    col1, col2 = st.columns(2)
    
    with col1:
        selected_locations = st.multiselect(
            "📍 Pilih Lokasi (Kosongkan untuk semua):",
            df['Lokasi'].unique(),
            default=[]
        )
    
    with col2:
        selected_difficulty = st.multiselect(
            "📚 Pilih Tingkat Kesulitan:",
            df['Tingkat Kesulitan'].unique(),
            default=df['Tingkat Kesulitan'].unique()
        )
    
    # Preferensi user
    preferences = {
        'Rasio Keketatan': keketatan,
        'Tingkat Kelulusan (%)': tingkat_kelulusan,
        'Maks. Waktu Tunggu Kerja (Bulan)': waktu_tunggu,
        'Gaji Awal Min': gaji_min,
        'Gaji Awal Max': gaji_max
    }
    
    # Filter dataset
    filtered_df = df.copy()
    
    if selected_locations:
        filtered_df = filtered_df[filtered_df['Lokasi'].isin(selected_locations)]
    
    if selected_difficulty:
        filtered_df = filtered_df[filtered_df['Tingkat Kesulitan'].isin(selected_difficulty)]
    
    # Prioritas (jurusan sepi peminat atau tidak)
    prioritas_sepi = st.checkbox("🔍 Prioritaskan Jurusan Sepi Peminat")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Tombol untuk mendapatkan rekomendasi
    if st.button("🔎 Dapatkan Rekomendasi"):
        if len(filtered_df) > 0:
            # Jumlah rekomendasi
            num_recommendations = min(10, len(filtered_df))
            
            # Dapatkan rekomendasi
            recommendations = get_recommendations(filtered_df, preferences, n=num_recommendations)
            
            # Jika prioritas jurusan sepi peminat
            if prioritas_sepi:
                recommendations = recommendations.sort_values(['Peminat 2024', 'Gaji Awal Max'], ascending=[True, False])
            
            # Tampilkan rekomendasi
            st.markdown("<h3 style='text-align: center;'>🎯 Jurusan yang Direkomendasikan Untuk Anda</h3>", unsafe_allow_html=True)
            
            # Analisis singkat rekomendasi
            avg_peminat = recommendations['Peminat 2024'].mean()
            avg_gaji = recommendations['Gaji Awal Max'].mean()
            avg_rasio = recommendations['Rasio Keketatan'].mean()
            
            st.markdown(f"""
            #### 📋 Ringkasan Rekomendasi
            
            Berdasarkan preferensi Anda, kami merekomendasikan {len(recommendations)} jurusan yang cocok. Secara umum, jurusan-jurusan ini memiliki:
            
            - Rata-rata peminat: **{avg_peminat:.0f}** orang
            - Rata-rata gaji maksimum: **Rp {avg_gaji:,.0f}**
            - Rata-rata rasio keketatan: **{avg_rasio:.2f}**
            
            **💡 Saran untuk Anda:**
            
            - {'✅ Jurusan-jurusan ini relatif sepi peminat dengan prospek kerja baik' if prioritas_sepi else '✅ Jurusan-jurusan ini memiliki keseimbangan antara peluang masuk dan prospek kerja'}
            - {'⚠️ Perhatikan lokasi dan fakultas untuk kenyamanan studi Anda' if selected_locations else '⚠️ Pertimbangkan lokasi studi karena Anda belum memilih preferensi lokasi'}
            - {'✅ Fokus pada jurusan dengan kebutuhan industri tinggi untuk prospek jangka panjang' if recommendations['Kebutuhan Industri'].value_counts().idxmax() == 'Tinggi' else '⚠️ Jurusan dengan kebutuhan industri sedang mungkin memerlukan keahlian tambahan'}
            """)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Untuk setiap rekomendasi, buat card
            for i, (idx, row) in enumerate(recommendations.iterrows()):
                col1, col2 = st.columns([1, 3])
                
                with col1:
                    # Placeholder untuk gambar
                    st.markdown(f"### {i+1}")
                
                with col2:
                    st.markdown(f"### {row['Nama Jurusan']}")
                    st.markdown(f"**{row['Nama PTN']}** - {row['Fakultas']}")
                    st.markdown(f"**Lokasi:** {row['Lokasi']} | **Akreditasi:** {row['Akreditasi']}")
                    
                    met1, met2, met3 = st.columns(3)
                    with met1:
                        st.metric("👥 Peminat 2024", row['Peminat 2024'])
                    with met2:
                        st.metric("💰 Gaji Awal", f"Rp {row['Gaji Awal Min']:,} - {row['Gaji Awal Max']:,}")
                    with met3:
                        st.metric("🎓 Kelulusan", f"{row['Tingkat Kelulusan (%)']}%")
                    
                    st.markdown(f"**💼 Prospek Kerja Utama:** {row['Prospek Kerja Utama']}")
                    st.markdown(f"**🔄 Prospek Kerja Alternatif:** {row['Prospek Kerja Alternatif']}")
                    st.markdown(f"**📚 Tingkat Kesulitan:** {row['Tingkat Kesulitan']} | **⚔️ Persaingan Kerja:** {row['Tingkat Persaingan Kerja']}")
                    
                    # Tambahkan saran khusus untuk jurusan ini
                    st.markdown(f"""
                    **💡 Saran untuk jurusan ini:**
                    - {"✅ Jurusan ini memiliki gaji tinggi dengan peminat relatif sedikit, peluang bagus!" if row['Peminat 2024'] < avg_peminat and row['Gaji Awal Max'] > avg_gaji else "⚠️ Perhatikan rasio keketatan untuk menilai peluang masuk"}
                    - {"✅ Waktu tunggu kerja singkat, prospek cepat bekerja" if row['Maks. Waktu Tunggu Kerja (Bulan)'] < 24 else "⚠️ Siapkan diri untuk waktu tunggu kerja yang moderat"}
                    - {"✅ Kebutuhan industri tinggi, peluang kerja jangka panjang baik" if row['Kebutuhan Industri'] == 'Tinggi' else "⚠️ Perlu keterampilan tambahan untuk meningkatkan daya saing"}
                    """)
                
                st.markdown("---")
            
            # Visualisasi perbandingan rekomendasi
            st.markdown("<h3 style='text-align: center;'>📊 Perbandingan Rekomendasi</h3>", unsafe_allow_html=True)
            
            # Radar chart untuk perbandingan top 5 rekomendasi
            top_5 = recommendations.head(5)
            
            features_radar = ['Peminat 2024', 'Rasio Keketatan', 'Tingkat Kelulusan (%)', 
                        'Gaji Awal Max', 'Maks. Waktu Tunggu Kerja (Bulan)']
            
            # Membuat radar chart
            fig = go.Figure()
            
            for idx, row in top_5.iterrows():
                values = []
                for feature in features_radar:
                    if feature == 'Maks. Waktu Tunggu Kerja (Bulan)':
                        # Inverse normalization for wait time (lower is better)
                        min_val = df[feature].min()
                        max_val = df[feature].max()
                        norm_val = 1 - ((row[feature] - min_val) / (max_val - min_val))
                    else:
                        min_val = df[feature].min()
                        max_val = df[feature].max()
                        norm_val = (row[feature] - min_val) / (max_val - min_val)
                    
                    values.append(norm_val)
                
                # Menutup radar chart
                values.append(values[0])
                
                # Labels dengan feature pertama di akhir
                labels = features_radar + [features_radar[0]]
                
                fig.add_trace(go.Scatterpolar(
                    r=values,
                    theta=labels,
                    fill='toself',
                    name=row['Nama Jurusan']
                ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1],
                        color="white"
                    ),
                    bgcolor="rgba(30, 30, 30, 0.8)"
                ),
                title="📊 Perbandingan 5 Rekomendasi Teratas",
                showlegend=True,
                plot_bgcolor='rgba(30, 30, 30, 0.8)',
                paper_bgcolor='rgba(30, 30, 30, 0.8)',
                font=dict(color="white")
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("""
            #### 📌 Selanjutnya, Anda bisa:
            
            1. **Pelajari lebih lanjut** tentang jurusan yang direkomendasikan
            2. **Bandingkan** rasio keketatan dan lokasi untuk memperkirakan peluang masuk
            3. **Pertimbangkan** tingkat kesulitan jurusan yang sesuai dengan kemampuan Anda
            4. **Kunjungi** website resmi PTN untuk informasi lebih detail
            5. **Konsultasikan** dengan guru BK, alumni, atau profesional di bidang tersebut
            
            Selamat memilih jurusan, semoga sukses! 🎓
            """)
        else:
            st.warning("⚠️ Tidak ada jurusan yang sesuai dengan filter yang dipilih. Silakan ubah filter Anda.")

elif menu == "ℹ️ Tentang Aplikasi":
    st.markdown("## ℹ️ Tentang Aplikasi")
    
    st.markdown("""
    ### 🎓 Insight Jurusan IPS di Perguruan Tinggi Negeri Indonesia
    
    Aplikasi ini dikembangkan oleh Kelompok 10 Mini Tim D untuk menganalisis dan memvisualisasikan data jurusan IPS di Perguruan Tinggi Negeri (PTN) di Indonesia yang sepi peminat namun memiliki prospek kerja yang bagus.
    
    #### ✨ Fitur Utama:
    - 📊 Visualisasi interaktif tentang jurusan IPS di PTN
    - 🧩 Analisis cluster untuk mengelompokkan jurusan berdasarkan karakteristik serupa
    - 🧠 Sistem rekomendasi untuk membantu calon mahasiswa memilih jurusan
    - 🔄 Perbandingan antar jurusan dan PTN
    - 🗺️ Pemetaan distribusi geografis jurusan di Indonesia
    
    #### 🎯 Tujuan Aplikasi:
    - Membantu calon mahasiswa menemukan jurusan IPS yang sesuai dengan preferensi mereka
    - Menganalisis pola dalam data jurusan IPS untuk memberikan insight yang bermanfaat
    - Menyajikan visualisasi data yang mudah dipahami untuk pengambilan keputusan
    - Mendorong calon mahasiswa untuk mempertimbangkan jurusan yang sepi peminat namun memiliki prospek kerja bagus
    
    #### 🛠️ Teknologi yang Digunakan:
    - 🐍 Python dan Streamlit untuk pengembangan aplikasi
    - 📊 Pandas dan NumPy untuk manipulasi data
    - 🧠 Scikit-learn untuk machine learning (K-Means, PCA, Random Forest)
    - 📈 Plotly dan Matplotlib untuk visualisasi interaktif
    
    #### 📚 Sumber Data:
    Data yang digunakan dalam aplikasi ini berasal dari berbagai sumber seperti halaman resmi PTN, Kementerian Pendidikan dan Kebudayaan, serta sumber-sumber terpercaya lainnya.
    """)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    st.markdown("### 👨‍👩‍👧‍👦 Tim Pengembang")
    
    # Gunakan kolom untuk menampilkan tim
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Made Pranajaya Dibyacita**
        - NIM: 2208561122
        - Absen: 549
        - Institusi: Universitas Udayana
        
        **Maedelien Tiffany Kariesta Simatupang**
        - NIM: 2208561065
        - Absen: 550
        - Institusi: Universitas Udayana
        """)
    
    with col2:
        st.markdown("""
        **Merry Royanti Manalu**
        - NIM: 2208561069
        - Absen: 551
        - Institusi: Universitas Udayana
        
        **Mochamad Abra Ibnu Rais**
        - NIM: 2201561012
        - Absen: 552
        - Institusi: Universitas Udayana
        """)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    st.markdown("### 📋 Cara Menggunakan Aplikasi")
    
    st.markdown("""
    1. **🏠 Beranda**:
       - Lihat ringkasan dataset dan visualisasi utama
       - Pelajari tentang tujuan dan scope aplikasi
    
    2. **📊 Visualisasi Data**:
       - Eksplor berbagai visualisasi interaktif dari data jurusan
       - Pilih variabel yang ingin Anda analisis
       - Lihat distribusi geografis jurusan di seluruh Indonesia
    
    3. **🧩 Analisis Cluster**:
       - Pelajari hasil pengelompokan jurusan berdasarkan karakteristik serupa
       - Lihat detail setiap cluster dan jurusan yang termasuk di dalamnya
       - Pahami pola dalam data jurusan IPS
    
    4. **🔍 Sistem Rekomendasi**:
       - Masukkan preferensi Anda seperti gaji, lokasi, dan tingkat kesulitan
       - Dapatkan rekomendasi jurusan yang paling sesuai dengan preferensi Anda
       - Baca detail dan saran untuk setiap jurusan yang direkomendasikan
    """)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    st.markdown("### 📞 Kontak")
    
    st.markdown("""
    Jika Anda memiliki pertanyaan, saran, atau umpan balik tentang aplikasi ini, silakan hubungi kami melalui:
    
    - 🔍 GitHub: https://github.com/mdprana/ips-ptn-dashboard
    - 🌐 Website: https://tim10D-dashboard.streamlit.app
    - 📧 Email: mdpranajaya@gmail.com
    
    Kami sangat menghargai masukan Anda untuk pengembangan aplikasi ini lebih lanjut.
    """)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    st.markdown("### 📜 Lisensi dan Penggunaan")
    
    st.markdown("""
    © 2025 Kelompok 10 Mini Tim D, Universitas Udayana.
    
    Aplikasi ini dikembangkan untuk tujuan pendidikan dan dapat digunakan secara bebas oleh calon mahasiswa, orang tua, guru, dan pihak lain yang berkepentingan.
    
    Data yang disajikan dalam aplikasi ini bersifat informatif dan sebaiknya dikonfirmasi dengan sumber resmi sebelum pengambilan keputusan.
    """)