import streamlit as st
import librosa
import librosa.display
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Klasifikasi Suara Hewan", page_icon="🐾", layout="centered")

# --- JUDUL & DESKRIPSI ---
st.title("🐾 Klasifikasi Audio: Kucing vs Anjing")
st.markdown("""
Aplikasi ini mendeteksi probabilitas suara **Kucing** atau **Anjing** menggunakan *Deep Learning*.
Termasuk visualisasi **Waveform** dan **Spectrogram** untuk analisis sinyal.
""")

# --- FUNGSI UTAMA ---
def extract_features(file_path):
    try:
        # Load audio
        audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast', duration=3)
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        mfccs_processed = np.mean(mfccs.T, axis=0)
        return np.expand_dims(mfccs_processed, axis=0)
    except Exception as e:
        st.error(f"Error memproses audio: {e}")
        return None

# Fungsi untuk membuat Grafik Visualisasi
def plot_audio_analysis(file_path):
    y, sr = librosa.load(file_path, duration=3)
    
    # Buat container plot (2 baris: Waveform & Spectrogram)
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(10, 8))
    
    # 1. Plot Waveform (Domain Waktu)
    librosa.display.waveshow(y, sr=sr, ax=ax[0], color='blue')
    ax[0].set(title='Waveform (Domain Waktu)', xlabel='Waktu (s)', ylabel='Amplitudo')
    
    # 2. Plot Mel-Spectrogram (Domain Frekuensi - "Trafo")
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    S_dB = librosa.power_to_db(S, ref=np.max)
    img = librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr, fmax=8000, ax=ax[1])
    fig.colorbar(img, ax=ax[1], format='%+2.0f dB')
    ax[1].set(title='Mel-Spectrogram (Domain Frekuensi)', xlabel='Waktu (s)', ylabel='Hz')
    
    plt.tight_layout()
    return fig

# --- LOAD MODEL ---
@st.cache_resource
def load_my_model():
    if os.path.exists('audio_classifier_model.h5'):
        return tf.keras.models.load_model('audio_classifier_model.h5')
    return None

model = load_my_model()

# --- MAIN INTERFACE ---
if model is None:
    st.error("File model 'audio_classifier_model.h5' tidak ditemukan. Jalankan training dulu!")
else:
    st.subheader("1. Upload File Audio")
    uploaded_file = st.file_uploader("Upload file .wav", type=["wav"])

    if uploaded_file is not None:
        # Simpan file sementara
        temp_filename = "temp_audio_viz.wav"
        with open(temp_filename, "wb") as f:
            f.write(uploaded_file.getbuffer())
        # Tampilkan pemutar audio & info
        col1, col2 = st.columns([1, 2])
        with col1:
            st.audio(uploaded_file, format='audio/wav')
        with col2:
            st.info("File berhasil diunggah. Siap dianalisis.")

        # Tombol Prediksi & Analisis
        if st.button("🔍 Analisis & Prediksi"):
            with st.spinner('Sedang melakukan ekstraksi fitur & visualisasi...'):
                
                # A. VISUALISASI
                st.subheader("2. Visualisasi Sinyal (Data Understanding)")
                fig = plot_audio_analysis(temp_filename)
                st.pyplot(fig) # Menampilkan grafik matplotlib 
                
                # B. LAKUKAN PREDIKSI
                feature = extract_features(temp_filename)
                
                if feature is not None:
                    prediction = model.predict(feature)
                    is_dog = prediction[0][0] > 0.5
                    confidence = prediction[0][0] if is_dog else 1 - prediction[0][0]
                    
                    label = "ANJING (Dog) 🐶" if is_dog else "KUCING (Cat) 🐱"
                    
                    st.subheader("3. Hasil Klasifikasi")
                    
                    # Tampilan hasil yang menarik (Metrics)
                    metric_col1, metric_col2 = st.columns(2)
                    with metric_col1:
                        st.metric(label="Prediksi Kelas", value=label)
                    with metric_col2:
                        color = "normal" if confidence < 0.7 else "inverse"
                        st.metric(label="Confidence Score", value=f"{confidence*100:.2f}%")
                    
                    if confidence < 0.6:
                        st.warning("Model agak ragu (Confidence rendah). Mungkin audio kurang jelas atau bercampur noise.")
                    else:
                        st.success("Model cukup yakin dengan prediksi ini.")

        # Cleanup file sementara
        if os.path.exists(temp_filename):
           os.remove(temp_filename)