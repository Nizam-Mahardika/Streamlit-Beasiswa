import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

# Header aplikasi
st.title("Prediksi Status Beasiswa")
st.write("Aplikasi ini memprediksi status penerimaan beasiswa berdasarkan data yang dimasukkan.")

# Upload file
uploaded_file = "DataBeasiswa.xlsx"

if uploaded_file is not None:
    # Membaca file Excel
    data = pd.read_excel(uploaded_file)

    # Menampilkan beberapa baris pertama
    st.write("Data awal:")
    st.write(data.head())

    # Membersihkan data
    data_cleaned = data.drop(columns=["No", "Nama Lengkap", "Prodi", "Asal Sekolah"])

    categorical_columns = ["Jenis Kelamin", "Ikut Organisasi", "Ikut UKM", "Pekerjaan Orang Tua", "Penghasilan", "Status Beasiswa"]
    label_encoders = {col: LabelEncoder() for col in categorical_columns}

    for col in categorical_columns:
        data_cleaned[col] = label_encoders[col].fit_transform(data_cleaned[col])

    data_cleaned["Jarak Tempat Tinggal kekampus (Km)"] = data_cleaned["Jarak Tempat Tinggal kekampus (Km)"].map({"Dekat": 0, "Jauh": 1})

    # Menampilkan data yang sudah dibersihkan
    st.write("Data setelah pembersihan:")
    st.write(data_cleaned.head())

    # Memisahkan fitur dan target
    X = data_cleaned.drop(columns=["Status Beasiswa"])
    y = data_cleaned["Status Beasiswa"]

    # Membagi data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Menampilkan ukuran data
    st.write(f"Ukuran data latih: {X_train.shape}, Ukuran data uji: {X_test.shape}")

    # Melatih model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Prediksi pada data uji
    y_pred = model.predict(X_test)

    # Evaluasi model
    cm = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=["Tidak Terima", "Terima"])

    # Menampilkan hasil evaluasi
    st.subheader("Hasil Evaluasi Model")
    st.write(f"Akurasi: {accuracy}")
    st.write(f"Presisi: {precision}")
    st.write(f"Recall: {recall}")
    st.write(f"F1-Score: {f1}")

    st.text("Laporan Klasifikasi:")
    st.text(report)

    # Menampilkan confusion matrix
    st.subheader("Confusion Matrix")
    fig, ax = plt.subplots()
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Tidak Terima", "Terima"])
    disp.plot(ax=ax)
    st.pyplot(fig)
else:
    st.write("Silakan unggah file Excel untuk memulai analisis.")
