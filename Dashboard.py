import streamlit as st
import os
from PIL import Image
import cv2
import numpy as np
import requests
from io import BytesIO

st.title("Klasifikasi Kepadatan Lalu Lintas dengan Model Convolutional Neural Network Menggunakan Arsitektur EfficientNet ")
st.subheader(""" Kelompok 1""")    
st.markdown("""
            - Adrian Putra Pratama Badjideh (1305213041)
            - Ade Kurniawan (1305210002)
            - Abror Muhammad Hazim (1305213026)""")


col1,col2=st.columns(2)
with col1:
    st.image("Assets/Images/Screenshot(6917).png")
            
with col2:
    st.write("Kepadatan lalu lintas menyebabkan kemacetan di kota-kota besar karena jumlah kendaraan melintas melebihi kapasitas jalan. Proyek ini akan mengklasifikasi kepadatan lalu lintas menggunakan Convolutional Neural Network dengan Arsitektur EfficientNet, menggunakan dataset gambar dari CCTV Pemantauan Lingkungan Kota Bandung.")


st.subheader("Langkah Melakukan Prediksi")
st.markdown("""
            - Siapkan gambar jalan yang akan di prediksi tingkat kemacetannya
            - Patikan gambar yang akan di prediksi memiliki format JPG atau JPEG
            - Drag atau unggah manual gambar yang akan diprediksi pada kolom yang disediakan
            - Tekan "Detect" dan tunggu hasil prediksinya keluar """)

st.subheader("Link Repositories GitHub")
st.write("Untuk akses lebih lanjut pada proyek ini, silakan kunjungi repositori GitHub kami di [link GitHub](https://github.com/zim18/Bandung-Density-Traffic-Classification.git).")
