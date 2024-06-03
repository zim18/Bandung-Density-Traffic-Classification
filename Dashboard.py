import streamlit as st
import os
from PIL import Image
import cv2
import numpy as np

root_dir=os.path.dirname(os.path.abspath(__file__))
homepage_img_dir=os.path.join(root_dir,"images","home_page")
st.title("Klasifikasi Kepadatan Lalu Lintas dengan Model Convolutional Neural Network Menggunakan Arsitektur EfficientNet ")
st.subheader(""" Kelompok 1""")    
st.markdown("""
            - Adrian Putra Pratama Badjideh (1305213041)
            - Ade Kurniawan (1305210002)
            - Abror Muhammad Hazim (1305213026)""")
col1,col2=st.columns(2)
with col1:
    img_path=os.path.join(homepage_img_dir,"Screenshot (6917).png")
    img=Image.open(img_path)
    img=cv2.resize(np.array(img),(480,480))
    st.image(img)
with col2:
    st.write("**Kepadatan lalu lintas merupakan sebuah kondisi ketika jumlah kendaraan yang melintas melebihi kapasitas jalan yang tersedia. Kepadatan ini merupakan salah satu penyebab dari kemacetan yang terjadi terutama pada kota-kota besar. Pada proyek ini kepadatan lalu lintas yang terjadi akan diklasifikasi dengan model Convolutional Neural Network menggunakan Arsitektur EfficientNet menggunakan dataset gambar yang dikumpulkan dari persimpangan jalan Kota Bandung secara manual melalui website CCTV Pemantauan Lingkungan Kota Bandung. **")
st.subheader("Langkah Melakukan Prediksi")
st.markdown("""
            - Siapkan gambar jalan yang akan di prediksi tingkat kemacetannya
            - Patikan gambar yang akan di prediksi memiliki format JPG atau JPEG
            - Drag atau unggah manual gambar yang akan diprediksi pada kolom yang disediakan
            - Tekan "Detect" dan tunggu hasil prediksinya keluar """)
