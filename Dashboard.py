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


def display_image_from_url(url):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    return img

img_url = "https://github.com/zim18/Bandung-Density-Traffic-Classification/blob/de1bd6feae9f805ab4146a23945f8b4e12b86494/Assets/Images/Screenshot(6917).png"

col1,col2=st.columns(2)
with col1:
    img = display_image_from_url(img_url)
    st.image(img, caption='Gambar dari GitHub')
            
with col2:
    st.write("**Kepadatan lalu lintas merupakan sebuah kondisi ketika jumlah kendaraan yang melintas melebihi kapasitas jalan yang tersedia. Kepadatan ini merupakan salah satu penyebab dari kemacetan yang terjadi terutama pada kota-kota besar. Pada proyek ini kepadatan lalu lintas yang terjadi akan diklasifikasi dengan model Convolutional Neural Network menggunakan Arsitektur EfficientNet menggunakan dataset gambar yang dikumpulkan dari persimpangan jalan Kota Bandung secara manual melalui website CCTV Pemantauan Lingkungan Kota Bandung. **")


st.subheader("Langkah Melakukan Prediksi")
st.markdown("""
            - Siapkan gambar jalan yang akan di prediksi tingkat kemacetannya
            - Patikan gambar yang akan di prediksi memiliki format JPG atau JPEG
            - Drag atau unggah manual gambar yang akan diprediksi pada kolom yang disediakan
            - Tekan "Detect" dan tunggu hasil prediksinya keluar """)
