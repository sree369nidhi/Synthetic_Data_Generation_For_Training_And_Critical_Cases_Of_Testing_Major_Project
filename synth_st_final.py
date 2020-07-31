import streamlit as st
import streamlit.components.v1 as components

import PIL.Image
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import time
from PIL import Image

def stylize():
    
    #Footer vanish
    hide_footer_style = """
    <style>
    .reportview-container .main footer {visibility: hidden;}    
    """
    st.markdown(hide_footer_style, unsafe_allow_html=True)

    #Hamburger menu vanish
    hide_streamlit_style = """
                <style>
                #MainMenu {visibility: hidden;}
                footer {visibility: hidden;}
                </style>
                """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True) 

stylize()

st.markdown("<h1 style='text-align: center ; color: black;'>Major Project 2020</h1>", unsafe_allow_html=True)

html_temp = """
<div style="background-color:royalblue;padding:10px;border-radius:10px">
<h1 style="color:white;text-align:center;">Synthetic Data Generation for Training and Critical Cases of Testing</h1>
</div>
"""
components.html(html_temp,)

with open("style.css") as f:
    st.markdown('<style>{}</style>'.format(f.read()), unsafe_allow_html=True)

st.write("<h2 style='text-align: center ; color: black;'>Team : M.Sreenidhi Iyengar, K.Rajesh, K.Geethika, K.Sindhuja</h2>", unsafe_allow_html=True,)

def generate():
	latent_dim = 512
	progan = hub.load("dcgan/").signatures['default']
	
	vector = tf.random.normal([1, latent_dim])
	image = progan(vector)['default']
	image = tf.constant(image)
	image = tf.image.convert_image_dtype(image, tf.uint8)
	return PIL.Image.fromarray(image.numpy()[0])

image = generate()
width = st.slider('Adjustable Image Size', 150, 600, 200)
st.image(image, caption='Synthetic Image', width= width,)

if st.button('Re-Generate New Synthetic Image'):
    generate()