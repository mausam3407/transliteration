import joblib

import tensorflow as tf
import streamlit as st

from transliteration import Translator, UNIT

en = joblib.load('metadata/english')
hi = joblib.load('metadata/hindi')

model = Translator(UNIT, en, hi)

model.load_weights('metadata/weights_2')
st.title("Roman Devanagari Transliterator")

input = st.text_input("Input", value = "Dukh")

st.write("Output")

st.subheader(model.translate([input])[0])
