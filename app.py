import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.datasets import load_iris

def predict(data):
    clf = joblib.load("rf_model.sav")
    return clf.predict(data)

# Function to map classes to images
def class_to_image(class_name):
    if class_name == "setosa":
        return "images/setosa.jpg"  # Replace with the actual path to your setosa image
    elif class_name == "versicolor":
        return "images/versicolor.jpg"  # Replace with the actual path to your versicolor image
    elif class_name == "virginica":
        return "images/virginica.jpg"  # Replace with the actual path to your virginica image
    return None

st.title('Classifying Iris Flowers')
st.markdown('Model to classify iris flowers into \
     (setosa, versicolor, virginica) based on their sepal/petal \
    and length/width.')

st.header("Plant Features")
col1, col2 = st.columns(2)

with col1:
    st.text("Sepal characteristics")
    sepal_l = st.slider('Sepal length (cm)', 1.0, 8.0, 0.5)
    sepal_w = st.slider('Sepal width (cm)', 2.0, 4.4, 0.5)

with col2:
    st.text("Petal characteristics")
    petal_l = st.slider('Petal length (cm)', 1.0, 7.0, 0.5)
    petal_w = st.slider('Petal width (cm)', 0.1, 2.5, 0.5)

st.text('')
if st.button("Predict type of Iris"):
    result = predict(
        np.array([[sepal_l, sepal_w, petal_l, petal_w]]))
    # result[0] can be an integer label (0,1,2) or a string; map to class name
    raw_pred = result[0]

    # load target names from Iris dataset
    iris = load_iris()
    target_names = list(iris.target_names)

    # determine class_name robustly
    if isinstance(raw_pred, (np.integer, int)):
        class_name = target_names[int(raw_pred)]
    else:
        # if prediction is a bytestring or a string possibly like '0-setosa', normalize
        pred_str = str(raw_pred)
        if "-" in pred_str:
            # handle formats like '0-setosa' or 'label-setosa'
            class_name = pred_str.split("-", 1)[1]
        else:
            class_name = pred_str

    st.text(f"Predicted: {class_name}")

    # Display the image/icon corresponding to the predicted class
    image_path = class_to_image(class_name)
    if image_path:
        st.image(image_path, use_column_width=True)
    else:
        st.warning(f"No image available for class '{class_name}'")


st.text('')
