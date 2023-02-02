from numpy.core.numeric import True_
from sklearn import metrics
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error 
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.metrics import precision_score, recall_score

# @st.cache(persist= True)
# def load():
#     data= pd.read_csv("mushrooms.csv")
#     label= LabelEncoder()
#     for i in data.columns:
#         data[i] = label.fit_transform(data[i])
#     return data
# df = load()

@st.cache(persist= True)
def read_data():
    uploaded_file = st.file_uploader("Choose a file")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        label= LabelEncoder()
        for i in df.columns:
            df[i] = label.fit_transform(df[i])
        st.write(df)
    return df
df=read_data()

def main():
    st.title("Introduction to building Streamlit WebApp")
    st.sidebar.title("This is the sidebar")
    st.sidebar.markdown("Let’s start with binary classification!!")
if __name__ == '__main__':
    main()
