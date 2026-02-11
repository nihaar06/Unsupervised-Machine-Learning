import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

st.header("News Topic Discovery Dashboard")
st.subheader("This system uses Hierarchical Clustering to automatically group similar news articles based on textual similarity.")

st.sidebar.title("Input section")
f=st.sidebar.file_uploader("Upload your news dataset here",type=['csv'])

from sklearn.feature_extraction.text import TfidfVectorizer
if f is not None:
    df=pd.read_csv(f,encoding='latin-1',header=None)
    t_cols=df.select_dtypes(include='object').columns[0]
    tf_f=st.sidebar.slider("Select max number of features",min_value=100,max_value=2000,value=1000,step=100)
    st_cb=st.sidebar.checkbox("Remove stop words")
    stop_words=None
    if st_cb:
        stop_words='english'
    n_grams=st.sidebar.selectbox("Select n-gram range",options=['Unigrams','Bigrams','Unigram+Bigrams'])
    if n_grams == "Unigrams":
        ngram_range = (1, 1)
    elif n_grams == "Bigrams":
        ngram_range = (2, 2)
    else:
        ngram_range = (1, 2)
    tf = TfidfVectorizer(
        max_features=tf_f,
        stop_words=stop_words,
        ngram_range=ngram_range
    )
    X=tf.fit_transform(df[t_cols]).toarray()
    m=st.sidebar.selectbox("Select the linkage method for hierarchical clustering",options=['ward','complete','average','single'])
    dist=st.sidebar.selectbox("Distance Metric",options=['euclidean'])
    sample_size=st.sidebar.slider("Select number of articles for Dendrogram",min_value=20,max_value=200,value=100,step=10)
    from scipy.cluster.hierarchy import dendrogram,linkage
    gen_d=st.sidebar.button("Generate Dendrogram")
    if gen_d:
        X_s=X[:sample_size]
        fig,ax=plt.subplots(figsize=(10,5))
        dendrogram=dendrogram(linkage(X_s,method=m),ax=ax)
        ax.set_xlabel("Data points")
        ax.set_ylabel("Distance")
        st.pyplot(fig)

    clusters=st.sidebar.slider("Select number of clusters",min_value=2,max_value=10,value=3,step=1)
    from sklearn.cluster import AgglomerativeClustering
    if st.sidebar.button("Perform Clustering"):
        ag=AgglomerativeClustering(n_clusters=clusters,linkage=m,metric=dist)  
        y_hc=ag.fit_predict(X)
        from sklearn.metrics import silhouette_score
        score=silhouette_score(X,y_hc)
        st.subheader("Clustering Results")
        st.write("Silhouette Score:", round(score, 3))