# How to run:
# $ streamlit run main.py

import streamlit as st
import numpy as np
from pandas import read_csv as rcsv, merge, pivot_table, DataFrame as df
from scipy.spatial.distance import pdist, squareform
from warnings import filterwarnings as fw; fw("ignore")
from sklearn.metrics.pairwise import cosine_similarity

# ============
# --- UDFs ---
# ============

def get_similar_product(prd_id):
    if prd_id not in prd_ids:
        return None, None
    else:
        sim_cust = item_sim_df.sort_values(by=prd_id, ascending=False).index[1:]
        sim_score = item_sim_df.sort_values(by=prd_id, ascending=False).loc[:, prd_id].tolist()[1:]
        return sim_cust, sim_score

def predict_rating(cust_id, prd_id, max_neighbor=2):
    _prd, _score = get_similar_product(prd_id)
    prd_arr = np.array([x for x in _prd])
    sim_arr = np.array([x for x in _score])
    
    # select only the product that has already rated by user x
    filtering = user_item_matrix[prd_arr].loc[cust_id] != 0
    
    # calculate the predicted score
    sim_scores = sim_arr[filtering][:max_neighbor]
    closest_rating = user_item_matrix.loc[cust_id][prd_arr[filtering][:max_neighbor]]
    sum_sim_scores = np.sum(sim_arr[filtering][:max_neighbor])

    s = np.dot(sim_scores, closest_rating) / sum_sim_scores
    return s

def get_recommendation(cust_id, n_recommended=2):
    pred_ratings = [predict_rating(cust_id, p) for p in prd_ids]

    # do not recommend products that customer has already rated
    temp = df({'predicted' : pred_ratings, 'prd_id' : prd_ids})
    filt = (user_item_matrix.loc[cust_id] == 0.0)
    temp = temp.loc[filt.values].sort_values(by='predicted', ascending=False)

    return prd[prd.product_id.isin(temp.prd_id[:n_recommended])]

# =================
# --- MAIN CODE ---
# =================

if __name__ == "__main__":
    st.title("TERRA STORE")
    st.markdown("This is a simple (prototype) AI-powered app to predict the next item your customer will buy.")
    st.code("Developed by NICHOLAS DOMINIC (2024)\nnicholas.dominic@binus.ac.id\nhttps://linktr.ee/ndominic")
    
    # load datasets
    ph = rcsv("dataset/purchase_history.csv", delimiter=";")
    prd = rcsv("dataset/product_details.csv", delimiter=";")
    ci = rcsv("dataset/customer_interactions.csv")
    
    # join tables
    merged_data = merge(ci, ph, on="customer_id")
    merged_data = merge(merged_data, prd, on="product_id")

    st.caption("Current Database")
    st.dataframe(merged_data)
    
    # create a distance matrix based on ratings
    user_item_matrix = pivot_table(
        merged_data, index="customer_id", columns="product_id", values="ratings", fill_value=0
    )
    
    # handle missing values (e.g., impute with average rating)
    user_item_matrix.fillna(user_item_matrix.mean(), inplace=True)
    prd_ids = user_item_matrix.columns.to_list()
    cust_ids = user_item_matrix.index.to_list()

    # get similarities between products
    item_sim_df = df(cosine_similarity(user_item_matrix, user_item_matrix), index=prd_ids, columns=prd_ids)

    st.subheader("What will your customers buy next?")
    cid = st.selectbox("Select customer ID: ", ci.customer_id.tolist())
    nr = st.slider("Total recommended product(s): ", 1, 5, 3)
    
    if st.button("PREDICT"):
        st.subheader("Product recommendations for Customer ID = {}".format(cid))
        st.dataframe(get_recommendation(cust_id=cid, n_recommended=nr))

        st.success("Recommendation was successfully generated.")
        st.info("[INFO] You can directly select another Customer ID to generate more recommendations.")
        st.balloons()