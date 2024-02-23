# Item-based Collaborative Filtering Recommender System

## Project Description
**Business Context:** You are hired as a Data Science and AI for an e-commerce company named "Terra Store." Terra Store is looking to enhance its marketing strategy by predicting customer purchase behavior based on historical data. The company wants to build an AI-powered application that can provide insights into which products a customer is likely to purchase next.

## Setup
### Prerequisite Packages (Dependencies)
- numpy==1.26.4
- pandas==2.2.0
- scikit-learn==1.4.1
- scipy==1.12.0
- streamlit==1.31.1

### Environment
| | |
| --- | --- |
| CPU | AMD Ryzen 5 5600H |
| RAM | 16,384 MB |
| OS | Windows 11 64-bit (10.0, Build 22621) |

## Dataset
Check folder **datasets/** for more info.

## Methodology
Internal documentations can be found within the code (redirect to **docs/main_ipynb.pdf**).
1. All datasets were loaded and preprocessed (e.g., impute missing values, if any).
2. All tables were joined and then pivoted to create **a raw distance matrix**.
3. **Cosine similarity** was used to find correlation between items.
4. For ratings prediction,
   - The similarity scores (`sim_arr`) was filtered using the `filtering` array to keep only the scores for products that the user has rated.
   - Then, the top `max_neighbor` most similar products was selected based on their similarity scores.
   - The user's ratings for these top `max_neighbor` products was retrieved from the `user_item_matrix`.
   - Finally, the predicted rating is calculated as the weighted average of the user's ratings for the top similar products, where the weights are the similarity scores. The weights are normalized by their sum to ensure they add up to 1.
5. ...
6. ...

## How to Run the App (Streamlit)
1. Create the Python virtual environment (VENV) first: `python -m venv <VENV_NAME>`.
2. Activate your VENV and install all packages (from `requirements.txt` file).
3. To run the app, execute this command: `streamlit run main.py`.

![](https://github.com/NicholasDominic/sample-collaborative-filt-recsyst/blob/main/pics/run_streamlit.png)

## App Display
Notice that the highlighted red box is for the user's inputs.

![](https://github.com/NicholasDominic/sample-collaborative-filt-recsyst/blob/main/pics/terra_store_app.png)

## Remarks
- Due to limited datasets, the result may seems inaccurate (or even `nan`).
- For forthcoming works, kNN can be used to measure the similarity distance between items.

## Contact
- Email: nicholas.dominic@binus.ac.id
- Portfolio: https://linktr.ee/ndominic
