import os
import sys
import numpy as np
import pandas as pd
import flask
import pickle
import json
import time
import datetime
from scipy import sparse
from flask import Flask, render_template, request, jsonify, redirect, url_for
from lightfm import LightFM
from functions.helper_functions import (
    predict_for_user_implicit_lightfm,
    predict_for_user_explicit_lightfm,
    predict_for_user_knn_lightfm,
    get_ndcg_explicit_lightfm,
    ndcg_at_k,
    reviews_to_df,
    get_reviews,
    get_user_array,
    get_implicit_interactions
)
import keyring
api_key = keyring.get_password('Goodreads API Key', 'Goodreads API Key')

app=Flask(__name__)

# ratings = pd.read_csv('model/ratings.csv')
# to_read = pd.read_csv('model/to_read.csv')
books = pd.read_csv('model/books.csv')
# with open('model/implicit_model.pkl', 'rb') as f:
#     model_implicit = pickle.load(f)
model_implicit = LightFM(no_components=50, loss='warp', random_state=42)
with open('model/implicit_dataset.pkl', 'rb') as f:
    dataset_implicit = pickle.load(f)
with open('model/implicit_interactions.pkl', 'rb') as f:
    interactions_implicit = pickle.load(f)
# # This model has 10 components, uses the full item_features matrix and takes 1.5 minutes to train
# with open('model/explicit_model.pkl', 'rb') as f:
#     model_explicit = pickle.load(f)
# # This has 10 components and uses the truncated item_features matrix
# # Does not seem to have helped the slowness
# with open('model/model_trunc.pkl', 'rb') as f:
#     model_explicit = pickle.load(f)
# # This model uses adadelta instead of adagrad to run SGD
# with open('model/model_adadelta.pkl', 'rb') as f:
#     model_explicit = pickle.load(f)
# # This is an edited/extended/custom LightFM model
# # 10 components 1 epoch
# with open('model/model_lfe-10-components-1-epoch.pkl', 'rb') as f:
#     model_explicit = pickle.load(f)
# # This is an edited/extended/custom LightFM model
# # 10 components 200 epochs
# with open('model/model_lfe-10-components-200-epoch.pkl', 'rb') as f:
#     model_explicit = pickle.load(f)
model_explicit = LightFM(no_components=50, loss='warp', random_state=42)
with open('model/explicit_dataset.pkl', 'rb') as f:
    dataset_explicit = pickle.load(f)
with open('model/explicit_interactions.pkl', 'rb') as f:
    interactions_explicit = pickle.load(f)
with open('model/weights.pkl', 'rb') as f:
    weights_explicit = pickle.load(f)
# # This is the full item_features matrix
with open('model/item_features.pkl', 'rb') as f:
    item_features = pickle.load(f)
# # This is the truncated item features matrix
# with open('model/item_features_trunc.pkl', 'rb') as f:
#     item_features = pickle.load(f)

@app.route('/')
@app.route('/index')
@app.route('/book-recommender')
def index():
    return render_template('book-recommender-implicit.html')

@app.route('/implicit-recommendations', methods = ['GET','POST'])
def implicit_recs():
    if request.method == 'GET':
        user_id = request.args['goodreads-id']
        print(f'goodreads ID: {user_id}')
        sys.stdout.flush()

        start_time = time.time()
        print('Now requesting user reviews from Goodreads')
        sys.stdout.flush()
        reviews_df = get_implicit_interactions(api_key, user_id, books, dataset_implicit, shelf_names = ['read', 'currently-reading', 'to-read'])
        time_reviews_received = time.time()
        print('User reviews acquired')
        print(f'Time taken: {datetime.timedelta(seconds=time_reviews_received-start_time)}')
        sys.stdout.flush()
        interactions_array = get_user_array(reviews_df, explicit=False)
        print('Loaded interactions array')
        sys.stdout.flush()

        interactions_implicit_arr = interactions_implicit.toarray()
        print('Cast COO matrix as ndarray')
        sys.stdout.flush()
        interactions_implicit_arr[53424] = interactions_array
        print('Set last row of ndarray equal to interactions_array')
        sys.stdout.flush()
        interactions_implicit_aug = sparse.coo_matrix(interactions_implicit_arr)
        print('Cast ndarray as COO matrix')

        print('Now fitting updated matrix to model')
        sys.stdout.flush()
        # This takes between 1.5 to 8 minutes depending on the complexity of the model
        start_fitting_time = time.time()
        model_implicit.fit_partial(interactions_implicit_aug, item_features=item_features, epochs=1)
        end_fitting_time = time.time()
        print('Model fitting complete')
        print(f'Time to fit: {datetime.timedelta(seconds=end_fitting_time-start_fitting_time)}')
        sys.stdout.flush()
        predictions = predict_for_user_implicit_lightfm(model_implicit, dataset_implicit, interactions_implicit_aug,
        books, item_features=item_features, model_user_id = 53424, num_recs = 24).to_dict(orient='records')
        finish_time = time.time()
        time_elapsed = finish_time - start_time
        print('Predictions generated')
        print(f'Time to generate predictions: {datetime.timedelta(seconds=finish_time-end_fitting_time)}')
        print(f'Time to run: {datetime.timedelta(seconds=time_elapsed)}')

        return render_template('implicit_recommendations3.html', predictions = predictions)
    
    # if request.method == 'POST':
    #     start_time = time.time()
    #     msg = request.json['msg']
    #     print(msg)
    #     sys.stdout.flush()
    #     weights_array = json.loads(str(request.json['array']))
    #     interactions_array = np.where(np.array(weights_array)!=0, 1, 0)
    #     print('Loaded arrays')
    #     sys.stdout.flush()

    #     assert len(weights_array) == weights_explicit.shape[1]
    #     assert len(interactions_array) == interactions_explicit.shape[1]

    #     weights_explicit_arr = weights_explicit.toarray()
    #     interactions_explicit_arr = interactions_explicit.toarray()
    #     print('Cast COO matrices as ndarray')
    #     sys.stdout.flush()
    #     weights_explicit_arr[53424] = weights_array
    #     interactions_explicit_arr[53424] = interactions_array
    #     print('Set last rows of ndarrays equal to weights_array/interactions_array')
    #     sys.stdout.flush()
    #     weights_explicit_aug = sparse.coo_matrix(weights_explicit_arr)
    #     interactions_explicit_aug = sparse.coo_matrix(interactions_explicit_arr)
    #     print('Cast ndarrays as COO matrices')
    #     sys.stdout.flush()
    #     if msg=='update':
    #         print('Now updating last matrix row')
    #         sys.stdout.flush()
    #         model_explicit.fit_partial_by_row(53424, interactions_explicit_aug, sample_weight = weights_explicit_aug, item_features=item_features, epochs=50)
    #     else:
    #         print('Now fitting updated matrix to model')
    #         sys.stdout.flush()
    #         # This takes between 1.5 to 8 minutes depending on the complexity of the model
    #         model_explicit.fit_partial(interactions_explicit_aug, sample_weight = weights_explicit_aug, item_features=item_features, epochs=5)
    #     print('Model fitting complete')
    #     sys.stdout.flush()
    #     predictions = predict_for_user_explicit_lightfm(model_explicit, dataset_explicit, interactions_explicit_aug,
    #     books, item_features=item_features, model_user_id = 53424, num_recs = 24).to_dict(orient='records')
    #     time_elapsed = time.time() - start_time
    #     print('Predictions generated')
    #     print(f'Time to run: {datetime.timedelta(seconds=time_elapsed)}')
    #     sys.stdout.flush()
    #     item_id_map = dataset_explicit.mapping()[2]
    #     all_item_ids = sorted(list(item_id_map.values()))
    #     predicted = model_explicit.predict(53424, all_item_ids)
    #     actual = weights_explicit_arr[53424]
    #     nonzero_actual = np.nonzero(actual)
    #     sort_inds = predicted[nonzero_actual].argsort()[::-1]
    #     r = actual[nonzero_actual][sort_inds]
    #     ndcg = ndcg_at_k(r, 5)
    #     print(f'nDCG for current user: {ndcg}')

    #     return render_template('explicit_recommendations2.html', predictions = predictions)

if __name__ == "__main__":
    app.run()