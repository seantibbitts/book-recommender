# NDCG at k from this code:
# https://gist.github.com/bwhite/3726239

def dcg_at_k(r, k, method=0):
    """Score is discounted cumulative gain (dcg)
    Relevance is positive real values.  Can use binary
    as the previous methods.
    Example from
    http://www.stanford.edu/class/cs276/handouts/EvaluationNew-handout-6-per.pdf
    >>> r = [3, 2, 3, 0, 0, 1, 2, 2, 3, 0]
    >>> dcg_at_k(r, 1)
    3.0
    >>> dcg_at_k(r, 1, method=1)
    3.0
    >>> dcg_at_k(r, 2)
    5.0
    >>> dcg_at_k(r, 2, method=1)
    4.2618595071429155
    >>> dcg_at_k(r, 10)
    9.6051177391888114
    >>> dcg_at_k(r, 11)
    9.6051177391888114
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
        k: Number of results to consider
        method: If 0 then weights are [1.0, 1.0, 0.6309, 0.5, 0.4307, ...]
                If 1 then weights are [1.0, 0.6309, 0.5, 0.4307, ...]
    Returns:
        Discounted cumulative gain
    """
    import numpy as np
    r = np.asfarray(r)[:k]
    if r.size:
        if method == 0:
            return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
        elif method == 1:
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))
        else:
            raise ValueError('method must be 0 or 1.')
    return 0.


def ndcg_at_k(r, k, method=0):
    """Score is normalized discounted cumulative gain (ndcg)
    Relevance is positive real values.  Can use binary
    as the previous methods.
    Example from
    http://www.stanford.edu/class/cs276/handouts/EvaluationNew-handout-6-per.pdf
    >>> r = [3, 2, 3, 0, 0, 1, 2, 2, 3, 0]
    >>> ndcg_at_k(r, 1)
    1.0
    >>> r = [2, 1, 2, 0]
    >>> ndcg_at_k(r, 4)
    0.9203032077642922
    >>> ndcg_at_k(r, 4, method=1)
    0.96519546960144276
    >>> ndcg_at_k([0], 1)
    0.0
    >>> ndcg_at_k([1], 2)
    1.0
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
        k: Number of results to consider
        method: If 0 then weights are [1.0, 1.0, 0.6309, 0.5, 0.4307, ...]
                If 1 then weights are [1.0, 0.6309, 0.5, 0.4307, ...]
    Returns:
        Normalized discounted cumulative gain
    """
    dcg_max = dcg_at_k(sorted(r, reverse=True), k, method)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k, method) / dcg_max

def get_ndcg(predicted, actual, k):
    '''Returns the mean NDCG at k for a given predicted matrix and actual matrix'''
    import numpy as np
    import pandas as pd
    ndcgs = []
    for i in range(actual.shape[0]):
        sorted_results = pd.DataFrame(np.hstack((predicted[i,np.nonzero(actual[i,:])].T,
                                                 actual[i,np.nonzero(actual[i,:])].T)),
                                      columns = ['predicted','actual']).sort_values('predicted', ascending = False)
        r = sorted_results['actual'].values
        this_ndcg = ndcg_at_k(r, k)
        ndcgs.append(this_ndcg)
    return np.mean(ndcgs)

def get_ndcg_explicit_lightfm(model, dataset, weights, k):
    '''Returns the mean NDCG at k for a given model, dataset and weight matrix'''
    from scipy import sparse
    import pandas as pd
    import numpy as np
    if sparse.issparse(weights):
        actual = weights.toarray()
    else:
        actual = weights
    item_id_map = dataset.mapping()[2]
    all_item_ids = sorted(list(item_id_map.values()))
    ndcgs = []
    rs = []
    for user_id in range(weights.shape[0]):
        predicted = model.predict(user_id, all_item_ids)
        nonzero_actual = np.nonzero(actual[user_id])
        sort_inds = predicted[nonzero_actual].argsort()[::-1]
        r = actual[user_id][nonzero_actual][sort_inds]
        rs.append(r)
        this_ndcg = ndcg_at_k(r, k)
        ndcgs.append(this_ndcg)
    return np.mean(ndcgs)


def predict_for_user(ratings, all_predicted_df, books, user_id = 1,
                     num_recs = 5):
    '''Takes a long dataframe of ratings (not pivoted), a reconstructed (pivoted) dataframe of predictions, a long dataframe of books,
    a user id, and a number of recommendations and returns the top recommended items and the top rated items for comparison'''
    import pandas as pd
    orig_user_ratings = ratings[ratings['user_id'] == user_id]
    user_predictions = all_predicted_df.loc[user_id].sort_values(ascending = False)
    filtered_books =    books[~books['book_id'].isin(orig_user_ratings['book_id'].unique())]
    user_w_top_books = pd.merge(orig_user_ratings, books, how = 'left', on = 'book_id').sort_values('rating', ascending = False).head(5)[['book_id','authors','original_publication_year','original_title','title','rating']]
    
    recs = pd.merge(filtered_books, user_predictions.reset_index(), how = 'left', on = 'book_id').rename(columns = {user_id: 'predicted_rating'}).sort_values('predicted_rating', ascending = False).head(5)[['book_id','authors','original_publication_year','original_title','title','predicted_rating']]
    
    return user_w_top_books, recs

def predict_for_user_implicit_lightfm(model, dataset, interactions, books, item_features=None, model_user_id = 0, num_recs = 5):
    '''Takes a trained LightFM model, a LightFM dataset, the dataframes of ratings, to-read books and book information,
    the internal model user ID and a number of recommendations and returns the top 5 recommended titles that have not
    been interacted with'''
    import numpy as np
    import pandas as pd
    user_id_map = dataset.mapping()[0]
    item_id_map = dataset.mapping()[2]
    user_id_rev_map = {v:k for k, v in user_id_map.items()}
    item_id_rev_map = {v:k for k, v in item_id_map.items()}

    all_item_ids = list(item_id_map.values())

    raw_predictions = model.predict(model_user_id, all_item_ids, item_features = item_features)

    user_predictions = pd.DataFrame({'predictions':raw_predictions, 'model_book_id':all_item_ids})

    user_noninteractions = np.where(interactions.toarray()[model_user_id]==0)[0]

    user_recommendations = user_predictions[user_predictions['model_book_id'].isin(user_noninteractions) &
    (user_predictions['predictions']>0)].sort_values('predictions', ascending = False).iloc[:num_recs,:]
    user_recommendations['book_id'] = user_recommendations['model_book_id'].map(item_id_rev_map)

    return user_recommendations.merge(books[['book_id','authors','title','average_rating','image_url']])


def predict_for_user_explicit_lightfm(model, dataset, interactions, books, item_features=None, model_user_id = 0, num_recs = 5):
    '''Takes a trained LightFM model, a LightFM dataset, the dataframe of book information,
    the internal model user ID and a number of recommendations and returns the top 5 recommended titles that have not
    been interacted with'''
    import numpy as np
    import pandas as pd
    user_id_map = dataset.mapping()[0]
    item_id_map = dataset.mapping()[2]
    user_id_rev_map = {v:k for k, v in user_id_map.items()}
    item_id_rev_map = {v:k for k, v in item_id_map.items()}

    all_item_ids = list(item_id_map.values())

    raw_predictions = model.predict(model_user_id, all_item_ids, item_features = item_features)

    user_predictions = pd.DataFrame({'predictions':raw_predictions, 'model_book_id':all_item_ids})

    user_noninteractions = np.where(interactions.toarray()[model_user_id]==0)[0]

    user_recommendations = user_predictions[user_predictions['model_book_id'].isin(user_noninteractions) &
    (user_predictions['predictions']>=3)].sort_values('predictions', ascending = False).iloc[:num_recs,:]
    user_recommendations['book_id'] = user_recommendations['model_book_id'].map(item_id_rev_map)

    return user_recommendations.merge(books[['book_id','authors','title','average_rating','image_url','goodreads_book_id']])

def predict_for_user_knn_lightfm(lightfm_model, lightfm_dataset, lightfm_weights, books, user_vector, item_features=None, num_recs=5):
    # from scipy import sparse
    import numpy as np
    import pandas as pd
    from sklearn.metrics.pairwise import cosine_similarity

    user_id_map = lightfm_dataset.mapping()[0]
    item_id_map = lightfm_dataset.mapping()[2]
    user_id_rev_map = {v:k for k, v in user_id_map.items()}
    item_id_rev_map = {v:k for k, v in item_id_map.items()}

    all_item_ids = sorted(list(item_id_map.values()))
    # Compute cosine similarity between each row of weights and the user_vector
    # if sparse.issparse(lightfm_weights):
    #     weights = lightfm_weights.toarray()
    # else
    #     weights = lightfm_weights
    similarity = cosine_similarity(lightfm_weights, user_vector)
    pred_list = []
    for user_id in range(lightfm_weights.shape[0]):
        predicted = lightfm_model.predict(user_id, all_item_ids, item_features=item_features)
        pred_list.append(predicted)
    all_predicted = np.array(pred_list)
    all_pred_sim = all_predicted * similarity
    books_pred = all_pred_sim.sum(axis = 0)

    user_predictions = pd.DataFrame({'predictions':books_pred, 'model_book_id':all_item_ids})

    user_noninteractions = np.where(user_vector==0)[0]

    user_recommendations = user_predictions[user_predictions['model_book_id'].isin(user_noninteractions) &
                                            (user_predictions['predictions']>=3)].sort_values('predictions', ascending=False).iloc[:num_recs]
    user_recommendations['book_id'] = user_recommendations['model_book_id'].map(item_id_rev_map)

    return user_recommendations.merge(books[['book_id','authors','title','average_rating','image_url','goodreads_book_id']])