def train_test_split(ratings, size=10, max_percent = 0.8):
    '''Split a recommender matrix 'ratings' into train and test matrices of the same dimensions
    by removing random entries for each user (number removed = 'size') 
    Slightly modified from method found here:
    https://blog.insightdatascience.com/explicit-matrix-factorization-als-sgd-and-all-that-jazz-b00e4d9b21ea'''
    import numpy as np
    test = np.zeros(ratings.shape)
    train = ratings.copy()
    for user in range(ratings.shape[0]):
        max_size = int(max_percent * ratings[user, :].nonzero()[0].shape[0])
        if size <= max_size:
            test_ratings = np.random.choice(ratings[user, :].nonzero()[0], 
                                            size=size, 
                                            replace=False)
        else:
            test_ratings = np.random.choice(ratings[user, :].nonzero()[0], 
                                            size=max_size, 
                                            replace=False)
        train[user, test_ratings] = 0.
        test[user, test_ratings] = ratings[user, test_ratings]
        
    # Test and training are truly disjoint
    assert(np.all((train * test) == 0)) 
    return train, test

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
    for user_id in range(weights.shape[0]):
        predicted = model.predict(user_id, all_item_ids)
        nonzero_actual = np.nonzero(actual[user_id])
        sort_inds = predicted[nonzero_actual].argsort()[::-1]
        r = actual[user_id][nonzero_actual][sort_inds]
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

    # user_id = user_id_rev_map[model_user_id]

    all_item_ids = list(item_id_map.values())

    raw_predictions = model.predict(model_user_id, all_item_ids, item_features = item_features)

    user_predictions = pd.DataFrame({'predictions':raw_predictions, 'model_book_id':all_item_ids})

    #user_interactions = ratings.loc[ratings['user_id']==user_id, ['user_id','book_id']].append(to_read[to_read['user_id']==user_id])
    #user_interactions['model_book_id'] = user_interactions['book_id'].map(item_id_map)

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

    # user_id = user_id_rev_map[model_user_id]

    all_item_ids = list(item_id_map.values())

    raw_predictions = model.predict(model_user_id, all_item_ids, item_features = item_features)

    user_predictions = pd.DataFrame({'predictions':raw_predictions, 'model_book_id':all_item_ids})

    #user_interactions = ratings.loc[ratings['user_id']==user_id, ['user_id','book_id']].append(to_read[to_read['user_id']==user_id])
    #user_interactions['model_book_id'] = user_interactions['book_id'].map(item_id_map)

    user_noninteractions = np.where(interactions.toarray()[model_user_id]==0)[0]

    user_recommendations = user_predictions[user_predictions['model_book_id'].isin(user_noninteractions) &
    (user_predictions['predictions']>=3)].sort_values('predictions', ascending = False).iloc[:num_recs,:]
    user_recommendations['book_id'] = user_recommendations['model_book_id'].map(item_id_rev_map)

    return user_recommendations.merge(books[['book_id','authors','title','average_rating','image_url','goodreads_book_id']])


def dot_on_disk(A, B, name, file_name, chunk_size=None, chunk_multiple=None):
    '''Performs matrix multiplication in blocks, saving the result iteratively to disk instead of holding it in memory.
    Solution from this StackOverflow post:
    https://stackoverflow.com/questions/19684575/matrix-multiplication-using-hdf5'''
    import numpy as np
    import tables
    if chunk_size is None:
        chunk_size = 1000
    if chunk_multiple is None:
        chunk_multiple = 1
    shape = (A.shape[0],B.shape[1])
    chunk_shape = (chunk_size,chunk_size)
    block_size = chunk_multiple * chunk_size
    atom = tables.Float64Atom()
    h5f_C = tables.open_file(file_name, mode='w')
    C = h5f_C.create_carray(h5f_C.root, name, atom=atom, shape=shape, chunkshape=chunk_shape)

    sz = block_size

    for i in range(0, A.shape[0], sz):
        for j in range(0, B.shape[1], sz):
            for k in range(0, A.shape[1], sz):
                C[i:i+sz,j:j+sz] += np.dot(A[i:i+sz,k:k+sz],B[k:k+sz,j:j+sz])
    
    output = np.array(C)
    h5f_C.close()

    return output

def loguniform(low=0, high=1, size=None):
    import numpy as np
    '''Returns a log-uniform distribution of size = 'size' that starts at 'low' and goes to 'high'
    From this StackOverflow post:
    https://stackoverflow.com/questions/43977717/how-do-i-generate-log-uniform-distribution-in-python'''
    return np.exp(np.random.uniform(low, high, size))