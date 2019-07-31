import pyximport
pyximport.install()
import numpy as np
import scipy.sparse as sp
# from lightfm import LightFM
from .fit_warp_ext import (
    CSRMatrix,
    FastLightFM,
    fit_warp, 
    predict_lightfm,
    predict_ranks
)

CYTHON_DTYPE = np.float32

class LightFM_ext(object):
    def __init__(
        self,
        no_components=10,
        k=5,
        n=10,
        learning_schedule="adagrad",
        loss="logistic",
        learning_rate=0.05,
        rho=0.95,
        epsilon=1e-6,
        item_alpha=0.0,
        user_alpha=0.0,
        max_sampled=10,
        random_state=None,
    ):

        assert item_alpha >= 0.0
        assert user_alpha >= 0.0
        assert no_components > 0
        assert k > 0
        assert n > 0
        assert 0 < rho < 1
        assert epsilon >= 0
        assert learning_schedule in ("adagrad", "adadelta")
        assert loss in ("logistic", "warp", "bpr", "warp-kos")

        if max_sampled < 1:
            raise ValueError("max_sampled must be a positive integer")

        self.loss = loss
        self.learning_schedule = learning_schedule

        self.no_components = no_components
        self.learning_rate = learning_rate

        self.k = int(k)
        self.n = int(n)

        self.rho = rho
        self.epsilon = epsilon
        self.max_sampled = max_sampled

        self.item_alpha = item_alpha
        self.user_alpha = user_alpha

        if random_state is None:
            self.random_state = np.random.RandomState()
        elif isinstance(random_state, np.random.RandomState):
            self.random_state = random_state
        else:
            self.random_state = np.random.RandomState(random_state)

        self._reset_state()
    
    def _reset_state(self):

        self.item_embeddings = None
        self.item_embedding_gradients = None
        self.item_embedding_momentum = None
        self.item_biases = None
        self.item_bias_gradients = None
        self.item_bias_momentum = None

        self.user_embeddings = None
        self.user_embedding_gradients = None
        self.user_embedding_momentum = None
        self.user_biases = None
        self.user_bias_gradients = None
        self.user_bias_momentum = None

    def _check_initialized(self):

        for var in (
            self.item_embeddings,
            self.item_embedding_gradients,
            self.item_embedding_momentum,
            self.item_biases,
            self.item_bias_gradients,
            self.item_bias_momentum,
            self.user_embeddings,
            self.user_embedding_gradients,
            self.user_embedding_momentum,
            self.user_biases,
            self.user_bias_gradients,
            self.user_bias_momentum,
        ):

            if var is None:
                raise ValueError(
                    "You must fit the model before " "trying to obtain predictions."
                )

    def _initialize(self, no_components, no_item_features, no_user_features):
        """
        Initialise internal latent representations.
        """

        # Initialise item features.
        self.item_embeddings = (
            (self.random_state.rand(no_item_features, no_components) - 0.5)
            / no_components
        ).astype(np.float32)
        self.item_embedding_gradients = np.zeros_like(self.item_embeddings)
        self.item_embedding_momentum = np.zeros_like(self.item_embeddings)
        self.item_biases = np.zeros(no_item_features, dtype=np.float32)
        self.item_bias_gradients = np.zeros_like(self.item_biases)
        self.item_bias_momentum = np.zeros_like(self.item_biases)

        # Initialise user features.
        self.user_embeddings = (
            (self.random_state.rand(no_user_features, no_components) - 0.5)
            / no_components
        ).astype(np.float32)
        self.user_embedding_gradients = np.zeros_like(self.user_embeddings)
        self.user_embedding_momentum = np.zeros_like(self.user_embeddings)
        self.user_biases = np.zeros(no_user_features, dtype=np.float32)
        self.user_bias_gradients = np.zeros_like(self.user_biases)
        self.user_bias_momentum = np.zeros_like(self.user_biases)

        if self.learning_schedule == "adagrad":
            self.item_embedding_gradients += 1
            self.item_bias_gradients += 1
            self.user_embedding_gradients += 1
            self.user_bias_gradients += 1

    def _construct_feature_matrices(
        self, n_users, n_items, user_features, item_features
    ):

        if user_features is None:
            user_features = sp.identity(n_users, dtype=CYTHON_DTYPE, format="csr")
        else:
            user_features = user_features.tocsr()

        if item_features is None:
            item_features = sp.identity(n_items, dtype=CYTHON_DTYPE, format="csr")
        else:
            item_features = item_features.tocsr()

        if n_users > user_features.shape[0]:
            raise Exception(
                "Number of user feature rows does not equal the number of users"
            )

        if n_items > item_features.shape[0]:
            raise Exception(
                "Number of item feature rows does not equal the number of items"
            )

        # If we already have embeddings, verify that
        # we have them for all the supplied features
        if self.user_embeddings is not None:
            if not self.user_embeddings.shape[0] >= user_features.shape[1]:
                raise ValueError(
                    "The user feature matrix specifies more "
                    "features than there are estimated "
                    "feature embeddings: {} vs {}.".format(
                        self.user_embeddings.shape[0], user_features.shape[1]
                    )
                )

        if self.item_embeddings is not None:
            if not self.item_embeddings.shape[0] >= item_features.shape[1]:
                raise ValueError(
                    "The item feature matrix specifies more "
                    "features than there are estimated "
                    "feature embeddings: {} vs {}.".format(
                        self.item_embeddings.shape[0], item_features.shape[1]
                    )
                )

        user_features = self._to_cython_dtype(user_features)
        item_features = self._to_cython_dtype(item_features)

        return user_features, item_features

    def _get_positives_lookup_matrix(self, interactions):

        mat = interactions.tocsr()

        if not mat.has_sorted_indices:
            return mat.sorted_indices()
        else:
            return mat

    def _to_cython_dtype(self, mat):

        if mat.dtype != CYTHON_DTYPE:
            return mat.astype(CYTHON_DTYPE)
        else:
            return mat

    def _process_sample_weight(self, interactions, sample_weight):

        if sample_weight is not None:

            if self.loss == "warp-kos":
                raise NotImplementedError(
                    "k-OS loss with sample weights " "not implemented."
                )

            if not isinstance(sample_weight, sp.coo_matrix):
                raise ValueError("Sample_weight must be a COO matrix.")

            if sample_weight.shape != interactions.shape:
                raise ValueError(
                    "Sample weight and interactions " "matrices must be the same shape"
                )

            if not (
                np.array_equal(interactions.row, sample_weight.row)
                and np.array_equal(interactions.col, sample_weight.col)
            ):
                raise ValueError(
                    "Sample weight and interaction matrix "
                    "entries must be in the same order"
                )

            if sample_weight.data.dtype != CYTHON_DTYPE:
                sample_weight_data = sample_weight.data.astype(CYTHON_DTYPE)
            else:
                sample_weight_data = sample_weight.data
        else:
            if np.array_equiv(interactions.data, 1.0):
                # Re-use interactions data if they are all
                # ones
                sample_weight_data = interactions.data
            else:
                # Otherwise allocate a new array of ones
                sample_weight_data = np.ones_like(interactions.data, dtype=CYTHON_DTYPE)

        return sample_weight_data

    def _get_lightfm_data(self):

        lightfm_data = FastLightFM(
            self.item_embeddings,
            self.item_embedding_gradients,
            self.item_embedding_momentum,
            self.item_biases,
            self.item_bias_gradients,
            self.item_bias_momentum,
            self.user_embeddings,
            self.user_embedding_gradients,
            self.user_embedding_momentum,
            self.user_biases,
            self.user_bias_gradients,
            self.user_bias_momentum,
            self.no_components,
            int(self.learning_schedule == "adadelta"),
            self.learning_rate,
            self.rho,
            self.epsilon,
            self.max_sampled,
        )

        return lightfm_data

    def _check_finite(self):

        for parameter in (
            self.item_embeddings,
            self.item_biases,
            self.user_embeddings,
            self.user_biases,
        ):
            # A sum of an array that contains non-finite values
            # will also be non-finite, and we avoid creating a
            # large boolean temporary.
            if not np.isfinite(np.sum(parameter)):
                raise ValueError(
                    "Not all estimated parameters are finite,"
                    " your model may have diverged. Try decreasing"
                    " the learning rate or normalising feature values"
                    " and sample weights"
                )

    def _check_input_finite(self, data):

        if not np.isfinite(np.sum(data)):
            raise ValueError(
                "Not all input values are finite. "
                "Check the input for NaNs and infinite values."
            )

    @staticmethod
    def _progress(n, verbose):
        # Use `tqdm` if available,
        # otherwise fallback to `range()`.
        if not verbose:
            return range(n)

        try:
            from tqdm import trange

            return trange(n, desc="Epoch")
        except ImportError:

            def verbose_range():
                for i in range(n):
                    print("Epoch {}".format(i))
                    yield i

            return verbose_range()

    def fit_partial_by_row(
        self,
        user_id,
        interactions,
        user_features=None,
        item_features=None,
        sample_weight=None,
        epochs=1,
        num_threads=1,
        verbose=False
    ):
        """Fits only a single row of the model"""
        # We need this in the COO format.
        # If that's already true, this is a no-op.
        interactions = interactions.tocoo()

        if interactions.dtype != CYTHON_DTYPE:
            interactions.data = interactions.data.astype(CYTHON_DTYPE)

        sample_weight_data = self._process_sample_weight(interactions, sample_weight)

        n_users, n_items = interactions.shape
        (user_features, item_features) = self._construct_feature_matrices(
            n_users, n_items, user_features, item_features
        )

        for input_data in (
            user_features.data,
            item_features.data,
            interactions.data,
            sample_weight_data,
        ):
            self._check_input_finite(input_data)
        if self.item_embeddings is None:
            # Initialise latent factors only if this is the first call
            # to fit_partial.
            self._initialize(
                self.no_components, item_features.shape[1], user_features.shape[1]
            )

        # Check that the dimensionality of the feature matrices has
        # not changed between runs.
        if not item_features.shape[1] == self.item_embeddings.shape[0]:
            raise ValueError("Incorrect number of features in item_features")

        if not user_features.shape[1] == self.user_embeddings.shape[0]:
            raise ValueError("Incorrect number of features in user_features")

        if num_threads < 1:
            raise ValueError("Number of threads must be 1 or larger.")

        for _ in self._progress(epochs, verbose=verbose):
            self._run_epoch_by_row(
                item_features = item_features,
                user_features = user_features,
                user_id = user_id,
                interactions = interactions,
                sample_weight = sample_weight_data,
                num_threads = num_threads,
                loss = self.loss,
            )

            self._check_finite()

        return self

    def _run_epoch_by_row(
        self,
        item_features,
        user_features,
        user_id,
        interactions,
        sample_weight,
        num_threads,
        loss,
    ):
        """
        Run an individual epoch for a specific row.
        """

        # user_id_matrix = interactions.row[np.where(interactions.row == user_id)[0]]

        if loss in ("warp", "bpr", "warp-kos"):
            # The CSR conversion needs to happen before shuffle indices are created.
            # Calling .tocsr may result in a change in the data arrays of the COO matrix,
            positives_lookup = CSRMatrix(
                self._get_positives_lookup_matrix(interactions)
            )

        # Create shuffle indexes.
        shuffle_indices = np.array(np.where(interactions.row == user_id)[0], dtype=np.int32)
        self.random_state.shuffle(shuffle_indices)

        lightfm_data = self._get_lightfm_data()

        # Call the estimation routines.
        # if loss == "warp":
        fit_warp(
            CSRMatrix(item_features),
            CSRMatrix(user_features),
            positives_lookup,
            interactions.row,
            interactions.col,
            interactions.data,
            sample_weight,
            shuffle_indices,
            lightfm_data,
            self.learning_rate,
            self.item_alpha,
            self.user_alpha,
            num_threads,
            self.random_state,
        )
        # elif loss == "bpr":
        #     fit_bpr(
        #         CSRMatrix(item_features),
        #         CSRMatrix(user_features),
        #         positives_lookup,
        #         interactions.row,
        #         interactions.col,
        #         interactions.data,
        #         sample_weight,
        #         shuffle_indices,
        #         lightfm_data,
        #         self.learning_rate,
        #         self.item_alpha,
        #         self.user_alpha,
        #         num_threads,
        #         self.random_state,
        #     )
        # elif loss == "warp-kos":
        #     fit_warp_kos(
        #         CSRMatrix(item_features),
        #         CSRMatrix(user_features),
        #         positives_lookup,
        #         interactions.row,
        #         shuffle_indices,
        #         lightfm_data,
        #         self.learning_rate,
        #         self.item_alpha,
        #         self.user_alpha,
        #         self.k,
        #         self.n,
        #         num_threads,
        #         self.random_state,
        #     )
        # else:
        #     fit_logistic(
        #         CSRMatrix(item_features),
        #         CSRMatrix(user_features),
        #         interactions.row,
        #         interactions.col,
        #         interactions.data,
        #         sample_weight,
        #         shuffle_indices,
        #         lightfm_data,
        #         self.learning_rate,
        #         self.item_alpha,
        #         self.user_alpha,
        #         num_threads,
        #     )

    def fit(
        self,
        interactions,
        user_features=None,
        item_features=None,
        sample_weight=None,
        epochs=1,
        num_threads=1,
        verbose=False,
    ):
        """
        Fit the model.
        For details on how to use feature matrices, see the documentation
        on the :class:`lightfm.LightFM` class.
        Arguments
        ---------
        interactions: np.float32 coo_matrix of shape [n_users, n_items]
             the matrix containing
             user-item interactions. Will be converted to
             numpy.float32 dtype if it is not of that type.
        user_features: np.float32 csr_matrix of shape [n_users, n_user_features], optional
             Each row contains that user's weights over features.
        item_features: np.float32 csr_matrix of shape [n_items, n_item_features], optional
             Each row contains that item's weights over features.
        sample_weight: np.float32 coo_matrix of shape [n_users, n_items], optional
             matrix with entries expressing weights of individual
             interactions from the interactions matrix.
             Its row and col arrays must be the same as
             those of the interactions matrix. For memory
             efficiency its possible to use the same arrays
             for both weights and interaction matrices.
             Defaults to weight 1.0 for all interactions.
             Not implemented for the k-OS loss.
        epochs: int, optional
             number of epochs to run
        num_threads: int, optional
             Number of parallel computation threads to use. Should
             not be higher than the number of physical cores.
        verbose: bool, optional
             whether to print progress messages.
             If `tqdm` is installed, a progress bar will be displayed instead.
        Returns
        -------
        LightFM instance
            the fitted model
        """

        # Discard old results, if any
        self._reset_state()

        return self.fit_partial(
            interactions,
            user_features=user_features,
            item_features=item_features,
            sample_weight=sample_weight,
            epochs=epochs,
            num_threads=num_threads,
            verbose=verbose,
        )

    def fit_partial(
        self,
        interactions,
        user_features=None,
        item_features=None,
        sample_weight=None,
        epochs=1,
        num_threads=1,
        verbose=False,
    ):
        """
        Fit the model.
        Fit the model. Unlike fit, repeated calls to this method will
        cause training to resume from the current model state.
        For details on how to use feature matrices, see the documentation
        on the :class:`lightfm.LightFM` class.
        Arguments
        ---------
        interactions: np.float32 coo_matrix of shape [n_users, n_items]
             the matrix containing
             user-item interactions. Will be converted to
             numpy.float32 dtype if it is not of that type.
        user_features: np.float32 csr_matrix of shape [n_users, n_user_features], optional
             Each row contains that user's weights over features.
        item_features: np.float32 csr_matrix of shape [n_items, n_item_features], optional
             Each row contains that item's weights over features.
        sample_weight: np.float32 coo_matrix of shape [n_users, n_items], optional
             matrix with entries expressing weights of individual
             interactions from the interactions matrix.
             Its row and col arrays must be the same as
             those of the interactions matrix. For memory
             efficiency its possible to use the same arrays
             for both weights and interaction matrices.
             Defaults to weight 1.0 for all interactions.
             Not implemented for the k-OS loss.
        epochs: int, optional
             number of epochs to run
        num_threads: int, optional
             Number of parallel computation threads to use. Should
             not be higher than the number of physical cores.
        verbose: bool, optional
             whether to print progress messages.
             If `tqdm` is installed, a progress bar will be displayed instead.
        Returns
        -------
        LightFM instance
            the fitted model
        """

        # We need this in the COO format.
        # If that's already true, this is a no-op.
        interactions = interactions.tocoo()

        if interactions.dtype != CYTHON_DTYPE:
            interactions.data = interactions.data.astype(CYTHON_DTYPE)

        sample_weight_data = self._process_sample_weight(interactions, sample_weight)

        n_users, n_items = interactions.shape
        (user_features, item_features) = self._construct_feature_matrices(
            n_users, n_items, user_features, item_features
        )

        for input_data in (
            user_features.data,
            item_features.data,
            interactions.data,
            sample_weight_data,
        ):
            self._check_input_finite(input_data)
        if self.item_embeddings is None:
            # Initialise latent factors only if this is the first call
            # to fit_partial.
            self._initialize(
                self.no_components, item_features.shape[1], user_features.shape[1]
            )

        # Check that the dimensionality of the feature matrices has
        # not changed between runs.
        if not item_features.shape[1] == self.item_embeddings.shape[0]:
            raise ValueError("Incorrect number of features in item_features")

        if not user_features.shape[1] == self.user_embeddings.shape[0]:
            raise ValueError("Incorrect number of features in user_features")

        if num_threads < 1:
            raise ValueError("Number of threads must be 1 or larger.")

        for _ in self._progress(epochs, verbose=verbose):
            self._run_epoch(
                item_features,
                user_features,
                interactions,
                sample_weight_data,
                num_threads,
                self.loss,
            )

            self._check_finite()

        return self

    def _run_epoch(
        self,
        item_features,
        user_features,
        interactions,
        sample_weight,
        num_threads,
        loss,
    ):
        """
        Run an individual epoch.
        """

        if loss in ("warp", "bpr", "warp-kos"):
            # The CSR conversion needs to happen before shuffle indices are created.
            # Calling .tocsr may result in a change in the data arrays of the COO matrix,
            positives_lookup = CSRMatrix(
                self._get_positives_lookup_matrix(interactions)
            )

        # Create shuffle indexes.
        shuffle_indices = np.arange(len(interactions.data), dtype=np.int32)
        self.random_state.shuffle(shuffle_indices)

        lightfm_data = self._get_lightfm_data()

        # Call the estimation routines.
        if loss == "warp":
            fit_warp(
                CSRMatrix(item_features),
                CSRMatrix(user_features),
                positives_lookup,
                interactions.row,
                interactions.col,
                interactions.data,
                sample_weight,
                shuffle_indices,
                lightfm_data,
                self.learning_rate,
                self.item_alpha,
                self.user_alpha,
                num_threads,
                self.random_state,
            )
        elif loss == "bpr":
            fit_bpr(
                CSRMatrix(item_features),
                CSRMatrix(user_features),
                positives_lookup,
                interactions.row,
                interactions.col,
                interactions.data,
                sample_weight,
                shuffle_indices,
                lightfm_data,
                self.learning_rate,
                self.item_alpha,
                self.user_alpha,
                num_threads,
                self.random_state,
            )
        elif loss == "warp-kos":
            fit_warp_kos(
                CSRMatrix(item_features),
                CSRMatrix(user_features),
                positives_lookup,
                interactions.row,
                shuffle_indices,
                lightfm_data,
                self.learning_rate,
                self.item_alpha,
                self.user_alpha,
                self.k,
                self.n,
                num_threads,
                self.random_state,
            )
        else:
            fit_logistic(
                CSRMatrix(item_features),
                CSRMatrix(user_features),
                interactions.row,
                interactions.col,
                interactions.data,
                sample_weight,
                shuffle_indices,
                lightfm_data,
                self.learning_rate,
                self.item_alpha,
                self.user_alpha,
                num_threads,
            )

    def predict(
        self, user_ids, item_ids, item_features=None, user_features=None, num_threads=1
    ):
        """
        Compute the recommendation score for user-item pairs.
        For details on how to use feature matrices, see the documentation
        on the :class:`lightfm.LightFM` class.
        Arguments
        ---------
        user_ids: integer or np.int32 array of shape [n_pairs,]
             single user id or an array containing the user ids for the
             user-item pairs for which a prediction is to be computed. Note
             that these are LightFM's internal id's, i.e. the index of the
             user in the interaction matrix used for fitting the model.
        item_ids: np.int32 array of shape [n_pairs,]
             an array containing the item ids for the user-item pairs for which
             a prediction is to be computed. Note that these are LightFM's
             internal id's, i.e. the index of the item in the interaction
             matrix used for fitting the model.
        user_features: np.float32 csr_matrix of shape [n_users, n_user_features], optional
             Each row contains that user's weights over features.
        item_features: np.float32 csr_matrix of shape [n_items, n_item_features], optional
             Each row contains that item's weights over features.
        num_threads: int, optional
             Number of parallel computation threads to use. Should
             not be higher than the number of physical cores.
        Returns
        -------
        np.float32 array of shape [n_pairs,]
            Numpy array containing the recommendation scores for pairs defined
            by the inputs.
        """

        self._check_initialized()

        if not isinstance(user_ids, np.ndarray):
            user_ids = np.repeat(np.int32(user_ids), len(item_ids))

        if isinstance(item_ids, (list, tuple)):
            item_ids = np.array(item_ids, dtype=np.int32)

        assert len(user_ids) == len(item_ids)

        if user_ids.dtype != np.int32:
            user_ids = user_ids.astype(np.int32)
        if item_ids.dtype != np.int32:
            item_ids = item_ids.astype(np.int32)

        if num_threads < 1:
            raise ValueError("Number of threads must be 1 or larger.")

        if user_ids.min() < 0 or item_ids.min() < 0:
            raise ValueError(
                "User or item ids cannot be negative. "
                "Check your inputs for negative numbers "
                "or very large numbers that can overflow."
            )

        n_users = user_ids.max() + 1
        n_items = item_ids.max() + 1

        (user_features, item_features) = self._construct_feature_matrices(
            n_users, n_items, user_features, item_features
        )

        lightfm_data = self._get_lightfm_data()

        predictions = np.empty(len(user_ids), dtype=np.float64)

        predict_lightfm(
            CSRMatrix(item_features),
            CSRMatrix(user_features),
            user_ids,
            item_ids,
            predictions,
            lightfm_data,
            num_threads,
        )

        return predictions

    def _check_test_train_intersections(self, test_mat, train_mat):
        if train_mat is not None:
            n_intersections = test_mat.multiply(train_mat).nnz
            if n_intersections:
                raise ValueError(
                    "Test interactions matrix and train interactions "
                    "matrix share %d interactions. This will cause "
                    "incorrect evaluation, check your data split." % n_intersections
                )

    def predict_rank(
        self,
        test_interactions,
        train_interactions=None,
        item_features=None,
        user_features=None,
        num_threads=1,
        check_intersections=True,
    ):
        """
        Predict the rank of selected interactions. Computes recommendation
        rankings across all items for every user in interactions and calculates
        the rank of all non-zero entries in the recommendation ranking, with 0
        meaning the top of the list (most recommended) and n_items - 1 being
        the end of the list (least recommended).
        Performs best when only a handful of interactions need to be evaluated
        per user. If you need to compute predictions for many items for every
        user, use the predict method instead.
        For details on how to use feature matrices, see the documentation
        on the :class:`lightfm.LightFM` class.
        Arguments
        ---------
        test_interactions: np.float32 csr_matrix of shape [n_users, n_items]
             Non-zero entries denote the user-item pairs
             whose rank will be computed.
        train_interactions: np.float32 csr_matrix of shape [n_users, n_items], optional
             Non-zero entries denote the user-item pairs which will be excluded
             from rank computation. Use to exclude training set interactions
             from being scored and ranked for evaluation.
        user_features: np.float32 csr_matrix of shape [n_users, n_user_features], optional
             Each row contains that user's weights over features.
        item_features: np.float32 csr_matrix of shape [n_items, n_item_features], optional
             Each row contains that item's weights over features.
        num_threads: int, optional
             Number of parallel computation threads to use.
             Should not be higher than the number of physical cores.
        check_intersections: bool, optional, True by default,
            Only relevant when train_interactions are supplied.
            A flag that signals whether the test and train matrices should be checked
            for intersections to prevent optimistic ranks / wrong evaluation / bad data split.
        Returns
        -------
        np.float32 csr_matrix of shape [n_users, n_items]
            the [i, j]-th entry of the matrix will contain the rank of the j-th
            item in the sorted recommendations list for the i-th user.
            The degree of sparsity of this matrix will be equal to that of the
            input interactions matrix.
        """

        self._check_initialized()

        if num_threads < 1:
            raise ValueError("Number of threads must be 1 or larger.")

        if check_intersections:
            self._check_test_train_intersections(test_interactions, train_interactions)

        n_users, n_items = test_interactions.shape

        (user_features, item_features) = self._construct_feature_matrices(
            n_users, n_items, user_features, item_features
        )

        if not item_features.shape[1] == self.item_embeddings.shape[0]:
            raise ValueError("Incorrect number of features in item_features")

        if not user_features.shape[1] == self.user_embeddings.shape[0]:
            raise ValueError("Incorrect number of features in user_features")

        test_interactions = test_interactions.tocsr()
        test_interactions = self._to_cython_dtype(test_interactions)

        if train_interactions is None:
            train_interactions = sp.csr_matrix((n_users, n_items), dtype=CYTHON_DTYPE)
        else:
            train_interactions = train_interactions.tocsr()
            train_interactions = self._to_cython_dtype(train_interactions)

        ranks = sp.csr_matrix(
            (
                np.zeros_like(test_interactions.data),
                test_interactions.indices,
                test_interactions.indptr,
            ),
            shape=test_interactions.shape,
        )

        lightfm_data = self._get_lightfm_data()

        predict_ranks(
            CSRMatrix(item_features),
            CSRMatrix(user_features),
            CSRMatrix(test_interactions),
            CSRMatrix(train_interactions),
            ranks.data,
            lightfm_data,
            num_threads,
        )

        return ranks

    def get_item_representations(self, features=None):
        """
        Get the latent representations for items given model and features.
        Arguments
        ---------
        features: np.float32 csr_matrix of shape [n_items, n_item_features], optional
             Each row contains that item's weights over features.
             An identity matrix will be used if not supplied.
        Returns
        -------
        (item_biases, item_embeddings):
                (np.float32 array of shape n_items,
                 np.float32 array of shape [n_items, num_components]
            Biases and latent representations for items.
        """

        self._check_initialized()

        if features is None:
            return self.item_biases, self.item_embeddings

        features = sp.csr_matrix(features, dtype=CYTHON_DTYPE)

        return features * self.item_biases, features * self.item_embeddings

    def get_user_representations(self, features=None):
        """
        Get the latent representations for users given model and features.
        Arguments
        ---------
        features: np.float32 csr_matrix of shape [n_users, n_user_features], optional
             Each row contains that user's weights over features.
             An identity matrix will be used if not supplied.
        Returns
        -------
        (user_biases, user_embeddings):
                (np.float32 array of shape n_users
                 np.float32 array of shape [n_users, num_components]
            Biases and latent representations for users.
        """

        self._check_initialized()

        if features is None:
            return self.user_biases, self.user_embeddings

        features = sp.csr_matrix(features, dtype=CYTHON_DTYPE)

        return features * self.user_biases, features * self.user_embeddings

    def get_params(self, deep=True):
        """
        Get parameters for this estimator.
        Arguments
        ---------
        deep: boolean, optional
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.
        Returns
        -------
        params : mapping of string to any
            Parameter names mapped to their values.
        """

        params = {
            "loss": self.loss,
            "learning_schedule": self.learning_schedule,
            "no_components": self.no_components,
            "learning_rate": self.learning_rate,
            "k": self.k,
            "n": self.n,
            "rho": self.rho,
            "epsilon": self.epsilon,
            "max_sampled": self.max_sampled,
            "item_alpha": self.item_alpha,
            "user_alpha": self.user_alpha,
            "random_state": self.random_state,
        }

        return params

    def set_params(self, **params):
        """
        Set the parameters of this estimator.
        Returns
        -------
        self
        """

        valid_params = self.get_params()

        for key, value in params.items():
            if key not in valid_params:
                raise ValueError(
                    "Invalid parameter %s for estimator %s. "
                    "Check the list of available parameters "
                    "with `estimator.get_params().keys()`."
                    % (key, self.__class__.__name__)
                )

            setattr(self, key, value)

        return self