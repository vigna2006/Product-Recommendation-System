import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

class ProductRecommender:
    
    def __init__(self, ratings_file, products_file):
        self.ratings = pd.read_csv(ratings_file)
        self.products = pd.read_csv(products_file)

        self.X, self.customer_mapper, self.product_mapper, self.customer_inv_mapper, self.product_inv_mapper = self.create_sparse_matrix()
        

    def create_sparse_matrix(self):
        # number of unique customers and products
        num_customers = len(self.ratings['customer_id'].unique())
        num_products = len(self.ratings['product_id'].unique())

        # create mappings between customer/product IDs and integer indices
        customer_mapper = dict(zip(np.unique(self.ratings["customer_id"]), list(range(num_customers))))
        product_mapper = dict(zip(np.unique(self.ratings["product_id"]), list(range(num_products))))

        # create inverse mappings
        customer_inv_mapper = dict(zip(list(range(num_customers)), np.unique(self.ratings["customer_id"])))
        product_inv_mapper = dict(zip(list(range(num_products)), np.unique(self.ratings["product_id"])))

        # map customer and product IDs to integer indices
        customer_indices = [customer_mapper[i] for i in self.ratings['customer_id']]
        product_indices = [product_mapper[i] for i in self.ratings['product_id']]

        # create a sparse matrix of ratings
        X = csr_matrix((self.ratings["rating"], (product_indices, customer_indices)), shape=(num_products, num_customers))

        return X, customer_mapper, product_mapper, customer_inv_mapper, product_inv_mapper

    def find_similar_products(self, product_id, k, metric='cosine', show_distance=False):
        
        # find the index of the product in the sparse matrix
        product_ind = self.product_mapper[product_id]
        product_vec = self.X[product_ind]
        
        # add 1 to the number of neighbors to include the product itself
        k += 1
        
        # create and fit the nearest neighbors model
        kNN = NearestNeighbors(n_neighbors=k, algorithm="brute", metric=metric)
        kNN.fit(self.X)
        
        # reshape the product vector and find the nearest neighbors
        product_vec = product_vec.reshape(1,-1)
        neighbor_indices = kNN.kneighbors(product_vec, return_distance=show_distance)
        
        # map the indices of the neighbors to their product IDs
        neighbor_ids = []
        for i in range(0, k):
            n = neighbor_indices.item(i)
            neighbor_ids.append(self.product_inv_mapper[n])
        
        # remove the first item, which is the product itself
        neighbor_ids.pop(0)
        
        return neighbor_ids        
    
    def get_products_rated_by_user(self, user_id):
        user_ratings = self.ratings.loc[self.ratings['customer_id'] == user_id]
        return user_ratings['product_id']

