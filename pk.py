import numpy as np
from collections import defaultdict
from sklearn.exceptions import NotFittedError
class PK:
    
    def __init__(self, n_bins_per_dim=2):
        self.n_bins_per_dim = n_bins_per_dim
        self.bins_ = None
        self.n_features_in_ = None
        self.feature_names_in_ = None



    def fit(self, X, feature_names=None):
        X = np.asarray(X, dtype=float)
        n_samples, n_features = X.shape

        all_nan_columns = np.all(np.isnan(X), axis=0)
        if np.any(all_nan_columns):
            raise ValueError(f"All columns are nan. Column number: {np.where(all_nan_columns)[0]}")

        bins = []
        for j in range(n_features):
            col = X[:, j]
            non_nan_values = col[~np.isnan(col)]
            non_nan_values.sort()

            if len(non_nan_values) >= self.n_bins_per_dim:
                q = np.linspace(0, 100, self.n_bins_per_dim)
                centers = np.percentile(non_nan_values, q)
            else:
                centers = np.pad(non_nan_values,
                            (0, self.n_bins_per_dim - len(non_nan_values)),
                            mode='edge')

            bins.append(centers)

        self.bins_ = bins
        self.n_features_in_ = n_features
        self.feature_names_in_ = feature_names if feature_names is not None else \
                            [f"feature_{j}" for j in range(n_features)]
        
        return self
    
    def transform(self, X):
       
        if self.bins_ is None:
            raise NotFittedError("Please fit first!")
            
        X = np.asarray(X, dtype=float)
        n_samples, n_features = X.shape
        
        n_bins = self.n_bins_per_dim
        
        cell_assignments = np.full((n_samples, n_features), -1)
        for j in range(n_features):
            col = X[:, j]
            not_nan = ~np.isnan(col)
            if np.any(not_nan):
                distances = np.abs(col[not_nan, None] - self.bins_[j])
                cell_assignments[not_nan, j] = np.argmin(distances, axis=1)
        
        feature_cell_to_samples = defaultdict(set)
        for j in range(n_features):
            for cell_id in range(n_bins):
                matches = np.where(cell_assignments[:, j] == cell_id)[0]
                feature_cell_to_samples[(j, cell_id)] = set(matches)
        
        onehot_matrix = np.zeros((n_samples, n_features * n_bins))
        for i in range(n_samples):
            for j in range(n_features):
                if cell_assignments[i, j] != -1:
                    cell_id = int(cell_assignments[i, j])
                    col_start = j * n_bins
                    onehot_matrix[i, col_start + cell_id] = 1.0

        self.global_onehot_mean_ = []
        for j in range(n_features):
            non_nan_values = X[:, j][~np.isnan(X[:, j])]
            assignments = np.digitize(non_nan_values, self.bins_[j]) - 1
            hist = np.bincount(assignments, minlength=self.n_bins_per_dim)
            self.global_onehot_mean_.append(hist / hist.sum())

        for i in range(n_samples):
            missing_features = [j for j in range(n_features) if cell_assignments[i, j] == -1]
            observed_features = [j for j in range(n_features) if cell_assignments[i, j] != -1]
            
            if not missing_features or not observed_features:
                continue
            
            candidate_samples = None
            for j in observed_features:
                cell_id = int(cell_assignments[i, j])
                others = feature_cell_to_samples[(j, cell_id)] - {i}
                if candidate_samples is None:
                    candidate_samples = others
                else:
                    candidate_samples &= others
            
            if not candidate_samples:
                candidate_samples = set()
                for j in observed_features:
                    cell_id = int(cell_assignments[i, j])
                    others = feature_cell_to_samples[(j, cell_id)] - {i}
                    candidate_samples |= others
            if not candidate_samples:
                for missing_feature in missing_features:
                    col_start = missing_feature * n_bins
                    col_end = col_start + n_bins
                    onehot_matrix[i, col_start:col_end] = self.global_onehot_mean_[missing_feature]
                continue


            for missing_feature in missing_features:
                valid_samples = [c for c in candidate_samples 
                            if cell_assignments[c, missing_feature] != -1]
                col_start = missing_feature * n_bins
                if valid_samples:
                    avg_onehot = onehot_matrix[valid_samples, col_start:col_start + n_bins].mean(axis=0)
                    onehot_matrix[i, col_start:col_start + n_bins] = avg_onehot
                else:
                    onehot_matrix[i, col_start:col_start + n_bins] = self.global_onehot_mean_[missing_feature]
                
        return onehot_matrix
    
    def fit_transform(self, X, feature_names=None):
        return self.fit(X, feature_names).transform(X)
    
