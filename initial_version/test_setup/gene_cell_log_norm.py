import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from sklearn.preprocessing import LabelEncoder
from helper import train_logistic_regression, evaluate_model

adata = sc.read_h5ad('pbmc68k(2).h5ad')

sc.pp.normalize_total(adata, target_sum=1e4)  
sc.pp.log1p(adata)  
X = adata.X 

cell_type_series = adata.obs['cell_type']
le = LabelEncoder()
y = le.fit_transform(cell_type_series)

model = train_logistic_regression(X, y)
evaluate_model(model, X, y, le,"test")