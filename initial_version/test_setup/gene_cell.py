import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import scipy.sparse as sp
from scipy.sparse import csr_matrix
from sklearn.preprocessing import LabelEncoder,normalize
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import os


adata = sc.read_h5ad('pbmc68k(2).h5ad')

X = adata.X

print(X)