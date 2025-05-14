import scanpy as sc

file_path = '/data1/data/corpus/panglao_human.h5ad'
adata = sc.read_h5ad(file_path)

gene_expression_counts = (adata.X > 0).sum(axis=0)

if hasattr(gene_expression_counts, 'A1'):
    gene_expression_counts = gene_expression_counts.A1

gene_names = adata.var_names
gene_counts = dict(zip(gene_names, gene_expression_counts))

top_genes = sorted(gene_counts.items(), key=lambda x: x[1], reverse=True)[:10]
for gene, count in top_genes:
    print(f"{gene}: {count} cells")
