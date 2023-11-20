import pandas as pd


df1 = pd.read_csv('BRCA.tsv', sep="\t", index_col=0)
df1 = df1.loc[:, df1.iloc[0] =='raw_counts']
df1 = df1.drop('gene').astype('int64')

df2 = pd.read_csv('GBMLGG.tsv', sep="\t", index_col=0)
df2 = df2.loc[:, df2.iloc[0] =='raw_count']
df2 = df2.drop('gene_id').astype('float64')

# Join 2 files
df = df1.join(df2, how='inner')

# Normalise w.r.t. sequencing depth
df = df / df.sum()

# optionally standardise as well
df = df.loc[df.sum(axis=1)>0, :]
df = df.div(df.max(axis=1), axis='rows')

# Export to 1 file
df.to_csv("gdac_X.tsv", sep='\t')

# Binary classification labels.
# 0: from df1
# 1: from df2
y = [ *[0]*df1.shape[1], *[1]*df2.shape[1] ]
df_y = pd.DataFrame(y, index=list(df))
df_y.to_csv("gdac_y.tsv", sep='\t')
