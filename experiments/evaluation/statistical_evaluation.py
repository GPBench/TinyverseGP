import pandas as pd

data = pd.read_csv('../csv/max_plus_mul_tgp_cgp_1_10.csv')
print(data.groupby(["d","model"])["num_iters"].describe().apply(lambda s: s.apply(lambda s: round(s))).to_csv(sep = "&", lineterminator="\\\\ \n"))
