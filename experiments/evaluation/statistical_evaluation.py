import pandas as pd

data = pd.read_csv('../plot/csv/jmlr26/tgp_sr_coeff.csv')
print(data.groupby(["d","model"])["mse"].describe().apply(lambda s: s.apply(lambda s: s)).to_csv(sep = "&", lineterminator="\\\\ \n"))
#print(data.groupby(["d","model"])["mse"].describe().apply(lambda s: s.apply(lambda s: round(s))).to_csv(sep = "&", lineterminator="\\\\ \n"))
