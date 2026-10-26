import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns

plt.rcParams.update({"font.size": 16})
plt.rcParams.update({"font.family": "serif"})
plt.rcParams.update({"mathtext.fontset":"cm"})

MODEL = ["simple_cgp_sam", "simple_cgp_prob"]

D_MIN = 5
D_MAX = 30

Y_MIN = 0
Y_MAX = 200000

STEP = 5
X_TICKS = [n for n in range(D_MIN, D_MAX + 1, STEP)]
X_TICKS += [30]

Y_LIM = True
X_LIM = True

data = pd.read_csv('../csv/max_plus_mul_log_cgp_5_30.csv')
data = data.drop(data[data["d"]<D_MIN].index) # remove noise from too small D

## -- linear fitting in the log-log scale with numpy
x = list(range(D_MIN,D_MAX+1,1))
y = {m:[data[(data["d"]==d) & (data["model"]==m)]["num_iters"].mean() for d in x] # extract the mean values
     for m in MODEL}

coeff = {m:np.polyfit(np.log(x),np.log(y[m]),deg=1) for m in MODEL} # linear fitting in the log-scale
yref = {m:[np.exp(coeff[m][1])*np.power(d,coeff[m][0]) for d in x] # generate the fitting curves
        for m in MODEL}
ylabel= {m:"${:.2f} \\times D^{{{:.2f}}}$".format(np.exp(coeff[m][1]),coeff[m][0]) for m in MODEL} # labels for the curves

# create the dataframe for the fitting curve
dfref = pd.DataFrame(data={"x":x, **yref})

sns.set(font_scale=1.1)

#plt.title('Strict selection', fontsize=16)
plt.xlabel(xlabel='D', fontsize=14)
plt.ylabel(ylabel='Average Number of Iterations', fontsize=14)

# create the dataframe for the fitting curve
dfref = pd.DataFrame(data={"x":x, **yref})

# start plotting
fig, ax = plt.subplots(figsize=(8, 6))
p = sns.lineplot(
    data=data,
    x="d", y="num_iters",
    hue="model", style="model",
    markers=True, dashes=False, linestyle="", #errorbar=None,
    ax=ax
)

for m in MODEL:
    sns.lineplot(data=dfref,
                 x="x", y=m,
                 label=ylabel[m],
                 ax=ax)


p.set(xlabel="$D$", ylabel="Average Number of Iterations", yscale="linear")
p.set_xticks(X_TICKS)
p.set_xticklabels(str(d) for d in X_TICKS)

#yscale="log"
p.set(yscale="linear")

#p.set_xlabel(p.get_xlabel(), fontdict={'weight': 'bold'})
#p.set_ylabel(p.get_ylabel(), fontdict={'weight': 'bold'})
p.set_title(p.get_title(), fontdict={'weight': 'bold'})

if Y_LIM:
    plt.ylim(Y_MIN, Y_MAX)

if X_LIM:
    plt.xlim(D_MIN, D_MAX)

fig.tight_layout()
plt.savefig("max_plus_mul_log_cgp_sam_prob_5_30_logloglinfit.png", dpi=150)
plt.show()