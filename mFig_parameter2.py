from turtle import color
import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rc("font", family='MicroSoft YaHei')

X_label = ["0.001", "0.005", "0.01", "0.05", "0.1"]
# delta = [0.001, 0.005, 0.01, 0.05, 0.1]
tick_X = np.arange(5)

hamming = [0.115212381,0.109786821,0.107387109,0.109856551,0.113086143]
coverage = [0.266168113,0.263027721,0.260476566,0.262396283,0.260051282]
ranking = [0.158981807,0.156560679,0.152916133,0.154202279,0.152572202]
avgpre = [0.68801843,0.69266336,0.696230846,0.693542667,0.695777092]
num_rounds = [3.373803795,2.5934872,2.284946591,1.650533351,1.373628131]
max_rounds = [7,5,5,4,4]

performances = [hamming, coverage, ranking, avgpre, num_rounds, max_rounds]

y1 = [0.104,0.110,0.116]
y2 = [0.26,0.264,0.268]
y3 = [0.15,0.155,0.16]
y4 = [0.68,0.69,0.7]
y5 = [0,2,4]
y6 = [3,5,7]
y = [y1, y2, y3, y4, y5, y6]

titles = ["Hamming Loss"+r'$\downarrow$', "Coverage"+r'$\downarrow$', "Ranking Loss"+r'$\downarrow$', "Average Precision"+r'$\uparrow$',
    "T in average", "T in 99% quantile"]
colors = ['#c82423','#c82423','#c82423','#c82423','k','k']
plt.subplots(figsize=(8,3))
for test in range(6):
    ax = plt.subplot(2, 3, test+1)
    ax.plot(X_label,performances[test], color=colors[test])
    if(test<3):
        ax.set_xticklabels([])
    ax.set_title(titles[test], fontsize=10)
    ax.set_yticks(y[test])
    ax.set_yticklabels(y[test])
plt.tight_layout(w_pad=0.5,h_pad=0.3)
# plt.show()
plt.savefig('box/parameter3', bbox_inches='tight', dpi=1000)
