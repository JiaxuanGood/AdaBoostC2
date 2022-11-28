from turtle import color
import seaborn as sb
import numpy as np
import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
import pandas as pd

matplotlib.rc("font", family='MicroSoft YaHei')

def readMat(name,dim1,dim2):
    mat = np.zeros((dim1,dim2))
    i = 0
    filename = "result/" + name# + ".txt"
    for line in open(filename, 'r'):
        if(i==dim1):
            break
        lines = line.split('\t')
        for j in range(dim2):
            mat[i][j] = float(lines[j])
        i += 1
    return np.array(mat)

filenames = ["hamming", "coverage", "rankloss", "avgPre"]
algnames_new2 = ['Base1', 'Base2', 'AdaC2']
# for test in range(4):
#     fig1, ax1 = plt.subplots(figsize=(3,3))
#     data = readMat(filenames[test],26,3)
#     ax1.boxplot(data, labels=algnames_new2, patch_artist=True, 
#         boxprops={'edgecolor':'#2878b5','facecolor':'white','linewidth':1.1}, medianprops={'color':'#c82423','linewidth':1.2})
#     # plt.show()
#     plt.savefig('box/rotbox_'+filenames[test], bbox_inches='tight', dpi=1000)

xlabels = ["Hamming Loss"+r'$\downarrow$', "Coverage"+r'$\downarrow$', "Ranking Loss"+r'$\downarrow$', "Average Precision"+r'$\uparrow$']
yticks = np.arange(6)*0.2
plt.subplots(figsize=(6,2))
for test in range(4):
    ax = plt.subplot(1, 4, test+1)
    data = readMat(filenames[test],26,3)
    ax.boxplot(data, patch_artist=True, 
        boxprops={'edgecolor':'#2878b5','facecolor':'white','linewidth':1.1}, medianprops={'color':'#c82423','linewidth':1.2})
    ax.set_xticklabels(algnames_new2, rotation=20)
    # ax.set_xlabel(xlabels[test])
    ax.set_title(xlabels[test], fontsize=10)
    ax.set_yticks(yticks)
    if(test>0):
        ax.set_yticklabels(['','','','','',''])
plt.tight_layout(w_pad=0.5)
# plt.show()
plt.savefig('box/rotbox_all', bbox_inches='tight', dpi=1000)
