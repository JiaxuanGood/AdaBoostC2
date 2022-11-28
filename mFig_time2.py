from turtle import color
import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
import numpy as np

numdata = 26+1
matplotlib.rc("font", family='MicroSoft YaHei')

time_AdaMH = [6.533072186,8.737853312,7.555697989,6.002291775,27.33282599,112.9283999,1.403811145,246.3553279,0.371865726,2.965137625,17.70979757,1.007323551,85.08870549,3195.281526,34.88612998,609.2091822,189.8067103,322.7382751,3226.051377,156.8694812,11.13744426,0.535152292,452.8576006,22645.17006,2485.482323,2115.836076]
time_BOOMER = [2.148031425,3.318823314,3.994099736,3.827477288,48.95097492,2057.909498,0.504691185,61.48972082,0.818095636,1.070405579,4.545908594,1.219727325,99.67668943,89.08435559,42.85874903,5.461809945,264.7695985,21.55780189,18531.82455,12032.03093,321.3156283,0.397846699,127.2978245,11.86951091,103725.1906,30804.68461]
time_AdaC2 = [4.59073782,3.271660304,3.891207957,3.281249285,4.565643907,37.68655851,2.337214661,369.964102,0.357384706,1.168773127,37.06930182,1.505942822,33.88038938,952.4859453,60.30780635,103.9795242,27.19213138,578.6340583,6038.985984,311.3940475,20.96884151,0.512158203,74.91944122,4232.806614,5270.101159,3707.277754]
time_AdaMH.append(np.average(time_AdaMH))
time_BOOMER.append(np.average(time_BOOMER))
time_AdaC2.append(np.average(time_AdaC2))
time1 = np.log10(np.array(time_AdaMH)+10)
time2 = np.log10(np.array(time_BOOMER)+10)
timem = np.log10(np.array(time_AdaC2)+10)

plt.figure(figsize=(12, 2.8))
width = 0.16 # width of a histogram
gap = 0.04
x = np.linspace(1,numdata,numdata)
x0 = x - width - gap # 第一组数据的中线位置
x1 = x
x2 = x + width + gap

# y = np.linspace(0,1.2,7)
plt.bar(x0, time1, width=width, label="AdaBoost.MH", color='#BB9727', alpha=0.9)#c97937,BB9727,54B345
plt.bar(x1, time2, width=width, label="BOOMER", color='#3b6291', alpha=0.9)
plt.bar(x2, timem, width=width, label="AdaBoost.C2", color='#943c39', alpha=0.9)
datas = []
for i in range(numdata):
    datas.append('#'+str(i+1))
datas[-1] = 'Avg'

print(len(x),len(x0),len(datas))

plt.xticks( ticks=x, labels=datas)
# plt.xlabel('Missing rate')
# plt.yticks( ticks=y, labels=y_label)
plt.legend()
plt.savefig('box/runtime2', bbox_inches='tight', dpi=1000)
# plt.show()
