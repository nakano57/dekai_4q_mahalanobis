import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import DistanceMetric
from mpl_toolkits.mplot3d import Axes3D
from scipy import stats
from scipy.stats import norm
import random
import time


def hanbetu(x):
    return 'グループ' + ('1' if x > 0 else '2') + 'です'


random.seed(time.time())
# データの読み込み
df = pd.read_excel('DA-E-191105.xlsx', sheet_name=1)
df = df.drop(df.index[[0, 1]])
G1 = df[['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4']]
G1.columns = ['Attribute1', 'Attribute2', 'Attribute3']
G2 = df[['Unnamed: 7', 'Unnamed: 8', 'Unnamed: 9']]
G2.columns = ['Attribute1', 'Attribute2', 'Attribute3']
G1_n = G1.values.astype(float).T
G2_n = G2.values.astype(float).T
G1_ave = np.array([[G1.iloc[:, [0]].mean()[0], G1.iloc[:, [1]].mean()[0],
                    G1.iloc[:, [2]].mean()[0]]])
G2_ave = np.array([[G2.iloc[:, [0]].mean()[0], G2.iloc[:, [1]].mean()[0],
                    G2.iloc[:, [2]].mean()[0]]])
Data = pd.concat([G1, G2]).values.astype(float).T

G = (((G1_n.size / len(G1_n) - 1) * np.cov(G1_n)) +
     ((G2_n.size / len(G2_n) - 1) * np.cov(G2_n))) / ((G1_n.size / len(G1_n) - 1) + (G2_n.size / len(G2_n) - 1))


p1 = np.array([[2.01235, 0.0, 1.45679]])
p2 = np.array([[1.98000, -1.01000, 0.99000]])

u_ave = (G1_ave+G2_ave)/2
_z = np.dot((G1_ave - G2_ave), np.linalg.inv(G))

coef = _z[0]*2
intercept = np.dot(_z, u_ave.T)[0] * 2*-1
w1, w2, w3 = np.round(coef, 3)
w0 = np.round(intercept[0])
print('y= ({0})x1 + ({1})x2 + ({2})x3 + ({3})'.format(w1, w2, w3, w0))

z = np.dot(_z, (p1 - u_ave).T) * 2
print('点Aは'+hanbetu(z[0][0]))
z = np.dot(_z, (p2 - u_ave).T)*2
print('点Bは'+hanbetu(z[0][0]))

print()
n = np.dot(_z, (G1_ave - G2_ave).T)
u = np.sqrt(n)/2
print('判別効率:{}'.format(n[0][0]))
print('誤判別率:{}'.format(norm.sf(x=u, loc=0, scale=1.00)[0][0]))


def r():
    a = '#'
    for _ in range(3):
        n = random.randrange(255)
        a += ('0' + str(hex(n).lstrip('0x'))
              ) if n < 16 else str(hex(n).lstrip('0x'))
    a = a if len(a) == 7 else r()
    return a


# Plot
x = np.arange(np.min(Data[0])*0.9, np.max(Data[0])*1.1, 0.1)
y = np.arange(np.min(Data[1])*0.9, np.max(Data[1])*1.1, 0.1)
X, Y = np.meshgrid(x, y)
Z = np.array(list(map(lambda x, y: -1 * (w1 * x + w2 * y + w0) / w3, X, Y)))

fig = plt.figure(facecolor=r(), edgecolor=r())
ax = fig.add_subplot(111, projection="3d")
ax.set_facecolor(r())
ax.scatter(G1_n[0], G1_n[1], G1_n[2], label='Group1', c=r())
ax.scatter(G2_n[0], G2_n[1], G2_n[2], label='Group2', c=r())
ax.scatter(p1[0][0], p1[0][1], p1[0][2],  c=r(), marker='^')
ax.scatter(p2[0][0], p2[0][1], p2[0][2], c=r(), marker='^')
for i in range(15):
    ax.text(G1_n[0][i], G1_n[1][i], G1_n[2][i], str(i+1), zorder=1, c=r())
    ax.text(G2_n[0][i], G2_n[1][i], G2_n[2][i], str(i+16), zorder=1, c=r())
ax.text(p1[0][0], p1[0][1], p1[0][2], 'A', zorder=1, color=r())
ax.text(p2[0][0], p2[0][1], p2[0][2], 'B', zorder=1, color=r())
ax.plot_surface(X, Y, Z, alpha=0.5, color=r())
ax.legend()

plt.figure(facecolor=r(), edgecolor=r())
plt.rcParams["axes.facecolor"] = r()
plt.cla()
x_norm = np.arange(-4, 4, 0.1)
x_fill = np.arange(u, 4, 0.1)
y_norm = [stats.norm.pdf(x=m, loc=0, scale=1) for m in x_norm]
y_fill1 = [stats.norm.pdf(x=m, loc=0, scale=1) for m in x_fill]
y_fill2 = np.zeros(x_fill.size)
plt.scatter(u, stats.norm.pdf(x=u, loc=0, scale=1), c=r())
plt.plot(x_norm, y_norm, c=r())
plt.fill_between(x_fill, y_fill1, y_fill2, alpha=0.5, color=r())
plt.text(0.5, 0.4, 'UNCHI BURI', ha='center',
         va='center', transform=ax.transAxes, fontsize=32, weight='bold', c=r())
plt.show()
