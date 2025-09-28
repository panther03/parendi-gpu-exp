import subprocess

commbytes = [8,16,32,64,128,256]
threadcnts = [64] + [x*256 for x in range(1,17)]

'''
datasets = []
for commbyte in commbytes: 
    dataset = []
    for threadcnt in threadcnts:
        p = subprocess.run([f"./build/tCommTest{commbyte}", str(threadcnt)], capture_output=True)
        print(p.stderr.decode('utf-8'))
        dataset.append(float(p.stdout.decode('utf-8')[:-1]))
    datasets.append(dataset)

print(datasets)
with open("datasets.txt", 'w') as f:
    f.write(str(datasets))
    f.write("\n")

exit(0)'''
datasets = [[1236.8, 1236.46, 1243.23, 1236.86, 1229.97, 1231.97, 1234.06, 1227.66, 1230.15, 1233.0, 1231.72, 1235.99, 1235.58, 1249.89, 1254.68, 1261.23, 1270.25], [1160.65, 1171.4, 1178.75, 1177.27, 1166.26, 1169.04, 1168.21, 1171.56, 1172.24, 1174.38, 1177.85, 1180.46, 1180.23, 1194.97, 1201.57, 1206.87, 1218.47], [1186.65, 1193.85, 1199.94, 1196.13, 1197.1, 1197.59, 1195.18, 1196.77, 1196.19, 1198.7, 1205.68, 1206.3, 1207.67, 1220.44, 1223.31, 1228.14, 1244.62], [1338.63, 1343.63, 1350.91, 1350.69, 1349.68, 1365.71, 1354.55, 1365.88, 1367.02, 1370.98, 1396.69, 1396.07, 1390.03, 1438.7, 1436.86, 1442.64, 1478.19], [1421.1, 1427.75, 1437.26, 1434.14, 1464.75, 1467.17, 1469.95, 1510.18, 1504.03, 1508.27, 1581.45, 1590.56, 1604.04, 1691.22, 1720.75, 1747.78, 1800.83], [1644.48, 1647.05, 1655.12, 1659.0, 1698.07, 1715.77, 1750.54, 1932.83, 1950.1, 1973.33, 2112.54, 2142.09, 2175.84, 2396.35, 2405.87, 2418.71, 2534.32]]
import numpy as np
import matplotlib.pyplot as plt

commbytes = [int(np.log2(cb)) for cb in commbytes]
#threadcnts = [int(np.log2(tc)) for tc in threadcnts]
# Sample data (replace with your actual data)
X, Y = np.meshgrid(commbytes, threadcnts)
##[[2506.48, 2543.99, 3414.61, 3405.31, 3405.44, 3410.61], [3513.9, 3843.68, 5748.55, 5735.1, 5737.58, 5757.92], [5606.36, 8206.71, 13095.1, 13100.1, 13090.9, 13122.5], [10170.4, 17069.2, 28280.6, 28306.4, 28199.7, 28354.4], [18516.1, 41499.2, 64019.7, 71852.0, 71913.9, 72299.6], [36199.1, 81729.3, 142512.0, 142346.0, 142370.0, 143218.0]]
Z = 1e6 / np.transpose(np.array(datasets))
print(Z[0][0])

# Create figure and axis
fig, ax = plt.subplots(figsize=(8,6))

# Create the heatmap with pcolormesh
heatmap = ax.pcolormesh(X, Y, Z, shading='nearest', cmap="Oranges")

xticks = commbytes  # Positions of ticks (same as your data points)
xtick_labels = [f"${{{1<<x}}}$" for x in xticks]  # Labels as 2^3, 2^4, etc.
ax.set_xticks(xticks)
ax.set_xticklabels(xtick_labels)

#yticks = threadcnts  # Positions of ticks (same as your data points)
#ytick_labels = [f"${{{1<<y}}}$" for y in yticks]  # Labels as 2^3, 2^4, etc.
#ax.set_yticks(yticks)
#ax.set_yticklabels(ytick_labels)

# Add colorbar
plt.colorbar(heatmap, label='\"Simulation\" rate (KHz)')

# Labels and title
ax.set_xlabel('# bytes')
ax.set_ylabel('# threads')
ax.set_title('1000000 iters tcomm+tsync')

plt.tight_layout()
plt.savefig('heattest.png')
exit(0)

datasets = [[(854.579, 0.0), (854.462, 0.0), (854.408, 0.0), (854.51, 0.0), (856.723, 0.0), (868.072, 0.0), (1088.23, 0.0)], [(6630.54, 0.0), (6629.63, 0.0), (6647.0, 0.0), (6646.53, 0.0), (6647.96, 0.0), (6818.07, 0.0), (7747.38, 0.0)], [(64764.1, 0.0), (64756.3, 0.0), (64749.9, 0.0), (64923.7, 0.0), (64884.3, 0.0), (66397.6, 0.0), (76644.4, 0.0)], [(647066.0, 0.0), (646920.0, 0.0), (646881.0, 0.0), (646694.0, 0.0), (647454.0, 0.0), (662883.0, 0.0), (780948.0, 0.0)]]
# datasets = [[(0.0, 1586.82), (0.0, 1593.41), (0.0, 1593.56), (0.0, 1593.93), (0.0, 1589.48), (0.0, 1602.5), (0.0, 2781.02)], [(0.0, 7344.51), (0.0, 7344.01), (0.0, 7366.91), (0.0, 7368.05), (0.0, 7372.2), (0.0, 14350.2), (0.0, 16193.9)], [(0.0, 65484.8), (0.0, 72324.6), (0.0, 65479.1), (0.0, 65621.5), (0.0, 65612.0), (0.0, 131953.0), (0.0, 148556.0)], [(0.0, 647591.0), (0.0, 1294660.0), (0.0, 1290550.0), (0.0, 1289770.0), (0.0, 1294100.0), (0.0, 1308140.0), (0.0, 1491620.0)]]
#datasets = [[15.7912, 19.2178, 19.1396, 32.1305, 30.0782, 33.9752, 37.1612], [48.8832, 60.6431, 60.5936, 63.871, 71.6325, 77.3887, 80.995], [89.2259, 94.3231, 94.2753, 78.3594, 81.0652, 85.3082, 80.2053], [98.0072, 62.118, 62.2408, 76.4811, 82.9937, 86.8695, 81.2124]]
# datasets = [[(102.512, 649.931), (129.907, 676.2), (129.897, 678.736), (222.039, 691.422), (323.001, 1074.11), (423.229, 1244.89), (522.947, 1406.76)], [(530.51, 1085.26), (832.687, 1373.29), (833.491, 1372.88), (1616.65, 2531.69), (2404.13, 3333.08), (3193.89, 4133.76), (3981.0, 4929.15)], [(4953.5, 5555.32), (7959.33, 8437.6), (7978.58, 8465.14), (15724.0, 20063.3), (23480.8, 28943.3), (31311.8, 36694.6), (39130.9, 48770.6)], [(49221.3, 50236.7), (79099.7, 127279.0), (79344.2, 127479.0), (157005.0, 205284.0), (234818.0, 282908.0), (312864.0, 360478.0), (390710.0, 481042.0)]]

# plot each dataset as a line graph

for i, dataset in enumerate(datasets):
    plt.plot(threadcnts, list(map(lambda x: dataset[0][0]/x[0], dataset)), label=f"FPT: {fpts[i]}")

plt.xlabel("Thread Count")
plt.ylabel("Rate")
plt.legend()
plt.savefig('nosync.png')
exit(0)

plt.close()
for i, dataset in enumerate(datasets):
    plt.plot(threadcnts, list(map(lambda x: dataset[0][0]/x[0], dataset)), label=f"FPT: {fpts[i]}")

plt.xlabel("Thread Count")
plt.ylabel("Rate")
plt.legend()
plt.savefig('nosync.png')
