import subprocess

commbytes = [8,16,32,64,128,256]
#threadcnts = [64,256,512,1024,2048,4096,8192,16384,32768]
#threadcnts = [64,256,512,1024,2048,4096]
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
datasets = [[1157.35, 1310.78, 1435.7, 1422.58, 1500.1, 1437.21, 1487.86, 1922.33, 1925.95, 1923.64, 1926.9, 1926.23, 1929.19, 1928.4, 1931.58, 1930.44, 1934.82], [1204.6, 2202.46, 3147.7, 3516.82, 3147.39, 3498.02, 3489.15, 3494.6, 3482.78, 3488.33, 3484.4, 3489.39, 3486.67, 3491.27, 3481.27, 3487.91, 3472.49], [2238.48, 5736.74, 9515.68, 9112.21, 9195.03, 9142.54, 9158.55, 9142.2, 9159.31, 9163.67, 9167.29, 9167.55, 9172.5, 9174.81, 9181.14, 9181.26, 9184.31], [4229.07, 13538.6, 22578.2, 23111.7, 23023.9, 23042.5, 23030.9, 23058.6, 23038.6, 23066.6, 23043.3, 23045.0, 23110.6, 23160.3, 23176.9, 23213.0, 23161.1], [10726.4, 39352.5, 62429.8, 63011.6, 63145.0, 63073.6, 63001.4, 63008.9, 63015.2, 63025.2, 63028.0, 63034.6, 63044.3, 63071.1, 63493.6, 63410.8, 63554.0], [20630.4, 77639.2, 124514.0, 124507.0, 124426.0, 123991.0, 124022.0, 124018.0, 124508.0, 124514.0, 124505.0, 124535.0, 124564.0, 124622.0, 125351.0, 125353.0, 125425.0]]
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
