import subprocess
import numpy as np
import matplotlib.pyplot as plt
threadcnts = np.linspace(1,32,32) * 32

dataset = []
for tc in threadcnts:
    p = subprocess.run([f"./build/tCommTest8", str(tc)], capture_output=True)
    print(p.stderr.decode('utf-8'))
    dataset.append(float(p.stdout.decode('utf-8')[:-1]))

print(dataset)
dataset = 1e6 / np.array(dataset)
plt.plot(threadcnts, dataset, label=f"tcomm single thread block")

plt.xlabel("Thread Count")
plt.ylabel("Rate (KHz)")
plt.savefig('tcomm_mini.png')
