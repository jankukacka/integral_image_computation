import matplotlib.pyplot as plt
import numpy as np

def main():
    sizes = ['128x128','256x256','512x512','1024x1024','2048x2048','4096x4096','8192x8192']
    t_cpu = [0,0,1,5.6,26,112,428]
    t_omp = [0,0,0.12,3.48,17.16,51.56,260]
    t_gpu = [0,0,0,2,6,21,77]
    t_gpu2 = [0,0,0,1,3,9,34]
    t_copy = [0,0,0,2,12,58,240]
    data = np.array([t_cpu,t_omp,t_gpu,t_gpu2])
    labels = ['CPU', 'CPUx8', 'GPUv1', 'GPUv2']

    fig,ax = plt.subplots()
    for row, label in zip(data, labels):
        ax.plot(row, label=label)
    ax.plot(t_copy, 'k--', label='GPU memcpy')
    ax.set_xticks(range(len(sizes)))
    ax.set_xticklabels(sizes)
    ax.set_xlim([3,6])
    ax.set_xlabel('Input size')
    ax.set_ylabel('Time (ms)')
    ax.legend()
    plt.show()

if __name__ == '__main__':
    main()
