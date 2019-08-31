import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



def get_gaussion_mask(img_x, img_y, loc_x, loc_y, radio):
    y = np.arange(img_y).reshape((img_y, 1)).repeat(img_x, 1)
    x = np.arange(img_x).reshape((1, img_x)).repeat(img_y, 0)
    distance = np.sqrt(np.square(x - loc_x) + np.square(y - loc_y))
    mask = np.zeros_like(distance)
    x = distance[distance < radio]
    mask[distance < radio] = np.sum(((1/(radio**2))*(x**2), (-2/radio)*x, 1))
    mask[distance >= radio] = 0
    return mask


if __name__ == "__main__":
    fig = plt.figure()
    ax = Axes3D(fig)
    start_time = time.time()
    height = 2
    width = 2
    locx = 1
    locy = 1
    z = get_gaussion_mask(height, width, locx, locx, 1)
    print(z)
    print(((z<0.01)&(z>0)).sum())
    print(time.time()-start_time)
    y = np.arange(height).reshape((height, 1)).repeat(width, 1)
    x = np.arange(width).reshape((1, width)).repeat(height, 0)
    ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap='rainbow')
    # plt.savefig('fig.png', bbox_inches='tight')
    plt.show()

