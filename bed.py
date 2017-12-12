import numpy as np
import matplotlib.pyplot as plt
import cv2

def plot_quiver(dx, dy):
    tmp = np.where(dx == dx)
    r_s, c_s = dx.shape
    X = tmp[1].reshape(r_s, c_s)
    Y = tmp[0].reshape(r_s, c_s)
    U, V = dx, dy
    plt.quiver(X, Y, U, V, angles='xy', scale_units='xy', scale=1)

