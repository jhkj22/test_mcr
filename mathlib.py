import numpy as np

def interior_angle(p):
    return 180 - exterior_angle(p)
def exterior_angle(p):
    p = np.array(p)
    p -= p[1]
    ff = -complex(p[2][0], p[2][1]) / complex(p[0][0], p[0][1])
    return -np.angle(ff) * 180 / np.pi
def rad_deg(r):
    return r * 180 / np.pi
def ver_distance(a, b, c, p):
    return (a * p[0] + b * p[1] + c) / np.sqrt(a ** 2 + b ** 2)

def quantize_dir(v):
    a = np.arctan2(v[1], v[0])
    if -np.pi * 3 / 4 <= a and a < -np.pi / 4:
        return [0, -1]
    elif -np.pi / 4 <= a and a < np.pi / 4:
        return [1, 0]
    elif np.pi / 4 <= a and a < np.pi * 3 / 4:
        return [0, 1]
    else:
        return [-1, 0]































