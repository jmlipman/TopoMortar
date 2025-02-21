import numpy as np
import cc3d, cv2

def update_simple_point(distance, gt):
    non_zero = np.nonzero(distance)
    indice = np.unravel_index(np.argsort(-distance, axis=None), distance.shape)

    if len(gt.shape) == 2:
        for i in range(len(non_zero[0])):
            x = indice[0][len(non_zero[0]) - i - 1]
            y = indice[1][len(non_zero[0]) - i - 1]

            gt = decide_simple_point_2D(gt, x, y)
    else:
        for i in range(len(non_zero[0])):
            x = indice[0][len(non_zero[0]) - i - 1]
            y = indice[1][len(non_zero[0]) - i - 1]
            z = indice[2][len(non_zero[0]) - i - 1]

            gt = decide_simple_point_3D(gt, x, y, z)
    return gt

def decide_simple_point_3D(gt, x, y, z):
    ## extract local patch
    patch = gt[x-1:x+2, y-1:y+2, z-1:z+2]
    if patch.shape != (3,3,3):
        return gt

    ## check local topology
    if patch.shape[0] != 0 and patch.shape[1] != 0 and patch.shape[2] != 0:
        try:
            _, number_fore = cc3d.connected_components(patch, 6, return_N = True)
            _, number_back = cc3d.connected_components(1-patch, 26, return_N = True)
        except:
            number_fore = 0
            number_back = 0
            pass
        label = number_fore * number_back

        ## flip the simple point
        if (label == 1):
            gt[x,y,z] = 1 - gt[x,y,z]

    return gt

def decide_simple_point_2D(gt, x, y):

    ## extract local patch
    patch = gt[x-1:x+2, y-1:y+2]
    if patch.shape != (3,3):
        return gt

    ## check local topology
    number_fore, _ = cv2.connectedComponents(patch, 4)
    number_back, _ = cv2.connectedComponents(1-patch, 8)

    label = (number_fore-1) * (number_back-1)

    ## flip the simple point
    if (label == 1):
        gt[x,y] = 1 - gt[x,y]

    return gt

