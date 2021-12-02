import numpy as np

def point_3d_offset_joint(joint, point):
    """
    joint: [x, y, z] or [[x, y, z] + [rx, ry, rz]]
    point: N * 3
    """
    if len(joint) == 2:
        P0 = np.array(joint[0])
        P  = np.array(point)
        l  = np.array(joint[1]).reshape(1, 3)
        P0P= P - P0
        # projection of P in joint minus P
        PP = np.dot(P0P, l.T) * l / np.linalg.norm(l)**2  - P0P
    return PP
