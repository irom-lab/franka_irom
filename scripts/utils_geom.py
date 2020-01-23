import numpy as np

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


##########################################################################
######### Conversion between S03 and euler angle, quaternion #########
##########################################################################

def euler2rot(a):
    """
    Generate rotation matrix from euler anglesusing intrinsic ZYZ convention
    Reference: MLS Chap.2 Euler Angles
    INPUTS
      a - (3,) - euler angles around intrinsic z,y,z axes
    OUTPUTS
      R - (3,3) - rotation matrix as Rz * Ry * Rz
    """

    a = np.asarray(a).flatten()

    # Trig functions of rotations
    c = np.cos(a[0:3])
    s = np.sin(a[0:3])

    # Rotation matrix as Rz * Ry * Rz
    R = np.array([
        [c[0]*c[1]*c[2]-s[0]*s[2], -c[0]*c[1]*s[2]-s[0]*c[2], c[0]*s[1]],
        [s[0]*c[1]*c[2]+c[0]*s[2], -s[0]*c[1]*s[2]+c[0]*c[2], s[0]*s[1]],
        [-s[1]*c[2], s[1]*s[2], c[1]]
        ])
    return R


def rot2euler(R):
    """
    Generate euler angles from rotation matrix using intrinsic ZYZ convention
    Reference: MLS Chap.2 Euler Angles
    Inputs
      R - (3,3) - rotation matrix SO(3)
    OUTPUTS
      a - (3,) - euler angles, i.e. rotations around z,y,z axes
    """

    beta = np.arctan2(np.sqrt(R[2,0]**2+R[2,1]**2), R[2,2])
    alpha = np.arctan2(R[1,2]/np.sin(beta), R[0,2]/np.sin(beta))
    gamma = np.arctan2(R[2,1]/np.sin(beta), -R[2,0]/np.sin(beta))
    return np.array([alpha, beta, gamma])


# Convert quatenornion (a,b,c,w, unnormalized) to SO(3)
def quat2rot(q, w_first=False):
    if w_first:
        a = q[1]
        b = q[2]
        c = q[3]
        w = q[0]
    else:
        a = q[0]
        b = q[1]
        c = q[2]
        w = q[3]

    out = np.zeros((3,3))
    out[0,0] = w**2+a**2-b**2-c**2
    out[0,1] = 2*a*b-2*c*w
    out[0,2] = 2*a*c+2*b*w
    out[1,0] = 2*a*b+2*c*w
    out[1,1] = w**2-a**2+b**2-c**2
    out[1,2] = 2*b*c-2*a*w
    out[2,0] = 2*a*c-2*b*w
    out[2,1] = 2*b*c+2*a*w
    out[2,2] = w**2-a**2-b**2+c**2

    return out

# Convert SO(3) to quatenornion (a,b,c,w, unnormalized)
# https://en.wikipedia.org/wiki/Rotation_formalisms_in_three_dimensions#Rotation_matrix_%E2%86%94_quaternion
def rot2quat(R):
    w = 0.5*np.sqrt(1+np.trace(R))
    a = 0.25/w*(R[2,1]-R[1,2])
    b = 0.25/w*(R[0,2]-R[2,0])
    c = 0.25/w*(R[1,0]-R[0,1])
    return np.array([a,b,c,w])

def euler2quat(a):
    return rot2quat(euler2rot(a))

def quat2euler(q):
    return rot2euler(quat2rot(q))


def log_rot(R):
    """
    Generate angular velocity so(3) from SO(3)
    """

    theta = np.arccos((min(np.trace(R),1.0)-1)/2)
    
    return np.array([R[2,1]-R[1,2], R[0,2]-R[2,0], R[1,0]-R[0,1]])/(2*np.sin(theta))


# def exp_rot(w):
#     """
#     Generate SO(3) from angular velocity so(3)
#     """
    
    


############################################################################
################## SO(3) Conversion with normal vector #####################
############################################################################

# Find SO(3) that rotate x vector to y, not unique since not aligning frames but just normals (x,y required to be unit vectors!!!!)
def vec2rot(x,y):
# https://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d
# rotate around cross product of x and y by arccos(dot(x,y))
# wont work if exactly opposite direction for x and y
    v = np.cross(x,y)
    s = np.linalg.norm(v)
    c = np.dot(x,y)
    vs = skew(v)

    # if x == -y

    return np.eye(3) + vs + vs.dot(vs)*(1/(1+c))

def orient(z):
    """
    R = orient  rotation matrix bringing vector in line with [0,0,1]
    INPUTS
      z - 3 x 1 - vector to align with [0,0,1]
    OUTPUTS
      R - 3 x 3 - rogation matrix which orients coordinate system
    """
    x0 = z.reshape((3,1))
    R1 = euler(np.array([0,np.arctan2(x0[0,0],x0[2,0]),0]))
    x1 = np.dot(R1,x0)
    R2 = euler(np.array([-np.arctan2(x1[1,0],x1[2,0]),0,0]))
    x2 = np.dot(R2,x1)

    return np.dot(R2,R1)

# Rotate a vector v by a quaternion q (a,b,c,w), return a 3D vector
def vecQuat2vec(v,q):
    r = np.concatenate((v,[0]))  # add zero to the end of the array
    q_conj = np.array([-q[0],-q[1],-q[2],q[3]])
    out = quatMult(quatMult(np.array(q),r),q_conj)[:3]
    return out/np.linalg.norm(out)

# Find quaternion that rotates x vector to y , not unique since not aligning frames but just normals
# https://stackoverflow.com/questions/1171849/finding-quaternion-representing-the-rotation-from-one-vector-to-another
def vecs2quat(x,y):
    out = np.zeros(4)
    out[:3] = np.cross(x, y)
    out[3] = np.linalg.norm(x)*np.linalg.norm(y)+np.dot(x, y)
    if np.linalg.norm(out) < 1e-4:
        return np.append(-x, [0])  # 180 rotation
    return out/np.linalg.norm(out)


################################################################################

# Multiply two quaternions (a,b,c,w)
def quatMult(p, q):
    w = p[3]*q[3] - np.dot(p[:3], q[:3])
    abc = p[3]*q[:3] + q[3]*p[:3] + np.cross(p[:3], q[:3])
    return np.hstack((abc, w))

# Convert 3D vector to 3x3 skew-symmetric matrix
def skew(z):
    return np.array([[0,    -z[2], z[1]],
                    [z[2],  0,    -z[0]],
                    [-z[1], z[0], 0]])

# Get angle between two vectors
def angleBwVec(p,q):
    p = np.array(p)
    q = np.array(q)
    ct = np.dot(p,q)/(np.linalg.norm(p)*np.linalg.norm(q))
    return np.arccos(ct)


def SO3_6D_np(b1, a2):
    b2 = a2 - np.dot(b1, a2)*b1
    b2 /= np.linalg.norm(b2)
    b3 = np.cross(b1, b2)
    return b2, b3
