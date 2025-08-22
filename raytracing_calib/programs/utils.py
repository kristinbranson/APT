import torch
import numpy as np

# Euclidean distance loss function
def euclidean_distance(output, label):
    """
    Arrange data points along dim=0 and components along dim=1
    """    
    if not isinstance(output, torch.Tensor):
        output = torch.tensor(output, dtype=torch.float64)
    if not isinstance(label, torch.Tensor):
        label = torch.tensor(label, dtype=torch.float64)
    distance = torch.norm(output - label, 
                        p=2, 
                        dim=0)
    return distance


def mult_quat(quat1, quat2):
    """Computes the Hamilton product of two quaternions `quat1` * `quat2`.
    This is a general multiplication, the input quaternions do not have to be
    unit quaternions.

    Any number of leading batch dimensions is supported.

    Broadcast rules:
        One of the input quats can be (4,) while the other is (B, 4).

    Args:
        quat1, quat2: Arrays of shape (B, 4) or (4,).

    Returns:
        Product of quat1*quat2, array of shape (B, 4) or (4,).
    """
    a1, b1, c1, d1 = quat1[0,:], quat1[1,:], quat1[2,:], quat1[3,:]
    a2, b2, c2, d2 = quat2[0,:], quat2[1,:], quat2[2,:], quat2[3,:]
    prod = torch.empty_like(quat1) if quat1.shape[1] > quat2.shape[1] else torch.empty_like(
        quat2)
    prod[0,:] = a1 * a2 - b1 * b2 - c1 * c2 - d1 * d2
    prod[1,:] = a1 * b2 + b1 * a2 + c1 * d2 - d1 * c2
    prod[2,:] = a1 * c2 - b1 * d2 + c1 * a2 + d1 * b2
    prod[3,:] = a1 * d2 + b1 * c2 - c1 * b2 + d1 * a2    
    return prod


def conj_quat(quat):
    """Returns the conjugate quaternion of `quat`.
    Any number of leading batch dimensions is supported.
    Args:
        quat: Array of shape (B, 4).
    Returns:
        Conjugate quaternion(s), array of shape (B, 4).
    """
    quat = quat.clone()
    quat[1:,:] = (-1) * quat[1:,:]
    return quat


def reciprocal_quat(quat):
    """Returns the reciprocal quaternion of `quat` such that the product
    of `quat` and its reciprocal gives unit quaternion:
    mult_quat(quat, reciprocal_quat(quat)) == [1., 0, 0, 0]
    Any number of leading batch dimensions is supported.
    Args:
        quat: Array of shape (B, 4).
    Returns:
        Reciprocal quaternion(s), array of shape (B, 4).
    """
    return conj_quat(quat) / torch.linalg.norm(quat, dim=0, keepdims=True)**2


def rotate_vec_with_quat(vec, quat):
    """Uses unit quaternion `quat` to rotate vector `vec` according to:
        vec' = quat vec quat^-1.
    Any number of leading batch dimensions is supported.

    Technically, `quat` should be a unit quaternion, but in this particular
    multiplication (quat vec quat^-1) it doesn't matter because an arbitrary
    constant cancels out in the product.

    Broadcasting works in both directions. That is, for example:
    (i) vec and quat can be [1, 1, 3] and [2, 7, 4], respectively.
    (ii) vec and quat can be [2, 7, 3] and [1, 1, 4], respectively.

    Args:
        vec: Cartesian position vector to rotate, shape (B, 3). Does not have
            to be a unit vector.
        quat: Rotation unit quaternion, (B, 4).

    Returns:
        Rotated vec, (B, 3,).
    """
    vec_aug = torch.zeros(4, vec.shape[1]).to(torch.float64)
    vec_aug[1:,:] = vec
    vec = mult_quat(quat, mult_quat(vec_aug, reciprocal_quat(quat)))
    return vec[1:,:]

def quaternion_to_rotation_matrix(quaternion):
    """
    Convert a quaternion tensor to a rotation matrix while preserving gradients.

    Args:
        quaternion (torch.Tensor): A tensor of shape (4,) representing the quaternion (q_w, q_x, q_y, q_z).

    Returns:
        torch.Tensor: A rotation matrix of shape (3, 3).
    """
    # Extract quaternion components
    q_w, q_x, q_y, q_z = quaternion[0,:], quaternion[1,:], quaternion[2,:], quaternion[3,:]

    # Compute the rotation matrix elements
    r11 = 1 - 2 * (q_y**2 + q_z**2)
    r12 = 2 * (q_x * q_y - q_w * q_z)
    r13 = 2 * (q_x * q_z + q_w * q_y)

    r21 = 2 * (q_x * q_y + q_w * q_z)
    r22 = 1 - 2 * (q_x**2 + q_z**2)
    r23 = 2 * (q_y * q_z - q_w * q_x)

    r31 = 2 * (q_x * q_z - q_w * q_y)
    r32 = 2 * (q_y * q_z + q_w * q_x)
    r33 = 1 - 2 * (q_x**2 + q_y**2)

    # Stack the rows to form the rotation matrix
    R = torch.stack([
        torch.stack([r11, r12, r13]),
        torch.stack([r21, r22, r23]),
        torch.stack([r31, r32, r33])
    ])[:,:,0]
    return R

def rotation_matrix_to_quaternion(R):
    # Ensure R is a valid rotation matrix
    assert R.shape == (3, 3), "Input must be a 3x3 rotation matrix."
    
    trace = torch.trace(R)
    
    if trace > 0:
        S = torch.sqrt(trace + 1.0) * 2  # S=4*q.w
        q_w = 0.25 * S
        q_x = (R[2, 1] - R[1, 2]) / S
        q_y = (R[0, 2] - R[2, 0]) / S
        q_z = (R[1, 0] - R[0, 1]) / S
    else:
        if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:  # q_x is the largest
            S = torch.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2  # S=4*q.x
            q_w = (R[2, 1] - R[1, 2]) / S
            q_x = 0.25 * S
            q_y = (R[0, 1] + R[1, 0]) / S
            q_z = (R[0, 2] + R[2, 0]) / S
        elif R[1, 1] > R[2, 2]:  # q_y is the largest
            S = torch.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2  # S=4*q.y
            q_w = (R[0, 2] - R[2, 0]) / S
            q_x = (R[0, 1] + R[1, 0]) / S
            q_y = 0.25 * S
            q_z = (R[1, 2] + R[2, 1]) / S
        else:  # q_z is the largest
            S = torch.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2  # S=4*q.z
            q_w = (R[1, 0] - R[0, 1]) / S
            q_x = (R[0, 2] + R[2, 0]) / S
            q_y = (R[1, 2] + R[2, 1]) / S
            q_z = 0.25 * S
    quat = torch.stack([q_w, q_x, q_y, q_z]).to(torch.float64)
    if quat.ndim == 1:
        quat = quat[:,None]
    return quat