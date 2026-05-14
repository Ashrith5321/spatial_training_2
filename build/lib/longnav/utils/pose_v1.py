import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------------------------------------------------------------
# MATH HELPERS
# ------------------------------------------------------------------------------

def q_rotate_vector(q, v):
    """
    Rotates vector v by quaternion q.
    q: (B, 4) [x, y, z, w]
    v: (B, 3)
    """
    q_vec = q[:, :3]
    q_w = q[:, 3:]
    # dim=-1 ensures robust cross product
    t = 2.0 * torch.linalg.cross(q_vec, v, dim=-1)
    return v + (q_w * t) + torch.linalg.cross(q_vec, t, dim=-1)

def q_inverse(q):
    """Conjugate of unit quaternion is inverse"""
    inv = q.clone()
    inv[:, :3] = -inv[:, :3]
    return inv

def q_multiply(q1, q2):
    """
    Multiply two quaternions. Output is q1 * q2 (rotation q2 followed by q1)
    """
    x1, y1, z1, w1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    x2, y2, z2, w2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
    
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    
    return torch.stack([x, y, z, w], dim=-1)

def deadzone_loss(input_tensor, threshold=15.0):
    """
    Penalizes values only if they exceed the threshold radius.
    """
    mag = input_tensor.norm(dim=-1)
    excess = F.relu(mag - threshold)
    return (excess ** 2).mean()

def matrix_to_quaternion(R):
    """
    Robust conversion of 3x3 Rotation Matrix to Quaternion [x, y, z, w].
    Handles the trace singularities for batch size 1.
    """
    tr = R[0,0] + R[1,1] + R[2,2]
    if tr > 0:
        S = torch.sqrt(tr + 1.0) * 2
        qw = 0.25 * S
        qx = (R[2,1] - R[1,2]) / S
        qy = (R[0,2] - R[2,0]) / S
        qz = (R[1,0] - R[0,1]) / S
    elif (R[0,0] > R[1,1]) and (R[0,0] > R[2,2]):
        S = torch.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2]) * 2
        qw = (R[2,1] - R[1,2]) / S
        qx = 0.25 * S
        qy = (R[0,1] + R[1,0]) / S
        qz = (R[0,2] + R[2,0]) / S
    elif (R[1,1] > R[2,2]):
        S = torch.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2]) * 2
        qw = (R[0,2] - R[2,0]) / S
        qx = (R[0,1] + R[1,0]) / S
        qy = 0.25 * S
        qz = (R[1,2] + R[2,1]) / S
    else:
        S = torch.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1]) * 2
        qw = (R[1,0] - R[0,1]) / S
        qx = (R[0,2] + R[2,0]) / S
        qy = (R[1,2] + R[2,1]) / S
        qz = 0.25 * S
    return torch.stack([qx, qy, qz, qw]).to(R.device)

def align_trajectory_umeyama(p, g):
    """
    Aligns points p to g using Umeyama (SVD).
    Returns: p_aligned, s, R, g_centered
    """
    # 1. Center the clouds
    mu_p = p.mean(dim=0, keepdim=True)
    mu_g = g.mean(dim=0, keepdim=True)
    
    p_centered = p - mu_p
    g_centered = g - mu_g
    
    # 2. Covariance H = P^T * G
    H = torch.matmul(p_centered.transpose(0, 1), g_centered)
    
    # 3. SVD: H = U S V^T
    U, S, Vh = torch.linalg.svd(H)
    V = Vh.mH 
    
    # Rotation R = V @ U.T
    R = torch.matmul(V, U.transpose(-2, -1))
    
    # 4. Reflection Check (Fix for In-place Error)
    if torch.linalg.det(R) < 0:
        # Use diagonal matrix correction instead of in-place V modification
        S_ref = torch.eye(3, device=p.device, dtype=p.dtype)
        S_ref[2, 2] = -1
        
        # R = V @ S_ref @ U.T
        R = torch.matmul(torch.matmul(V, S_ref), U.transpose(-2, -1))
    
    # 5. Optimal Scale
    p_rotated = torch.matmul(p_centered, R.transpose(0, 1))
    
    denom = torch.sum(p_centered ** 2)
    nom = torch.sum(p_rotated * g_centered)
    
    if denom > 1e-6:
        s = nom / denom
        s = torch.clamp(s, min=0.001, max=1000.0)
    else:
        s = torch.tensor(1.0, device=p.device)
        
    return p_rotated, s, R, g_centered

# ------------------------------------------------------------------------------
# LOSS MODULE
# ------------------------------------------------------------------------------

class SfMPoseLoss(nn.Module):
    def __init__(self, 
                 up_vector=(0, 1, 0),       # HM3D/Habitat usually Y-up
                 frame0_leash=15.0):         # Radius of the deadzone for Frame 0
        super().__init__()
        self.register_buffer("up_vector", torch.tensor(up_vector, dtype=torch.float32))
        self.frame0_leash = frame0_leash

    def forward(self, pred_t, pred_q, gt_t, gt_q, batch_counts):
        """
        Calculates Scale-Invariant and Rotation-Aligned Trajectory Loss (Umeyama).
        Also enforces Gravity constraints and absolute drift limits.
        """
        total_rel_trans_loss = 0
        total_rel_rot_loss = 0
        total_gravity_loss = 0
        total_scale_loss = 0
        total_frame0_loss = 0
        
        cursor = 0
        num_episodes = len(batch_counts)
        if num_episodes == 0:
            return {}

        for count in batch_counts:
            # === 1. Slice Episode ===
            p_t = pred_t[cursor : cursor+count]
            p_q = pred_q[cursor : cursor+count]
            g_t = gt_t[cursor : cursor+count]
            g_q = gt_q[cursor : cursor+count]
            
            # === 2. Frame 0 Leash (Regularization) ===
            # Prevents the raw output from drifting to 1e9 meters
            total_frame0_loss += deadzone_loss(p_t[0], threshold=self.frame0_leash)

            # === 3. Gravity Loss (Absolute Orientation) ===
            # Ensures predicted "Up" aligns with World "Up" (Y-axis)
            # This constrains Roll/Pitch, which Umeyama cannot fix if path is flat.
            up_vec = self.up_vector.to(p_q.device).unsqueeze(0).expand(count, -1)
            pred_up = q_rotate_vector(q_inverse(p_q), up_vec)
            gt_up = q_rotate_vector(q_inverse(g_q), up_vec)
            total_gravity_loss += (1.0 - F.cosine_similarity(pred_up, gt_up, dim=-1)).mean()

            # === 4. Global Consensus Alignment (Umeyama) ===
            # Solves Yaw, Scale, and Translation ambiguity globally.
            
            # Check for degenerate ground truth (stationary or too short)
            g_var = torch.var(g_t, dim=0).sum()
            
            if count < 4 or g_var < 1e-4:
                # Fallback: Simple Center-alignment if sequence is too simple
                p_aligned = p_t - p_t.mean(dim=0)
                g_aligned = g_t - g_t.mean(dim=0)
                
                # Fallback Rotation: Relative to Frame 0
                p_q_aligned = q_multiply(q_inverse(p_q[0:1].expand(count, -1)), p_q)
                g_q_aligned = q_multiply(q_inverse(g_q[0:1].expand(count, -1)), g_q)
                
            else:
                # A. Align Translation Trajectories
                # p_rotated is centered and rotated to match g_centered
                p_rotated, s, R, g_centered = align_trajectory_umeyama(p_t, g_t)
                
                # Apply Scale
                p_aligned = p_rotated * s
                g_aligned = g_centered # Target is the centered GT
                
                # Scale Loss
                total_scale_loss += (torch.log(torch.abs(s) + 1e-6)**2)
                
                # B. Align Rotation (The "Crab Walk" Fix)
                # If we rotated the Path by R to match GT, we must rotate the
                # Visual Headings by R as well.
                q_align = matrix_to_quaternion(R).to(p_q.device)
                q_align_expanded = q_align.unsqueeze(0).expand(count, -1)
                
                # Apply global correction to predicted quaternions
                p_q_aligned = q_multiply(q_align_expanded, p_q)
                g_q_aligned = g_q

            # === 5. Trajectory Losses ===
            
            # Translation: MSE on aligned point clouds
            total_rel_trans_loss += F.mse_loss(p_aligned, g_aligned)
            
            # Rotation: Cosine distance on aligned quaternions
            # Double cover check: 1 - <q1, q2>^2
            dot_prod = (p_q_aligned * g_q_aligned).sum(dim=1)
            total_rel_rot_loss += (1.0 - dot_prod**2).mean()

            cursor += count

        return {
            "loss_trans": total_rel_trans_loss / num_episodes,
            "loss_rot": total_rel_rot_loss / num_episodes,
            "loss_grav": total_gravity_loss / num_episodes,
            "loss_leash": total_frame0_loss / num_episodes,
            "loss_scale": total_scale_loss / num_episodes
        }