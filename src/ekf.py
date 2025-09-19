from __future__ import annotations
import numpy as np
from numpy.linalg import inv

class CornerEKF:
    def __init__(self, f_dyn, Q=None, R=None, x0=None):
        self.f = f_dyn
        self.x = np.array([95.0, 90.0, 85.0]) if x0 is None else x0.astype(float)
        self.P = np.eye(3)*4.0
        self.Q = np.eye(3)*0.05 if Q is None else Q
        self.R = np.eye(3)*0.3 if R is None else R

    def jacobian(self, x, u, dt, eps=1e-4):
        J = np.zeros((3,3))
        fx = self.f(x,u,dt)
        for i in range(3):
            x2 = x.copy(); x2[i]+=eps
            fx2 = self.f(x2,u,dt)
            J[:,i] = (fx2-fx)/eps
        return J

    def step(self, z, u, dt):
        x_pred = self.f(self.x, u, dt)
        F = self.jacobian(self.x, u, dt)
        P_pred = F @ self.P @ F.T + self.Q

        z = np.asarray(z, dtype=float)
        m = z.shape[0]
        if m == 1:
            H = np.array([[1.0, 0.0, 0.0]])
        elif m == 2:
            H = np.array([[1.0, 0.0, 0.0],
                          [0.0, 0.0, 1.0]])
        else:
            H = np.eye(3)

        S = H @ P_pred @ H.T + (H @ self.R @ H.T) 
        K = P_pred @ H.T @ inv(S)
        y = z - (H @ x_pred)
        self.x = x_pred + K @ y
        self.P = (np.eye(3) - K @ H) @ P_pred
        return self.x
