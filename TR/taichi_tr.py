# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 03:14:23 2023

@author: JChonpca_Huang
"""

import taichi as ti
import taichi.math as tm
import os
os.environ["KMP_DUPLICATE_LIB_OK"]  =  "TRUE"


ti.init(arch=ti.cuda, default_fp=ti.f64 ,debug=True, kernel_profiler=True)

n_dim = 2

eps = 0.00001

x = ti.field(dtype=ti.f64, shape=n_dim)

x_new = ti.field(dtype=ti.f64, shape=n_dim)

G = ti.field(dtype=ti.f64, shape=(n_dim,1))

H = ti.field(dtype=ti.f64, shape=(n_dim,n_dim))

x_numpy = x.to_numpy().tolist()

G_numpy = G.to_numpy().tolist()

H_numpy = H.to_numpy().tolist()


@ti.func
def F(xx: ti.template()) -> float:
    
    y = 0.0
    
    x1 = xx[0]
    
    x2 = xx[1]
    
    y = 100*(x2-x1**2)**2+(1-x1)**2
    
    return y



@ti.kernel
def F_k(xx: ti.template()) -> float:
    
    y = 0.0

    x1 = xx[0]

    x2 = xx[1]

    y = 100*(x2-x1**2)**2+(1-x1)**2

    return y


@ti.func
def num_diff(xx: ti.template()) -> ti.types.matrix(n_dim,1,ti.f64): 
    
    gradient = ti.Matrix(arr=G_numpy)
    
    ti.loop_config(serialize=False)
    
    for i in range(n_dim):
                
        left_point = ti.Vector(arr=x_numpy)
        
        right_point = ti.Vector(arr=x_numpy)
        
        ti.loop_config(serialize=False)
        
        for j in range(n_dim):
            
            left_point[j] = xx[j]
            
            right_point[j] = xx[j]
        
        left_point[i] = xx[i] - eps
        
        right_point[i] = xx[i] + eps
        
        gradient[i,0] = (F(right_point) - F(left_point))/(2*eps)
    
    return gradient




@ti.kernel
def num_diff_k(xx: ti.template()) -> ti.types.matrix(n_dim,1,ti.f64): 

    gradient = ti.Matrix(arr=G_numpy)

    ti.loop_config(serialize=False)

    for i in range(n_dim):

        left_point = ti.Vector(arr=x_numpy)

        right_point = ti.Vector(arr=x_numpy)

        ti.loop_config(serialize=False)

        for j in range(n_dim):

            left_point[j] = xx[j]

            right_point[j] = xx[j]

        left_point[i] = xx[i] - eps

        right_point[i] = xx[i] + eps

        gradient[i,0] = (F(right_point) - F(left_point))/(2*eps)

    return gradient


@ti.func
def num_hassian(xx: ti.template()) -> ti.types.matrix(n_dim,n_dim,ti.f64): 
    
    hassian = ti.Matrix(arr=H_numpy)
    
    ti.loop_config(serialize=False)    
    
    for i,j in ti.ndrange(n_dim,n_dim):
        
        left_point_j = ti.Vector(arr=x_numpy)
        
        right_point_j = ti.Vector(arr=x_numpy)
        
        left_point_i = ti.Vector(arr=x_numpy)
        
        right_point_i = ti.Vector(arr=x_numpy)
        
        ti.loop_config(serialize=False)
        
        for k in range(n_dim):
            
            left_point_j[k] = xx[k]
            
            right_point_j[k] = xx[k]
            
            left_point_i[k] = xx[k]
            
            right_point_i[k] = xx[k]


        right_point_j[j] = xx[j] + eps
        
        left_point_j[j] = xx[j] - eps
        
        diff_i = (num_diff(right_point_j)[i,0] - num_diff(left_point_j)[i,0])/(4*eps)


        right_point_i[i] = xx[i] + eps
        
        left_point_i[i] = xx[i] - eps
        
        diff_j = (num_diff(right_point_i)[j,0] - num_diff(left_point_i)[j,0])/(4*eps)

                
        hassian[i,j] = diff_i + diff_j
    
    return hassian



@ti.kernel
def num_hassian_k(xx: ti.template()) -> ti.types.matrix(n_dim,n_dim,ti.f64): 

    hassian = ti.Matrix(arr=H_numpy)

    ti.loop_config(serialize=False)    

    for i,j in ti.ndrange(n_dim,n_dim):

        left_point_j = ti.Vector(arr=x_numpy)

        right_point_j = ti.Vector(arr=x_numpy)

        left_point_i = ti.Vector(arr=x_numpy)

        right_point_i = ti.Vector(arr=x_numpy)

        ti.loop_config(serialize=False)

        for k in range(n_dim):

            left_point_j[k] = xx[k]

            right_point_j[k] = xx[k]

            left_point_i[k] = xx[k]

            right_point_i[k] = xx[k]


        right_point_j[j] = xx[j] + eps

        left_point_j[j] = xx[j] - eps

        diff_i = (num_diff(right_point_j)[i,0] - num_diff(left_point_j)[i,0])/(4*eps)


        right_point_i[i] = xx[i] + eps

        left_point_i[i] = xx[i] - eps

        diff_j = (num_diff(right_point_i)[j,0] - num_diff(left_point_i)[j,0])/(4*eps)


        hassian[i,j] = diff_i + diff_j

    return hassian




@ti.func
def mk_function(xx: ti.template(),
                p: ti.template()) -> float:
    
    mk = 0.0
    
    fk = F(xx)
    
    gk = num_diff(xx)
    
    Bk = num_hassian(xx)
        
    mk = fk + ((gk.transpose()) @ p)[0,0] + 0.5 * ((p.transpose() @ Bk) @ p)[0,0]
    
    return mk



@ti.kernel
def mk_function_k(xx: ti.template(),
                p: ti.template()) -> float:
    
    mk = 0.0
    
    fk = F(xx)
    
    gk = num_diff(xx)
    
    Bk = num_hassian(xx)
        
    mk = fk + ((gk.transpose()) @ p)[0,0] + 0.5 * ((p.transpose() @ Bk) @ p)[0,0]
    
    return mk


@ti.func
def Dogleg_Method(xx: ti.template(),
                  delta: float) -> ti.types.matrix(n_dim,1,ti.f64):

    g = num_diff(xx)
    
    B = num_hassian(xx)
    
    inv_B = B.inverse()
        
    PB = (-inv_B) @ g
        
    PU = - (((g.transpose()) @ g) / (((g.transpose()) @ B) @ g))[0,0] * g
        
    PB_U = PB-PU
        
    PB_norm = PB.norm()
        
    PU_norm = PU.norm()
        
    PB_U_norm = PB_U.norm()
        
    tao = 0.0   # !!! important
     
    if PB_norm <= delta:
        
        tao += 2
                
    elif PU_norm >= delta:
        
        tao += delta/PU_norm
        
    else:
        
        factor = (PU.transpose() @ PB_U)[0,0] * (PU.transpose() @ PB_U)[0,0]
                
        tmp_tao = -2 * (PU.transpose() @ PB_U)[0,0] + 2 * (tm.max(0,factor-PB_U_norm*PB_U_norm*(PU_norm*PU_norm-delta*delta))**0.5)
                
        tao += tmp_tao / (2 * PB_U_norm * PB_U_norm) + 1.0
            
    s_k = ti.Matrix(arr=G_numpy) # !!! important
     
    if 0.0 <= tao <= 1.0:
        
        s_k += tao*PU
        
    elif 1.0 < tao <= 2.0 :
        
        s_k += PU+(tao-1)*(PB-PU)
                
    return s_k


@ti.kernel
def Dogleg_Method_k(xx: ti.template(),
                  delta: float) -> ti.types.matrix(n_dim,1,ti.f64):

    g = num_diff(xx)
    
    B = num_hassian(xx)
    
    inv_B = B.inverse()
        
    PB = (-inv_B) @ g
        
    PU = - (((g.transpose()) @ g) / (((g.transpose()) @ B) @ g))[0,0] * g
        
    PB_U = PB-PU
        
    PB_norm = PB.norm()
        
    PU_norm = PU.norm()
        
    PB_U_norm = PB_U.norm()
        
    tao = 0.0   # !!! important
     
    if PB_norm <= delta:
        
        tao += 2
                
    elif PU_norm >= delta:
        
        tao += delta/PU_norm
        
    else:
        
        factor = (PU.transpose() @ PB_U)[0,0] * (PU.transpose() @ PB_U)[0,0]
                
        tmp_tao = -2 * (PU.transpose() @ PB_U)[0,0] + 2 * (tm.max(0,factor-PB_U_norm*PB_U_norm*(PU_norm*PU_norm-delta*delta))**0.5)
                
        tao += tmp_tao / (2 * PB_U_norm * PB_U_norm) + 1.0
    
    s_k = ti.Matrix(arr=G_numpy) # !!! important
            
    if 0.0 <= tao <= 1.0:
        
        s_k += tao*PU
        
    elif 1.0 < tao <= 2.0 :
        
        s_k += PU+(tao-1)*(PB-PU)
                
    return s_k

@ti.func      
def TrustRegion(xx: ti.template(),
                delta_max: float):
            
    delta = delta_max
            
    epsilon = 1e-9
    
    maxk = 1000
    
    ti.loop_config(serialize=True)
    
    for j in range(maxk):
        
        g = num_diff(x)
        
        g_norm = g.norm()
        
        if g_norm < epsilon:
            
            break

        sk = Dogleg_Method(x, delta)
        
        sk_norm = sk.norm()
        
        ti.loop_config(serialize=False)

        for i in range(n_dim):
            
            x_new[i] = xx[i] + sk[i,0]
            
        fun_k = F(x)
        
        fun_new = F(x_new)
        
        zero = ti.Matrix(arr=G_numpy)
        
        r = 0.0
        
        r = (fun_k - fun_new) / (mk_function(x, zero) - mk_function(x, sk))
        
        if r < 0.25:
            
            delta = delta / 4.0
            
        elif (r > 0.75) and (sk_norm == delta):
            
            delta = tm.min(2 * delta, delta_max)
        
        
        if r > 0.0:
                        
            ti.loop_config(serialize=False)
    
            for i in range(n_dim):
                
                x[i] = x_new[i]
    

@ti.kernel    
def TrustRegion_k(xx: ti.template(),
                delta_max: float):
        
    delta = delta_max
            
    epsilon = 1e-9
    
    maxk = 1000
    
    ti.loop_config(serialize=True)
    
    for j in range(maxk):
        
        g = num_diff(xx)
        
        g_norm = g.norm()
        
        if g_norm < epsilon:
            
            break

                    
        sk = Dogleg_Method(xx, delta)
        
        sk_norm = sk.norm()
        
        ti.loop_config(serialize=False)

        for i in range(n_dim):
            
            x_new[i] = xx[i] + sk[i,0]
            
        fun_k = F(xx)
        
        fun_new = F(x_new)
        
        zero = ti.Matrix(arr=G_numpy)
        
        r = 0.0
        
        r = (fun_k - fun_new) / (mk_function(xx, zero) - mk_function(xx, sk))
        
        if r < 0.25:
            
            delta = delta / 4.0
            
        elif (r > 0.75) and (sk_norm == delta):
            
            delta = tm.min(2 * delta, delta_max)
        
        
        if r > 0.0:
                        
            ti.loop_config(serialize=False)
    
            for i in range(n_dim):
                
                x[i] = x_new[i]


@ti.kernel
def test():
    
    TrustRegion(x,20.0)


import time

if __name__ == '__main__':
    
    x.fill(0.0)
    
    x_new.fill(0.0)
    
    a = time.time()
    
    test()
    
    b = time.time()
    
    print(b-a)
    
    x.fill(0.0)
    
    x_new.fill(0.0)
    
    a = time.time()
    
    test()
    
    b = time.time()
    
    print(b-a)
    
    x.fill(0.0)
    
    x_new.fill(0.0)
    
    a = time.time()
    
    TrustRegion_k(x,20.0)
    
    b = time.time()
    
    print(b-a)
    
    x.fill(0.0)
    
    x_new.fill(0.0)
    
    a = time.time()
    
    TrustRegion_k(x,20.0)
    
    b = time.time()
    
    print(b-a)