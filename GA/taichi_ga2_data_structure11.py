# -*- coding: utf-8 -*-
"""
Created on Sat Feb 25 15:37:14 2023

@author: JChonpca_Huang
"""

import taichi as ti
import taichi.math as tm
import os
os.environ["KMP_DUPLICATE_LIB_OK"]  =  "TRUE"

ti.init(arch=ti.cuda, default_fp=ti.f32 ,debug=True, kernel_profiler=True)



@ti.func
def F(g: int,
      n: int):
    
    x = Envo_DX[n]
        
    y = 3*(1-x[0])**2*tm.exp(-(x[0]**2)-(x[1]+1)**2)-10*(x[0]/5-x[0]**3-x[1]**5)*tm.exp(-x[0]**2-x[1]**2)-1/3**tm.exp(-(x[0]+1)**2-x[1]**2)
        
    return -y




@ti.kernel
def init_pop(g:int):
    
    
    for i,j in ti.ndrange(POP_SIZE,DNA_SIZE*N_DIM):
        
        init_x = int(ti.random()*2)
        
        Envo_EX[i,j]  = init_x
        

@ti.kernel
def select(g: int):
        
    for i in range(POP_SIZE):
        
        best_index = -1
        
        best_y = -1.0
        
        for j in range(TOURN_SIZE):
            
            if j == 0:
                
                best_index = int(ti.random()*POP_SIZE)
                
                best_y = Envo_Y[best_index]
            
            else:
                
                tmp_best_index = int(ti.random()*POP_SIZE)
                
                tmp_best_y = Envo_Y[tmp_best_index]
                
                if tmp_best_y > best_y:
                    
                    best_index = tmp_best_index
                
        ti.loop_config(serialize=True)
        
        for k in range(DNA_SIZE*N_DIM):
            
            Envo_SEX[i,k] = Envo_EX[best_index,k]            
        
        if (int(i%2) == 1):
            
            if ti.random() < CROSSOVER_RATE:
                
                n1 = ti.random()
                
                n2 = ti.random()
                
                if n1 > n2:
                    
                    n1, n2 = n2, n1
                
                n1 = int(n1*DNA_SIZE*N_DIM)
                
                n2 = int(n2*DNA_SIZE*N_DIM)
                
                exchange(n1, n2, g, i)

            
            
@ti.func
def exchange(n1: int, 
             n2: int,
             g: int,
             n: int):
    
    
    for i in range(n1,n2):
        
        
        Envo_SEX[n-1,i] = Envo_SEX[n,i]    
        
        Envo_SEX[n,i] = Envo_SEX[n-1,i] 

                

    

@ti.func
def extractx(g: int):
    
    for i,j in ti.ndrange(POP_SIZE,DNA_SIZE*N_DIM):
                
        if ti.random() < MUTATION_RATE:
            
            Envo_SEX[i,j] = int(ti.random()*2)

        dim_count = int(j // N_DIM) # 整数
        
        dim_index = int(j % N_DIM) # 余数
        
        Envo_EDX[i,dim_index][dim_count] = Envo_SEX[i,j]

@ti.func
def transx(g: int,
           n: int,
           m: int):
    
    num = 0.0
    
    for i in range(DNA_SIZE):
        
        num += (Envo_EDX[n,m][i])*(2**(DNA_SIZE-i))
    
    num = num/float(2**DNA_SIZE-1)
    
    num = num*(HIGH_BOUND[m]-LOW_BOUND[m]) + LOW_BOUND[m]
    
    Envo_DX[n][m] = num
    
@ti.kernel
def decodex(g: int):
    
    extractx(g)
    
    for i,j in ti.ndrange(POP_SIZE,N_DIM):
        
        transx(g, i, j)
    
    # Envo_DX.from_numpy(s)
    



# @ti.func
# def decodex3(g: int):
    
#     Envo_DX.fill(0)
    
#     for i,j in ti.ndrange(POP_SIZE,DNA_SIZE*N_DIM):
                
#         if ti.random() < MUTATION_RATE:
            
#             Envo_SEX[i,j] = int(ti.random()*2)

#         dim_count = int(j // N_DIM) # 整数 # N_DIM
        
#         dim_index = int(j % N_DIM) # 余数  # DNA_SIZE
        
#         Envo_EDX[i,dim_index][dim_count] = Envo_SEX[i,j]
        
        
#     num = 0.0
    
#     for i in range(DNA_SIZE):
        
#         num += (Envo_EDX[n,m][i])*(2**(DNA_SIZE-i))
    
#     num = num/float(2**DNA_SIZE-1)
    
#     num = num*(HIGH_BOUND[m]-LOW_BOUND[m]) + LOW_BOUND[m]
    
#     Envo_DX[n][m] = num










@ti.kernel
def select2(g: int):
        
    for i in range(POP_SIZE):
        
        best_index = -1
        
        best_y = -1.0
        
        for j in range(TOURN_SIZE):
            
            if j == 0:
                
                best_index = int(ti.random()*POP_SIZE)
                
                best_y = Envo_Y[best_index]
            
            else:
                
                tmp_best_index = int(ti.random()*POP_SIZE)
                
                tmp_best_y = Envo_Y[tmp_best_index]
                
                if tmp_best_y > best_y:
                    
                    best_index = tmp_best_index
                
        ti.loop_config(serialize=True)
        
        for k in range(DNA_SIZE*N_DIM):
            
            Envo_EX[i,k] = Envo_SEX[best_index,k]            
        
        if (int(i%2) == 1):
            
            if ti.random() < CROSSOVER_RATE:
                
                n1 = ti.random()
                
                n2 = ti.random()
                
                if n1 > n2:
                    
                    n1, n2 = n2, n1
                
                n1 = int(n1*DNA_SIZE*N_DIM)
                
                n2 = int(n2*DNA_SIZE*N_DIM)
                
                exchange2(n1, n2, g, i)

@ti.func
def exchange2(n1: int, 
             n2: int,
             g: int,
             n: int):
    
    
    for i in range(n1,n2):
        
        
        Envo_EX[n-1,i] = Envo_EX[n,i]    
        
        Envo_EX[n,i] = Envo_EX[n-1,i] 

                

    

@ti.func
def extractx2(g: int):
            
    for i,j in ti.ndrange(POP_SIZE,DNA_SIZE*N_DIM):
        
        # Envo_EX[i,j] = Envo_SEX[i,j]
        
        if ti.random() < MUTATION_RATE:
            
            Envo_EX[i,j] = int(ti.random()*2)

        dim_count = int(j // N_DIM) # 整数
        
        dim_index = int(j % N_DIM) # 余数
        
        Envo_EDX[i,dim_index][dim_count] = Envo_EX[i,j]
        

@ti.func
def transx2(g: int,
           n: int,
           m: int):
    
    num = 0.0
    
    for i in range(DNA_SIZE):
        
        num += (Envo_EDX[n,m][i])*(2**(DNA_SIZE-i))
    
    num = num/float(2**DNA_SIZE-1)
    
    num = num*(HIGH_BOUND[m]-LOW_BOUND[m]) + LOW_BOUND[m]
    
    Envo_DX[n][m] = num
    
@ti.kernel
def decodex2(g: int):
    
    extractx2(g)
    
    for i,j in ti.ndrange(POP_SIZE,N_DIM):
        
        transx2(g, i, j)
    



@ti.kernel
def fitness(g: int):
    
    for i in range(POP_SIZE):
        
        Envo_Y[i] = F(g,i)
        
@ti.kernel
def best_fitness(g: int):
        
    best_index = -1
    
    best_y = -1.0
    
    ti.loop_config(serialize=True)
    
    for i in range(POP_SIZE):
        
        if i == 0:
            
            best_index = i
            
            best_y = Envo_Y[best_index]
        
        else:
            
            if Envo_Y[i] > best_y:
                
                best_index = i
                
                best_y = Envo_Y[best_index]
                
    
    Envo_Best_Y[g] = best_y
    
    Envo_Best_Index[g][0] = best_index
    

    



def main(n_dim, low_bound, high_bound):
    
    global N_DIM
    global DNA_SIZE
    global POP_SIZE
    global N_GENERATIONS
    global TOURN_SIZE
    global CROSSOVER_RATE
    global MUTATION_RATE
    global HIGH_BOUND
    global LOW_BOUND
    global Envo_EX
    global Envo_DX
    global Envo_EDX
    global Envo_SEX
    global Envo_SEX
    global Envo_SEX
    global Envo_Y
    global Envo_Best_Y
    global Envo_Best_Index
    
    
    # N_DIM = ti.field(ti.i32, shape=())

    # DNA_SIZE = ti.field(ti.i32, shape=())

    # POP_SIZE = ti.field(ti.i32, shape=())

    # N_GENERATIONS = ti.field(ti.i32, shape=())
    
    # TOURN_SIZE = ti.field(ti.i32, shape=())

    # CROSSOVER_RATE = ti.field(ti.f32, shape=())

    # MUTATION_RATE = ti.field(ti.f32, shape=())

    N_DIM = n_dim

    DNA_SIZE = 12

    POP_SIZE = 500

    N_GENERATIONS = 1000
    
    TOURN_SIZE = 5

    CROSSOVER_RATE = 0.8

    MUTATION_RATE = 0.05
    
    
    HIGH_BOUND = ti.field(ti.f32, shape=N_DIM)
    
    for i in range(len(high_bound)):
        
        HIGH_BOUND[i] = high_bound[i] 


    LOW_BOUND  = ti.field(ti.f32, shape=N_DIM)

    for i in range(len(low_bound)):
        
        LOW_BOUND[i] = low_bound[i]
        
    
    # Envo_EX  = ti.field(dtype=ti.i32, shape=(POP_SIZE, DNA_SIZE*N_DIM))
    
    # Envo_SEX = ti.field(dtype=ti.i32, shape=(POP_SIZE, DNA_SIZE*N_DIM))
    
    # Envo_SEX = ti.field(dtype=ti.i32, shape=(POP_SIZE, DNA_SIZE*N_DIM))
    
    # Envo_SEX = ti.field(dtype=ti.i32, shape=(POP_SIZE, DNA_SIZE*N_DIM))
    
    Envo_EX  = ti.field(dtype=ti.i32)
    
    Envo_SEX = ti.field(dtype=ti.i32)
    
    # Envo_SEX = ti.field(dtype=ti.i32)
    
    # Envo_SEX = ti.field(dtype=ti.i32)
    
    
    # ti.root.dense(ti.ij, (POP_SIZE, DNA_SIZE*N_DIM)).place(Envo_EX,Envo_SEX)

    
    
    ti.root.dense(ti.ij, (POP_SIZE, DNA_SIZE*N_DIM)).place(Envo_EX)
    
    ti.root.dense(ti.ij, (POP_SIZE, DNA_SIZE*N_DIM)).place(Envo_SEX)
    
    # ti.root.dense(ti.ij, (POP_SIZE, DNA_SIZE*N_DIM)).place(Envo_SEX)
    
    # ti.root.dense(ti.ij, (POP_SIZE, DNA_SIZE*N_DIM)).place(Envo_SEX)
        
    
    # ti.root.dense(ti.jl, (POP_SIZE, DNA_SIZE*N_DIM)).place(Envo_EX, Envo_SEX)
    
    
    Envo_EDX  = ti.Vector.field(n= DNA_SIZE, dtype=ti.f32, shape=(POP_SIZE,N_DIM))
    
    Envo_DX  = ti.Vector.field(n= N_DIM, dtype=ti.f32, shape=(POP_SIZE))

    # Envo_Y   = ti.Vector.field(n= 1, dtype=ti.f32, shape=(POP_SIZE))
    
    # Envo_Best_Y = ti.Vector.field(n= 1, dtype=ti.f32, shape=(N_GENERATIONS))
    
    Envo_Y   = ti.field(dtype=ti.f32, shape=(POP_SIZE))
    
    Envo_Best_Y = ti.field(dtype=ti.f32, shape=(N_GENERATIONS))

    
    Envo_Best_Index = ti.Vector.field(n= 1, dtype=ti.f32, shape=N_GENERATIONS)
    
    
def main2():

    
    init_pop(0)
    
    decodex(0)
    
    fitness(0)
    
    best_fitness(0)
    
    # ti.loop_config(serialize=True)
    
    for i in range(1,N_GENERATIONS):
        
        if (i%2 == 1):
            
            select(i)
            
            decodex(i)
        
        else:
            
            select2(i)
            
            decodex2(i)
            
        
        
        fitness(i) 
        
        best_fitness(i)
        
        
import time

if __name__ == '__main__':
    
    main(2,[-3,-3],[3,3])
    
    a = time.time()
    
    main2()
    
    b = time.time()
    
    print(b-a)
    
    # print(Envo_Best_Y)
    
    a = time.time()
    
    main2()
    
    b = time.time()
    
    print(b-a)
    
    # print(Envo_Best_Y)