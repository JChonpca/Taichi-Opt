# -*- coding: utf-8 -*-
"""
Created on Sat Feb 25 15:37:14 2023

@author: JChonpca_Huang
"""

import taichi as ti
import taichi.math as tm
import os
os.environ["KMP_DUPLICATE_LIB_OK"]  =  "TRUE"

ti.init(arch=ti.cuda, default_fp=ti.f64 ,debug=True, kernel_profiler=True)



@ti.func
def F(g: int,
      n: int):
    
    x = Envo_DX[g,n]
        
    y = 3*(1-x[0])**2*tm.exp(-(x[0]**2)-(x[1]+1)**2)-10*(x[0]/5-x[0]**3-x[1]**5)*tm.exp(-x[0]**2-x[1]**2)-1/3**tm.exp(-(x[0]+1)**2-x[1]**2)
        
    return -y




@ti.kernel
def init_pop(g: int):
    
    # init the pop in G(p)
    
    # ti.loop_config(parallelize= AA, block_dim= BB)
    
    for i,j in ti.ndrange(POP_SIZE,DNA_SIZE*N_DIM):
        
        init_x = int(ti.random()*2)
        
        Envo_EX[g,i][j]  = init_x
        
        Envo_SEX[g,i][j] = init_x        
    
        Envo_CEX[g,i][j] = init_x    

        Envo_MEX[g,i][j] = init_x



@ti.kernel
def select(g: int):
    
    # ti.loop_config(parallelize= AA, block_dim= BB)
    
    for i in range(POP_SIZE):
        
        best_index = -1
        
        best_y = -1.0
        
        for j in range(TOURN_SIZE):
            
            if j == 0:
                
                best_index = int(ti.random()*POP_SIZE)
                
                best_y = Envo_Y[g-1,best_index][0]
            
            else:
                
                tmp_best_index = int(ti.random()*POP_SIZE)
                
                tmp_best_y = Envo_Y[g-1,tmp_best_index][0]
                
                if tmp_best_y > best_y:
                    
                    best_index = tmp_best_index
                
                                
        Envo_SEX[g,i] = Envo_EX[g-1,best_index]            
                

@ti.func
def exchange(n1: int, 
             n2: int,
             g: int,
             n: int):
    
    seg_1 = Envo_SEX[g,2*n]
    
    seg_2 = Envo_SEX[g,2*n+1]

    # ti.loop_config(parallelize= AA, block_dim= BB)
    
    for i in range(DNA_SIZE*N_DIM):
        
        if 0 <= i < n1:
            
            Envo_CEX[g,2*n][i] = seg_1[i]
            
            Envo_CEX[g,2*n+1][i] = seg_2[i]
                
        elif n1 <= i < n2:
            
            Envo_CEX[g,2*n][i] = seg_2[i]   
            
            Envo_CEX[g,2*n+1][i] = seg_1[i]        
                
        elif i >= n2:
            
            Envo_CEX[g,2*n][i] = seg_1[i]
            
            Envo_CEX[g,2*n+1][i] = seg_2[i]           
            

@ti.kernel
def crossover(g: int):
    
    # ti.loop_config(parallelize= AA, block_dim= BB)
    
    for i in range(int(POP_SIZE/2)):
        
        if ti.random() < CROSSOVER_RATE:
            
            n1 = ti.random()
            
            n2 = ti.random()
            
            if n1 > n2:
                
                n1, n2 = n2, n1
            
            n1 = int(n1*DNA_SIZE*N_DIM)
            
            n2 = int(n2*DNA_SIZE*N_DIM)
            
            exchange(n1, n2, g, i)

@ti.kernel
def mutation(g: int):
    
    # ti.loop_config(parallelize= AA, block_dim= BB)
    
    for i,j in ti.ndrange(POP_SIZE,DNA_SIZE*N_DIM):
        
        Envo_MEX[g,i][j] = Envo_CEX[g,i][j]
        
        if ti.random() < MUTATION_RATE:
            
            Envo_MEX[g,i][j] = int(ti.random()*2)
            

# @ti.func
# def best_fitness_stay(g: int):
    
#     if Envo_Best_Y[g-1][0] > Envo_Best_Y[g][0]:
        
#         index = int(ti.random()*POP_SIZE)
        
#         Envo_EX[g,index] = Envo_EX[g-1,Envo_Best_Index[g-1][0]]
        
#         Envo_DX[g,index] = Envo_DX[g-1,Envo_Best_Index[g-1][0]]
                
#         Envo_Y[g,index]  = Envo_Y[g-1,Envo_Best_Index[g-1][0]]
        
#         Envo_Best_Y[g][0] = Envo_Best_Y[g-1][0]
        
#         Envo_Best_Index[g][0] = index
        

@ti.kernel
def copy(g: int):
    
    # ti.loop_config(parallelize= AA, block_dim= BB)
        
    for i,j in ti.ndrange(POP_SIZE,DNA_SIZE*N_DIM):
                
        Envo_EX[g,i][j] = Envo_MEX[g,i][j]
    

@ti.func
def extractx(g: int):
    
    # ti.loop_config(parallelize= AA, block_dim= BB)
    
    for i,j in ti.ndrange(POP_SIZE,DNA_SIZE*N_DIM):
        
        dim_count = int(j // N_DIM)
        
        dim_index = int(j % N_DIM)
        
        Envo_EDX[g,i,dim_index][dim_count] = Envo_EX[g,i][j]
        

@ti.func
def transx(g: int,
           n: int,
           m: int):
    
    num = 0.0
    
    # ti.loop_config(parallelize= AA, block_dim= BB)
    
    for i in range(DNA_SIZE):
        
        num += (Envo_EDX[g,n,m][i])*(2**(DNA_SIZE-i))
    
    num = num/float(2**DNA_SIZE-1)
    
    num = num*(HIGH_BOUND[m]-LOW_BOUND[m]) + LOW_BOUND[m]
    
    Envo_DX[g,n][m] = num
    
@ti.kernel
def decodex(g: int):
    
    extractx(g)
    
    # ti.loop_config(parallelize= AA, block_dim= BB)
    
    for i,j in ti.ndrange(POP_SIZE,N_DIM):
        
        transx(g, i, j)

@ti.kernel
def fitness(g: int):
    
    # ti.loop_config(parallelize= AA, block_dim= BB)
    
    for i in range(POP_SIZE):
        
        Envo_Y[g,i][0] = F(g,i)
        
@ti.kernel
def best_fitness(g: int):
    
    best_index = -1
    
    best_y = -1.0
    
    ti.loop_config(serialize=True)
    
    for i in range(POP_SIZE):
        
        if i == 0:
            
            best_index = i
            
            best_y = Envo_Y[g,best_index][0]
        
        else:
            
            if Envo_Y[g,i][0] > best_y:
                
                best_index = i
                
                best_y = Envo_Y[g,best_index][0]
                
    
    Envo_Best_Y[g][0] = best_y
    
    Envo_Best_Index[g][0] = best_index
    

def GA_RUN():
        
    # ti.block_local(N_DIM)
    
    # ti.block_local(DNA_SIZE)
    
    # ti.block_local(POP_SIZE)
    
    # ti.block_local(N_GENERATIONS)
    
    # ti.block_local(TOURN_SIZE)
    
    # ti.block_local(CROSSOVER_RATE)
    
    # ti.block_local(MUTATION_RATE)
    
    # ti.block_local(HIGH_BOUND)
    
    # ti.block_local(LOW_BOUND)

    
    # ti.block_local(Envo_EX)
    
    # ti.block_local(Envo_DX)
    
    # ti.block_local(Envo_EDX)
    
    # ti.block_local(Envo_SEX)
    
    # ti.block_local(Envo_CEX)
    
    # ti.block_local(Envo_MEX)
    
    # ti.block_local(Envo_Y)
    
    # ti.block_local(Envo_Best_Y)
    
    # ti.block_local(Envo_Best_Index)
    
    
    
    init_pop(0)
    
    decodex(0)
    
    fitness(0)
    
    best_fitness(0)
    
    # ti.loop_config(serialize=True)
    
    for i in range(1,N_GENERATIONS):
        
        select(i)
        
        crossover(i)
        
        mutation(i)
        
        copy(i)
        
        decodex(i)
        
        fitness(i)
        
        best_fitness(i)
        
        # best_fitness_stay(i)
        

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
    global Envo_CEX
    global Envo_MEX
    global Envo_Y
    global Envo_Best_Y
    global Envo_Best_Index
    
    
    N_DIM = ti.field(ti.i32, shape=())

    DNA_SIZE = ti.field(ti.i32, shape=())

    POP_SIZE = ti.field(ti.i32, shape=())

    N_GENERATIONS = ti.field(ti.i32, shape=())
    
    TOURN_SIZE = ti.field(ti.i32, shape=())

    CROSSOVER_RATE = ti.field(ti.f32, shape=())

    MUTATION_RATE = ti.field(ti.f32, shape=())

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
        
    
    Envo_EX  = ti.Vector.field(n= DNA_SIZE*N_DIM, dtype=ti.f32, shape=(N_GENERATIONS, POP_SIZE))
    
    Envo_SEX = ti.Vector.field(n= DNA_SIZE*N_DIM, dtype=ti.i32, shape=(N_GENERATIONS, POP_SIZE))
    
    Envo_CEX = ti.Vector.field(n= DNA_SIZE*N_DIM, dtype=ti.i32, shape=(N_GENERATIONS, POP_SIZE))
    
    Envo_MEX = ti.Vector.field(n= DNA_SIZE*N_DIM, dtype=ti.i32, shape=(N_GENERATIONS, POP_SIZE))
        
    Envo_EDX  = ti.Vector.field(n= DNA_SIZE, dtype=ti.f32, shape=(N_GENERATIONS,POP_SIZE,N_DIM))

    Envo_DX  = ti.Vector.field(n= N_DIM, dtype=ti.f32, shape=(N_GENERATIONS,POP_SIZE))

    Envo_Y   = ti.Vector.field(n= 1, dtype=ti.f32, shape=(N_GENERATIONS,POP_SIZE))
    
    Envo_Best_Y = ti.Vector.field(n= 1, dtype=ti.f32, shape=N_GENERATIONS)
    
    Envo_Best_Index = ti.Vector.field(n= 1, dtype=ti.f32, shape=N_GENERATIONS)
    
    GA_RUN()
    
import time

if __name__ == '__main__':
    
    a = time.time()
    
    main(2,[-3,-3],[3,3])
    
    b = time.time()