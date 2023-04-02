import taichi as ti
import taichi.math as tm

ti.init(arch=ti.cuda, default_fp=ti.f32, debug=True, kernel_profiler=True,device_memory_fraction=0.9)


@ti.kernel
def init(g: int):
    for i, j in ti.ndrange(POP_SIZE, N_DIM):
        y = ti.random() * (HIGH_BOUND[j] - LOW_BOUND[j]) + LOW_BOUND[j]

        Envo_X[i][j] = y

        vi = ti.random() * (VMAX - VMIN) + VMIN

        Envo_V[i][j] = vi

        Envo_P_best_X[i][j] = y


@ti.func
def F(g: int,
      n: int):
    x = Envo_X[n]

    y = 3 * (1 - x[0]) ** 2 * tm.exp(-(x[0] ** 2) - (x[1] + 1) ** 2) - 10 * (x[0] / 5 - x[0] ** 3 - x[1] ** 5) * tm.exp(
        -x[0] ** 2 - x[1] ** 2) - 1 / 3 ** tm.exp(-(x[0] + 1) ** 2 - x[1] ** 2)

    return -y


@ti.kernel
def fitness_func(g: int):
    for i in range(POP_SIZE):
        Envo_Y[i] = F(g, i)


@ti.kernel
def pbest(g: int):  # 求出每个粒子的历史最优位置

    ti.loop_config(serialize=False)

    for i in range(POP_SIZE):

        if g == 0:

            Envo_P_best_Y[i] = Envo_Y[i]

        else:

            if Envo_Y[i] > Envo_P_best_Y[i]:

                Envo_P_best_Y[i] = Envo_Y[i]

                ti.loop_config(serialize=False)

                for j in range(N_DIM):
                    Envo_P_best_X[i][j] = Envo_X[i][j]


@ti.kernel
def gbest(g: int):  # 计算群体历史最优位置

    best_index = -1

    best_y = -1.0

    ti.loop_config(serialize=True)

    for i in range(POP_SIZE):

        if i == 0:

            best_index = i

            best_y = Envo_P_best_Y[best_index]

        else:

            if Envo_P_best_Y[i] > best_y:
                best_index = i

                best_y = Envo_P_best_Y[best_index]

    Envo_Best_Y[g] = best_y

    if best_y > Envo_G_best_Y[0]:

        Envo_G_best_Y[0] = best_y

        ti.loop_config(serialize=False)

        for j in range(N_DIM):
            Envo_G_best_X[0][j] = Envo_P_best_X[best_index][j]


@ti.kernel
def update(g: int):
    for i, j in ti.ndrange(POP_SIZE, N_DIM):

        W = WMAX - (WMAX - WMIN) * (g / N_GENERATIONS)

        r1 = ti.random()

        r2 = ti.random()

        Envo_V[i][j] = W * Envo_V[i][j] + C1 * r1 * (Envo_P_best_X[i][j] - Envo_X[i][j]) + C2 * r2 * (
                    Envo_G_best_X[0][j] - Envo_X[i][j])

        if not (VMIN < Envo_V[i][j] < VMAX):
            Envo_V[i][j] = ti.random() * (VMAX - VMIN) + VMIN

        Envo_X[i][j] = Envo_X[i][j] + Envo_V[i][j]

        if not (LOW_BOUND[j] < Envo_X[i][j] < HIGH_BOUND[j]):
            Envo_X[i][j] = ti.random() * (HIGH_BOUND[j] - LOW_BOUND[j]) + LOW_BOUND[j]


def main(n_dim, low_bound, high_bound, vmin, vmax, wmin, wmax, c1, c2):
    global Envo_X

    global Envo_V

    global Envo_Y

    global Envo_P_best_X

    global Envo_P_best_Y

    global Envo_G_best_X

    global Envo_G_best_Y

    global Envo_Best_Y

    global POP_SIZE

    global N_GENERATIONS

    global N_DIM

    global POP_SIZE

    global N_GENERATIONS

    global HIGH_BOUND

    global LOW_BOUND

    global VMIN

    global VMAX

    global WMIN

    global WMAX

    global W

    global C1

    global C2

    POP_SIZE = 500

    N_GENERATIONS = 1000

    N_DIM = n_dim

    VMIN = vmin

    VMAX = vmax

    WMIN = wmin

    WMAX = wmax

    C1 = c1

    C2 = c2

    HIGH_BOUND = ti.field(ti.f32, shape=N_DIM)

    for i in range(len(high_bound)):
        HIGH_BOUND[i] = high_bound[i]

    LOW_BOUND = ti.field(ti.f32, shape=N_DIM)

    for i in range(len(low_bound)):
        LOW_BOUND[i] = low_bound[i]

    Envo_V = ti.Vector.field(N_DIM, dtype=ti.f32)

    Envo_X = ti.Vector.field(N_DIM, dtype=ti.f32)

    Envo_P_best_X = ti.Vector.field(N_DIM, dtype=ti.f32)

    ti.root.dense(ti.i,POP_SIZE).place(Envo_V,Envo_X,Envo_P_best_X)

    Envo_Y = ti.field(dtype=ti.f32)

    Envo_P_best_Y = ti.field(dtype=ti.f32)

    ti.root.dense(ti.i, POP_SIZE).place(Envo_P_best_Y,Envo_Y)

    Envo_G_best_X = ti.Vector.field(N_DIM, dtype=ti.f32, shape=(1))

    Envo_G_best_Y = ti.field(dtype=ti.f32)

    ti.root.dense(ti.i, 1).place(Envo_G_best_Y)

    Envo_Best_Y = ti.field(dtype=ti.f32)

    ti.root.dense(ti.i, N_GENERATIONS).place(Envo_Best_Y )


def PSO():
    init(0)

    fitness_func(0)

    pbest(0)

    gbest(0)

    for i in range(1, N_GENERATIONS):
        update(i)

        fitness_func(i)

        pbest(i)

        gbest(i)


import time

if __name__ == '__main__':
    main(2, [-3, -3], [3, 3], -1, 1, 0, 1, 1, 1)

    a = time.time()

    PSO()

    b = time.time()

    print(b - a)

    # print(Envo_Best_Y)
    sum=0
    for i in range(1000):

      a = time.time()

      PSO()

      b = time.time()

      c = b-a

      sum = sum+c


    pinjun=sum/1000
    print(pinjun)

    # #

ti.profiler.print_kernel_profiler_info()
