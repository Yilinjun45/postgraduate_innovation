# -*- coding: utf-8 -*-
"""
多层膜结构双峰吸收优化
使用遗传算法优化多层膜的材料选择和厚度
目标：在1-3μm和3-8μm波段实现双峰吸收
"""
import sys, os
import numpy as np
from collections import OrderedDict
import matplotlib.pyplot as plt
import importlib.util
import lumapi
from scipy.interpolate import interpolate
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False
'''
sys.path.append("C:\\Program Files\\Lumerical\\v202\\api\\python\\")
sys.path.append(os.path.dirname(__file__))
os.add_dll_directory("C:\\Program Files\\Lumerical\\v202\\api\\python\\")

# 加载Lumerical API
spe = importlib.util.spec_from_file_location('lumapi', 'C:\\Program Files\\Lumerical\\v202\\api\\python\\lumapi.py')
lumapi = importlib.util.module_from_spec(spe)
spe.loader.exec_module(lumapi)
'''
# 遗传算法参数
DNA_SIZE_LAYER = 10  # 每层编码长度（3位材料选择 + 7位厚度）
MAX_LAYERS = 10  # 最大层数
POP_SIZE = 20  # 种群大小
CROSSOVER_RATE = 0.8
MUTATION_RATE = 0.01
N_GENERATIONS = 20

# 材料列表
MATERIALS = ['SiO2 (Glass) - Palik', 'Si (Silicon) - Palik', 'Al2O3 - Palik',
             'Cr (Chromium) - Palik', 'Ag (Silver) - CRC', 'Au (Gold) - Palik',
             'Cu (Copper) - Palik', 'Al (Aluminium) - Palik']

# 厚度范围 (nm)
THICKNESS_MIN = 10
THICKNESS_MAX = 500

# 仿真波长范围
LAM_MIN = 0.4  # μm
LAM_MAX = 15  # μm
FP = 1000  # 频率点数

# 目标波段
BAND1 = [1, 3]  # 第一个吸收带 1-3 μm
BAND2 = [5, 8]  # 第二个吸收带 3-8 μm


def decode_individual(individual):
    """解码个体基因，返回层数、材料序列和厚度序列"""
    # 前4位编码层数（1-10层）
    n_layers = int(individual[:4].dot(2 ** np.arange(4)[::-1]) / 15 * 9) + 1

    materials = []
    thicknesses = []

    for i in range(n_layers):
        start_idx = 4 + i * DNA_SIZE_LAYER
        if start_idx + DNA_SIZE_LAYER > len(individual):
            break

        # 3位编码材料（8种材料）
        mat_code = individual[start_idx:start_idx + 3]
        mat_idx = mat_code.dot(2 ** np.arange(3)[::-1])
        materials.append(MATERIALS[mat_idx])

        # 7位编码厚度
        thick_code = individual[start_idx + 3:start_idx + DNA_SIZE_LAYER]
        thickness = thick_code.dot(2 ** np.arange(7)[::-1]) / 127 * (THICKNESS_MAX - THICKNESS_MIN) + THICKNESS_MIN
        thicknesses.append(thickness)

    return n_layers, materials, thicknesses


def build_multilayer_structure(n_layers, materials, thicknesses):
    """在FDTD中构建多层膜结构并运行仿真"""
    nm = 1e-9
    um = 1e-6

    # 仿真区域参数
    sim_size = 0.05 * um
    z_span = 20 * um

    try:
        fdtd = lumapi.FDTD()

        # 添加基底（可选）
        fdtd.addrect(name='Substrate', x=0, y=0, x_span=sim_size, y_span=sim_size,
                     z_min=-z_span / 2, z_max=0, material='Si (Silicon) - Palik')

        # 添加多层膜
        z_start = 0
        for i in range(n_layers):
            fdtd.addrect(name=f'Layer_{i + 1}', x=0, y=0,
                         x_span=sim_size, y_span=sim_size,
                         z_min=z_start, z_max=z_start + thicknesses[i] * nm,
                         material=materials[i])
            z_start += thicknesses[i] * nm

        # 添加FDTD仿真区域
        fdtd.addfdtd(dimension='3D', x=0, y=0, x_span=sim_size, y_span=sim_size,
                     z_min=-2 * um, z_max=z_start + 2 * um,
                     x_min_bc='periodic', y_min_bc='periodic',
                     mesh_accuracy=3)

        # 添加平面波光源
        fdtd.addplane(injection_axis='z', direction='backward',
                      x=0, y=0, x_span=sim_size, y_span=sim_size,
                      z=z_start + 1 * um,
                      wavelength_start=LAM_MIN * um, wavelength_stop=LAM_MAX * um)

        # 添加反射监视器
        props_R = OrderedDict([
            ("name", "R"),
            ("override global monitor settings", True),
            ("use wavelength spacing", True),
            ("x", 0.), ("y", 0), ("z", z_start + 0.5 * um),
            ("x span", sim_size), ("y span", sim_size),
            ("monitor type", "2D Z-Normal"),
            ("frequency points", FP)
        ])
        fdtd.addpower(properties=props_R)

        # 添加透射监视器
        props_T = OrderedDict([
            ("name", "T"),
            ("override global monitor settings", True),
            ("use wavelength spacing", True),
            ("x", 0.), ("y", 0), ("z", -1 * um),
            ("x span", sim_size), ("y span", sim_size),
            ("monitor type", "2D Z-Normal"),
            ("frequency points", FP)
        ])
        fdtd.addpower(properties=props_T)

        # 保存并运行仿真
        fdtd.save("multilayer_absorber.fsp")
        fdtd.run()

        # 获取结果
        R = fdtd.transmission('R')
        T = -fdtd.transmission('T')
        wavelength = (3e8/fdtd.getdata('R', 'f')) * 1e6  # 转换为μm

        # 计算吸收
        A = 1 - R - T

        fdtd.close()

        return wavelength, A

    except Exception as e:
        print(f"仿真错误: {e}")
        return None, None


def calculate_fitness(wavelength, absorption):
    """计算适应度函数，优化双峰吸收"""
    if wavelength is None or absorption is None:
        return 0

    # 找到两个波段的索引
    band1_idx = np.where((wavelength >= BAND1[0]) & (wavelength <= BAND1[1]))[0]
    band2_idx = np.where((wavelength >= BAND2[0]) & (wavelength <= BAND2[1]))[0]

    # 计算两个波段的平均吸收
    if len(band1_idx) > 0 and len(band2_idx) > 0:
        avg_abs_band1 = np.mean(absorption[band1_idx])
        avg_abs_band2 = np.mean(absorption[band2_idx])

        # 计算两个波段外的平均吸收（希望最小化）
        other_idx = np.where((wavelength < BAND1[0]) |
                             ((wavelength > BAND1[1]) & (wavelength < BAND2[0])) |
                             (wavelength > BAND2[1]))[0]
        avg_abs_other = np.mean(absorption[other_idx]) if len(other_idx) > 0 else 0

        # 适应度函数：最大化目标波段吸收，最小化其他波段吸收
        fitness = (avg_abs_band1 + avg_abs_band2) - 0.5 * avg_abs_other

        return max(0, fitness)
    else:
        return 0


def evaluate_population(pop):
    """评估种群中所有个体的适应度"""
    fitness = np.zeros(POP_SIZE)

    for i in range(POP_SIZE):
        n_layers, materials, thicknesses = decode_individual(pop[i])
        wavelength, absorption = build_multilayer_structure(n_layers, materials, thicknesses)
        fitness[i] = calculate_fitness(wavelength, absorption)

        print(f"个体 {i + 1}/{POP_SIZE}: 层数={n_layers}, 适应度={fitness[i]:.4f}")

    return fitness


def crossover(parent1, parent2):
    """交叉操作"""
    if np.random.rand() < CROSSOVER_RATE:
        cross_point = np.random.randint(1, len(parent1))
        child1 = np.concatenate([parent1[:cross_point], parent2[cross_point:]])
        child2 = np.concatenate([parent2[:cross_point], parent1[cross_point:]])
        return child1, child2
    else:
        return parent1.copy(), parent2.copy()


def mutate(individual):
    """变异操作"""
    for i in range(len(individual)):
        if np.random.rand() < MUTATION_RATE:
            individual[i] = 1 - individual[i]
    return individual


def select(pop, fitness):
    """选择操作"""
    # 轮盘赌选择
    fitness_positive = fitness - np.min(fitness) + 1e-6
    probs = fitness_positive / np.sum(fitness_positive)
    idx = np.random.choice(np.arange(POP_SIZE), size=POP_SIZE, replace=True, p=probs)
    return pop[idx]


def plot_results(best_individual):
    """绘制最佳个体的光谱响应"""
    n_layers, materials, thicknesses = decode_individual(best_individual)
    wavelength, absorption = build_multilayer_structure(n_layers, materials, thicknesses)

    if wavelength is not None and absorption is not None:
        plt.figure(figsize=(10, 6))
        plt.plot(wavelength, absorption, 'b-', linewidth=2)
        plt.axvspan(BAND1[0], BAND1[1], alpha=0.2, color='red', label='目标波段1 (1-3 μm)')
        plt.axvspan(BAND2[0], BAND2[1], alpha=0.2, color='green', label='目标波段2 (3-8 μm)')
        plt.xlabel('波长 (μm)')
        plt.ylabel('吸收率')
        plt.title('优化后的多层膜吸收光谱')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xlim(LAM_MIN, LAM_MAX)
        plt.ylim(0, 1)

        # 打印结构信息
        print("\n最优结构:")
        print(f"层数: {n_layers}")
        for i in range(n_layers):
            print(f"第{i + 1}层: {materials[i]}, 厚度 = {thicknesses[i]:.1f} nm")

        plt.show()


def main():
    """主函数"""
    # 初始化种群
    DNA_LENGTH = 4 + MAX_LAYERS * DNA_SIZE_LAYER  # 4位层数 + 每层的编码
    pop = np.random.randint(2, size=(POP_SIZE, DNA_LENGTH))

    best_fitness_history = []
    avg_fitness_history = []

    for generation in range(N_GENERATIONS):
        print(f"\n第 {generation + 1}/{N_GENERATIONS} 代")

        # 评估适应度
        fitness = evaluate_population(pop)

        # 记录统计信息
        best_idx = np.argmax(fitness)
        best_fitness = fitness[best_idx]
        avg_fitness = np.mean(fitness)

        best_fitness_history.append(best_fitness)
        avg_fitness_history.append(avg_fitness)

        print(f"最佳适应度: {best_fitness:.4f}, 平均适应度: {avg_fitness:.4f}")

        # 保存最佳个体
        if generation == 0 or best_fitness > max(best_fitness_history[:-1]):
            best_individual = pop[best_idx].copy()

        # 选择
        pop = select(pop, fitness)

        # 交叉和变异
        new_pop = []
        for i in range(0, POP_SIZE, 2):
            child1, child2 = crossover(pop[i], pop[i + 1])
            new_pop.extend([mutate(child1), mutate(child2)])
        pop = np.array(new_pop[:POP_SIZE])

    # 绘制进化曲线
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, N_GENERATIONS + 1), best_fitness_history, 'r-', label='最佳适应度')
    plt.plot(range(1, N_GENERATIONS + 1), avg_fitness_history, 'b-', label='平均适应度')
    plt.xlabel('代数')
    plt.ylabel('适应度')
    plt.title('遗传算法进化曲线')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    # 显示最佳结果
    plot_results(best_individual)


if __name__ == "__main__":
    main()