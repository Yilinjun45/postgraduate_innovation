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
from scipy.interpolate import interp1d
from scipy.integrate import quad

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# 遗传算法参数（优化后的参数）
DNA_SIZE_LAYER = 10  # 每层编码长度（3位材料选择 + 7位厚度）
MAX_LAYERS = 10  # 最大层数
POP_SIZE = 10  # 种群大小（从20增加到50）
CROSSOVER_RATE = 0.8
MUTATION_RATE = 0.02  # 变异率（从0.01增加到0.02）
N_GENERATIONS = 10  # 进化代数（从10增加到50）

# 材料列表（保持原有材料库）
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

# 目标波段（双峰）
BAND1 = [1.5, 2.5]  # 第一个吸收带 1-3 μm
BAND2 = [5, 8]  # 第二个吸收带 5-8 μm

# 温度设置（用于黑体辐射计算）
TEMPERATURE = 873  # K

# 适应度函数权重设置
WEIGHT_BAND1 = 0.3  # 第一个峰的权重
WEIGHT_BAND2 = 0.7  # 第二个峰的权重
PENALTY_FACTOR = 0.5  # 波段外吸收的惩罚因子


def blackbody_radiation(wavelength, T):
    """
    计算黑体辐射强度
    wavelength: 波长 (μm)
    T: 温度 (K)
    返回: 辐射强度 (相对单位)
    """
    # 物理常数
    h = 6.62607015e-34  # 普朗克常数 (J⋅s)
    c = 2.99792458e8  # 光速 (m/s)
    k = 1.380649e-23  # 玻尔兹曼常数 (J/K)

    # 转换波长单位从μm到m
    wavelength_m = wavelength * 1e-6

    # 避免数值溢出
    try:
        # 普朗克黑体辐射公式
        numerator = 2 * h * c ** 2 / wavelength_m ** 5
        exponent = h * c / (wavelength_m * k * T)

        # 防止溢出
        if exponent > 700:
            return 0

        denominator = np.exp(exponent) - 1
        return numerator / denominator
    except:
        return 0


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

        # 添加基底
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
        f = fdtd.getdata('R', 'f')
        wavelength = (3e8 / f) * 1e6  # 转换为μm

        # 确保数据是数组格式
        R = np.array(R).flatten()
        T = np.array(T).flatten()
        wavelength = np.array(wavelength).flatten()

        # 检查数据长度
        print(f"  波长点数: {len(wavelength)}, R点数: {len(R)}, T点数: {len(T)}")

        # 确保所有数组长度相同
        min_len = min(len(wavelength), len(R), len(T))
        wavelength = wavelength[:min_len]
        R = R[:min_len]
        T = T[:min_len]

        # 计算吸收
        A = 1 - R - T

        fdtd.close()

        # 数据验证
        if len(wavelength) < 10:
            print("  警告: 数据点太少")
            return None, None

        return wavelength, A

    except Exception as e:
        print(f"  仿真错误: {e}")
        return None, None


def calculate_fitness_simple(wavelength, absorption):
    """简化的适应度函数，用于测试"""
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

        # 使用权重计算适应度
        fitness = (WEIGHT_BAND1 * avg_abs_band1 + WEIGHT_BAND2 * avg_abs_band2) - PENALTY_FACTOR * avg_abs_other

        return max(0, fitness)
    else:
        return 0


def calculate_fitness_double_peak(wavelength, absorption):
    """
    计算双峰适应度函数，基于论文中的FF公式
    包含可调整的权重参数
    """
    if wavelength is None or absorption is None:
        return 0

    # 数据验证
    if len(wavelength) != len(absorption):
        print(f"  警告: 波长和吸收率数组长度不匹配: {len(wavelength)} vs {len(absorption)}")
        return 0

    if len(wavelength) < 10:
        print(f"  警告: 数据点太少: {len(wavelength)}")
        return 0

    try:
        # 确保波长是递增的
        if not np.all(np.diff(wavelength) > 0):
            # 排序
            sort_idx = np.argsort(wavelength)
            wavelength = wavelength[sort_idx]
            absorption = absorption[sort_idx]

        # 创建插值函数
        absorb_interp = interp1d(wavelength, absorption,
                                 bounds_error=False, fill_value=0, kind='linear')

        # 使用简化的积分计算
        # 在目标波段内采样
        n_samples = 50

        # 第一个波段
        ff1 = 0
        if BAND1[0] >= wavelength.min() and BAND1[1] <= wavelength.max():
            lam1 = np.linspace(BAND1[0], BAND1[1], n_samples)
            abs1 = absorb_interp(lam1)
            bb1 = np.array([blackbody_radiation(l, TEMPERATURE) for l in lam1])

            if np.sum(bb1) > 0:
                ff1 = np.sum(abs1 * bb1) / np.sum(bb1)
            else:
                ff1 = np.mean(abs1)

        # 第二个波段
        ff2 = 0
        if BAND2[0] >= wavelength.min() and BAND2[1] <= wavelength.max():
            lam2 = np.linspace(BAND2[0], BAND2[1], n_samples)
            abs2 = absorb_interp(lam2)
            bb2 = np.array([blackbody_radiation(l, TEMPERATURE) for l in lam2])

            if np.sum(bb2) > 0:
                ff2 = np.sum(abs2 * bb2) / np.sum(bb2)
            else:
                ff2 = np.mean(abs2)

        # 波段外的惩罚
        other_idx = np.where((wavelength < BAND1[0]) |
                             ((wavelength > BAND1[1]) & (wavelength < BAND2[0])) |
                             (wavelength > BAND2[1]))[0]

        if len(other_idx) > 0:
            penalty = np.mean(absorption[other_idx]) * PENALTY_FACTOR
        else:
            penalty = 0

        # 使用可调权重计算总适应度
        fitness = WEIGHT_BAND1 * ff1 + WEIGHT_BAND2 * ff2 - penalty

        # 打印详细信息（可选）
        if fitness > 0.5:  # 只打印高适应度的详细信息
            print(f"    FF1={ff1:.3f}, FF2={ff2:.3f}, Penalty={penalty:.3f}, Total={fitness:.3f}")

        return max(0, fitness)

    except Exception as e:
        print(f"  适应度计算错误: {e}")
        # 回退到简单适应度函数
        return calculate_fitness_simple(wavelength, absorption)


def evaluate_population(pop):
    """评估种群中所有个体的适应度"""
    fitness = np.zeros(POP_SIZE)

    for i in range(POP_SIZE):
        n_layers, materials, thicknesses = decode_individual(pop[i])
        print(f"\n个体 {i + 1}/{POP_SIZE}: 层数={n_layers}")

        wavelength, absorption = build_multilayer_structure(n_layers, materials, thicknesses)

        # 使用适应度函数
        fitness[i] = calculate_fitness_double_peak(wavelength, absorption)

        print(f"  适应度={fitness[i]:.4f}")

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
    """选择操作 - 使用锦标赛选择"""
    selected = []
    tournament_size = 3

    for _ in range(POP_SIZE):
        # 随机选择tournament_size个个体
        tournament_idx = np.random.choice(POP_SIZE, tournament_size, replace=False)
        tournament_fitness = fitness[tournament_idx]

        # 选择最优的个体
        winner_idx = tournament_idx[np.argmax(tournament_fitness)]
        selected.append(pop[winner_idx])

    return np.array(selected)


def plot_results(best_individual):
    """绘制最佳个体的光谱响应"""
    n_layers, materials, thicknesses = decode_individual(best_individual)
    wavelength, absorption = build_multilayer_structure(n_layers, materials, thicknesses)

    if wavelength is not None and absorption is not None:
        # 计算黑体辐射谱
        bb_spectrum = [blackbody_radiation(lam, TEMPERATURE) for lam in wavelength]
        bb_spectrum = np.array(bb_spectrum) / np.max(bb_spectrum) if np.max(bb_spectrum) > 0 else bb_spectrum

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

        # 上图：吸收光谱
        ax1.plot(wavelength, absorption, 'b-', linewidth=2, label='吸收率')
        ax1.axvspan(BAND1[0], BAND1[1], alpha=0.2, color='red', label=f'目标波段1 (权重={WEIGHT_BAND1})')
        ax1.axvspan(BAND2[0], BAND2[1], alpha=0.2, color='green', label=f'目标波段2 (权重={WEIGHT_BAND2})')
        ax1.set_xlabel('波长 (μm)')
        ax1.set_ylabel('吸收率')
        ax1.set_title('优化后的多层膜吸收光谱')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(LAM_MIN, LAM_MAX)
        ax1.set_ylim(0, 1)

        # 下图：加权吸收（吸收率×黑体辐射）
        weighted_absorption = absorption * bb_spectrum
        ax2.plot(wavelength, weighted_absorption, 'r-', linewidth=2, label='吸收率×黑体辐射')
        ax2.plot(wavelength, bb_spectrum, 'k--', alpha=0.5, label=f'黑体辐射@{TEMPERATURE}K (归一化)')
        ax2.axvspan(BAND1[0], BAND1[1], alpha=0.2, color='red')
        ax2.axvspan(BAND2[0], BAND2[1], alpha=0.2, color='green')
        ax2.set_xlabel('波长 (μm)')
        ax2.set_ylabel('归一化强度')
        ax2.set_title('黑体辐射加权吸收光谱')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(LAM_MIN, LAM_MAX)

        plt.tight_layout()

        # 打印结构信息
        print("\n" + "=" * 60)
        print("最优结构详细信息:")
        print("=" * 60)
        print(f"层数: {n_layers}")
        print(f"温度: {TEMPERATURE} K")
        print(f"权重设置: 波段1={WEIGHT_BAND1}, 波段2={WEIGHT_BAND2}, 惩罚因子={PENALTY_FACTOR}")
        print("-" * 60)
        print("层结构:")
        total_thickness = 0
        for i in range(n_layers):
            print(f"  第{i + 1}层: {materials[i]:<25} 厚度 = {thicknesses[i]:6.1f} nm")
            total_thickness += thicknesses[i]
        print(f"  总厚度: {total_thickness:.1f} nm")
        print("=" * 60)

        # 计算并打印性能指标
        band1_idx = np.where((wavelength >= BAND1[0]) & (wavelength <= BAND1[1]))[0]
        band2_idx = np.where((wavelength >= BAND2[0]) & (wavelength <= BAND2[1]))[0]

        if len(band1_idx) > 0:
            avg_abs_band1 = np.mean(absorption[band1_idx])
            print(f"波段1 (1-3 μm) 平均吸收率: {avg_abs_band1:.3f}")

        if len(band2_idx) > 0:
            avg_abs_band2 = np.mean(absorption[band2_idx])
            print(f"波段2 (5-8 μm) 平均吸收率: {avg_abs_band2:.3f}")

        plt.show()


def main():
    """主函数"""
    print("=" * 60)
    print("多层膜双峰吸收优化程序")
    print("=" * 60)
    print(f"遗传算法参数:")
    print(f"  种群大小: {POP_SIZE}")
    print(f"  进化代数: {N_GENERATIONS}")
    print(f"  交叉率: {CROSSOVER_RATE}")
    print(f"  变异率: {MUTATION_RATE}")
    print(f"目标波段:")
    print(f"  波段1: {BAND1[0]}-{BAND1[1]} μm (权重={WEIGHT_BAND1})")
    print(f"  波段2: {BAND2[0]}-{BAND2[1]} μm (权重={WEIGHT_BAND2})")
    print(f"  惩罚因子: {PENALTY_FACTOR}")
    print("=" * 60)

    # 初始化种群
    DNA_LENGTH = 4 + MAX_LAYERS * DNA_SIZE_LAYER
    pop = np.random.randint(2, size=(POP_SIZE, DNA_LENGTH))

    best_fitness_history = []
    avg_fitness_history = []
    best_individual_overall = None
    best_fitness_overall = -float('inf')

    for generation in range(N_GENERATIONS):
        print(f"\n{'=' * 50}")
        print(f"第 {generation + 1}/{N_GENERATIONS} 代")
        print(f"{'=' * 50}")

        # 评估适应度
        fitness = evaluate_population(pop)

        # 记录统计信息
        valid_fitness = fitness[fitness > 0]  # 只统计有效的适应度
        if len(valid_fitness) > 0:
            best_idx = np.argmax(fitness)
            best_fitness = fitness[best_idx]
            avg_fitness = np.mean(valid_fitness)
        else:
            print("警告: 本代所有个体适应度都为0")
            best_fitness = 0
            avg_fitness = 0
            best_idx = 0

        best_fitness_history.append(best_fitness)
        avg_fitness_history.append(avg_fitness)

        print(f"\n本代统计: 最佳适应度={best_fitness:.4f}, 平均适应度={avg_fitness:.4f}")
        print(f"有效个体数: {len(valid_fitness)}/{POP_SIZE}")

        # 保存全局最佳个体
        if best_fitness > best_fitness_overall:
            best_fitness_overall = best_fitness
            best_individual_overall = pop[best_idx].copy()
            print(f"*** 发现新的全局最优! 适应度={best_fitness:.4f} ***")

        # 精英保留策略
        elite_size = max(2, int(POP_SIZE * 0.1))  # 保留10%的精英
        if len(valid_fitness) >= elite_size:
            elite_idx = np.argsort(fitness)[-elite_size:]
            elite_individuals = pop[elite_idx]
        else:
            elite_individuals = pop[:elite_size]

        # 选择
        pop = select(pop, fitness)

        # 交叉和变异
        new_pop = []
        for i in range(0, POP_SIZE - elite_size, 2):
            if i + 1 < POP_SIZE - elite_size:
                child1, child2 = crossover(pop[i], pop[i + 1])
                new_pop.extend([mutate(child1), mutate(child2)])
            else:
                new_pop.append(mutate(pop[i].copy()))

        # 添加精英个体
        new_pop.extend(elite_individuals)
        pop = np.array(new_pop[:POP_SIZE])

        # 定期报告进展
        if (generation + 1) % 10 == 0:
            print(f"\n进化进展报告:")
            print(f"  当前最佳适应度: {best_fitness_overall:.4f}")
            print(f"  适应度提升: {(best_fitness_overall / (best_fitness_history[0] + 1e-6) - 1) * 100:.1f}%")

    # 绘制进化曲线
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, N_GENERATIONS + 1), best_fitness_history, 'r-', linewidth=2, label='最佳适应度')
    plt.plot(range(1, N_GENERATIONS + 1), avg_fitness_history, 'b-', linewidth=2, label='平均适应度')
    plt.xlabel('代数')
    plt.ylabel('适应度')
    plt.title('遗传算法进化曲线')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 添加文本注释
    plt.text(0.02, 0.98, f'最终最佳适应度: {best_fitness_overall:.4f}',
             transform=plt.gca().transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.show()

    # 显示最佳结果
    if best_individual_overall is not None:
        print(f"\n最终全局最优适应度: {best_fitness_overall:.4f}")
        plot_results(best_individual_overall)
    else:
        print("未找到有效的最佳个体")


if __name__ == "__main__":
    main()