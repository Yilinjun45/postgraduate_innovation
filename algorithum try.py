#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
多层光学薄膜逆向优化系统
用于设计具有双峰吸收特性（可见光+中红外）的多层膜结构
材料选择：Si, Al2O3, Cr, Ag, Au
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import os
import sys

# 导入lumopt模块
from lumopt.utilities.wavelengths import Wavelengths
from lumopt.figures_of_merit.modematch import ModeMatch
from lumopt.optimizers.generic_optimizers import ScipyOptimizers
from lumopt.optimization import Optimization
from lumopt.geometries.polygon import FunctionDefinedPolygon
from lumopt.utilities.materials import Material


# 材料光学常数数据（简化版本，实际使用时应从RefractiveIndex.info导入）
class MaterialDatabase:
    """材料光学常数数据库"""

    @staticmethod
    def get_si_nk():
        """硅的折射率数据 (Palik)"""
        wavelengths = np.array(
            [400, 500, 600, 700, 800, 900, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000])  # nm
        n_values = np.array(
            [5.57, 4.29, 3.95, 3.75, 3.68, 3.64, 3.62, 3.53, 3.51, 3.50, 3.49, 3.48, 3.48, 3.47, 3.47, 3.46])
        k_values = np.array(
            [0.39, 0.073, 0.025, 0.016, 0.008, 0.005, 0.004, 0.001, 0.0008, 0.0006, 0.0005, 0.0004, 0.0003, 0.0002,
             0.0002, 0.0001])
        return wavelengths * 1e-9, n_values, k_values

    @staticmethod
    def get_al2o3_nk():
        """氧化铝的折射率数据 (Malitson)"""
        wavelengths = np.array(
            [400, 500, 600, 700, 800, 900, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000])  # nm
        n_values = np.array(
            [1.79, 1.77, 1.76, 1.75, 1.75, 1.74, 1.74, 1.72, 1.70, 1.68, 1.65, 1.62, 1.58, 1.53, 1.48, 1.42])
        k_values = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.001, 0.003, 0.01, 0.02, 0.04, 0.08])
        return wavelengths * 1e-9, n_values, k_values

    @staticmethod
    def get_cr_nk():
        """铬的折射率数据 (Johnson)"""
        wavelengths = np.array(
            [400, 500, 600, 700, 800, 900, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000])  # nm
        n_values = np.array(
            [3.18, 3.21, 3.48, 3.82, 4.14, 4.43, 4.62, 5.20, 5.30, 5.32, 5.30, 5.25, 5.18, 5.10, 5.02, 4.95])
        k_values = np.array(
            [3.34, 3.96, 4.36, 4.60, 4.74, 4.82, 4.87, 5.65, 6.15, 6.50, 6.75, 6.95, 7.10, 7.20, 7.28, 7.35])
        return wavelengths * 1e-9, n_values, k_values

    @staticmethod
    def get_ag_nk():
        """银的折射率数据 (Johnson)"""
        wavelengths = np.array(
            [400, 500, 600, 700, 800, 900, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000])  # nm
        n_values = np.array(
            [0.15, 0.14, 0.13, 0.14, 0.16, 0.17, 0.18, 0.65, 1.30, 2.00, 2.80, 3.60, 4.40, 5.20, 6.00, 6.80])
        k_values = np.array(
            [1.95, 3.09, 3.92, 4.64, 5.26, 5.85, 6.42, 13.5, 20.3, 27.5, 34.5, 41.5, 48.5, 55.5, 62.5, 69.5])
        return wavelengths * 1e-9, n_values, k_values

    @staticmethod
    def get_au_nk():
        """金的折射率数据 (Johnson)"""
        wavelengths = np.array(
            [400, 500, 600, 700, 800, 900, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000])  # nm
        n_values = np.array(
            [1.49, 0.83, 0.27, 0.17, 0.16, 0.17, 0.19, 0.70, 1.35, 2.05, 2.85, 3.65, 4.45, 5.25, 6.05, 6.85])
        k_values = np.array(
            [1.87, 2.01, 2.63, 3.15, 3.60, 4.03, 4.44, 9.50, 14.5, 20.0, 25.5, 31.0, 36.5, 42.0, 47.5, 53.0])
        return wavelengths * 1e-9, n_values, k_values


class MultilayerGeometry:
    """多层膜几何结构定义"""

    def __init__(self, materials_sequence, initial_thicknesses, bounds=None):
        """
        参数:
            materials_sequence: 材料序列列表，如 ['Cr', 'Au', 'Al2O3', 'Si']
            initial_thicknesses: 初始厚度列表 (米)
            bounds: 厚度边界 [(min1, max1), (min2, max2), ...]
        """
        self.materials = materials_sequence
        self.n_layers = len(materials_sequence)
        self.initial_thicknesses = np.array(initial_thicknesses)

        if bounds is None:
            self.bounds = [(10e-9, 500e-9)] * self.n_layers
        else:
            self.bounds = bounds

        # 获取材料数据
        self.material_data = self._load_material_data()

    def _load_material_data(self):
        """加载所有材料的光学常数"""
        material_functions = {
            'Si': MaterialDatabase.get_si_nk,
            'Al2O3': MaterialDatabase.get_al2o3_nk,
            'Cr': MaterialDatabase.get_cr_nk,
            'Ag': MaterialDatabase.get_ag_nk,
            'Au': MaterialDatabase.get_au_nk
        }

        data = {}
        for mat in set(self.materials):
            wl, n, k = material_functions[mat]()
            data[mat] = {
                'wavelengths': wl,
                'n': n,
                'k': k,
                'n_interp': interp1d(wl, n, kind='cubic', fill_value='extrapolate'),
                'k_interp': interp1d(wl, k, kind='cubic', fill_value='extrapolate')
            }
        return data

    def create_polygon_function(self):
        """创建多边形函数用于lumopt几何定义"""

        def polygon_func(params):
            """根据厚度参数创建多层膜多边形"""
            thicknesses = params
            z_positions = np.cumsum([0] + list(thicknesses))
            x_span = 10e-6  # 横向尺寸

            points = []
            for i in range(len(thicknesses)):
                z1, z2 = z_positions[i], z_positions[i + 1]
                # 每层定义为矩形
                layer_points = [
                    [-x_span / 2, z1],
                    [x_span / 2, z1],
                    [x_span / 2, z2],
                    [-x_span / 2, z2]
                ]
                points.extend(layer_points)

            return np.array(points)

        return polygon_func


class AbsorptionFOM:
    """吸收光谱匹配的品质因子"""

    def __init__(self, target_spectrum, wavelengths, T_monitor='T', R_monitor='R'):
        """
        参数:
            target_spectrum: 目标吸收光谱
            wavelengths: 波长数组
            T_monitor: 透射监视器名称
            R_monitor: 反射监视器名称
        """
        self.target = target_spectrum
        self.wavelengths = wavelengths
        self.T_monitor = T_monitor
        self.R_monitor = R_monitor

    def get_fom(self, simulation):
        """计算品质因子"""
        # 获取透射和反射
        T = simulation.getresult(self.T_monitor, 'T')['T']
        R = simulation.getresult(self.R_monitor, 'R')['R']

        # 计算吸收
        A = 1 - T - R

        # 计算与目标的均方误差
        error = np.mean((A - self.target) ** 2)

        # 返回品质因子（越大越好）
        return 1.0 / (1.0 + error)

    def get_gradient(self, simulation):
        """计算梯度（使用伴随方法）"""
        # lumopt会自动处理梯度计算
        pass


class OptimizationMonitor:
    """优化过程监控器"""

    def __init__(self, wavelengths):
        self.wavelengths = wavelengths
        self.history = {
            'iteration': [],
            'fom': [],
            'parameters': [],
            'spectra': [],
            'gradients': []
        }

    def update(self, iteration, fom, parameters, spectrum, gradients=None):
        """更新历史记录"""
        self.history['iteration'].append(iteration)
        self.history['fom'].append(fom)
        self.history['parameters'].append(parameters.copy())
        self.history['spectra'].append(spectrum.copy())
        if gradients is not None:
            self.history['gradients'].append(np.linalg.norm(gradients))

    def plot_convergence(self):
        """绘制收敛曲线"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 品质因子演化
        axes[0, 0].plot(self.history['iteration'], self.history['fom'], 'b-', linewidth=2)
        axes[0, 0].set_xlabel('迭代次数', fontsize=12)
        axes[0, 0].set_ylabel('品质因子', fontsize=12)
        axes[0, 0].set_title('优化收敛曲线', fontsize=14)
        axes[0, 0].grid(True, alpha=0.3)

        # 光谱演化
        wavelengths_nm = self.wavelengths * 1e9
        cmap = plt.cm.viridis
        n_spectra = len(self.history['spectra'])

        for i in range(0, n_spectra, max(1, n_spectra // 10)):
            color = cmap(i / n_spectra)
            axes[0, 1].plot(wavelengths_nm, self.history['spectra'][i],
                            color=color, alpha=0.7, linewidth=1.5)

        axes[0, 1].set_xlabel('波长 (nm)', fontsize=12)
        axes[0, 1].set_ylabel('吸收率', fontsize=12)
        axes[0, 1].set_title('吸收光谱演化', fontsize=14)
        axes[0, 1].grid(True, alpha=0.3)

        # 参数演化
        parameters_array = np.array(self.history['parameters']) * 1e9  # 转换为nm
        for i in range(parameters_array.shape[1]):
            axes[1, 0].plot(self.history['iteration'], parameters_array[:, i],
                            label=f'层 {i + 1}', linewidth=2)

        axes[1, 0].set_xlabel('迭代次数', fontsize=12)
        axes[1, 0].set_ylabel('层厚度 (nm)', fontsize=12)
        axes[1, 0].set_title('层厚度演化', fontsize=14)
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # 梯度范数
        if self.history['gradients']:
            axes[1, 1].semilogy(self.history['iteration'][:len(self.history['gradients'])],
                                self.history['gradients'], 'r-', linewidth=2)
            axes[1, 1].set_xlabel('迭代次数', fontsize=12)
            axes[1, 1].set_ylabel('梯度范数', fontsize=12)
            axes[1, 1].set_title('梯度收敛', fontsize=14)
            axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def save_results(self, filename):
        """保存优化结果"""
        np.savez(filename,
                 wavelengths=self.wavelengths,
                 history=self.history,
                 final_parameters=self.history['parameters'][-1],
                 final_spectrum=self.history['spectra'][-1])


def create_dual_peak_target(wavelengths):
    """创建双峰目标吸收光谱"""
    wavelengths_nm = wavelengths * 1e9

    # 可见光峰 (中心600nm，半宽100nm)
    visible_peak = 0.85 * np.exp(-((wavelengths_nm - 600) / 100) ** 2)

    # 中红外峰 (中心4500nm，半宽800nm)
    mir_peak = 0.90 * np.exp(-((wavelengths_nm - 4500) / 800) ** 2)

    # 组合双峰
    target_absorption = visible_peak + mir_peak

    # 归一化到合理范围
    target_absorption = np.clip(target_absorption, 0, 1)

    return target_absorption


def generate_lumerical_script(geometry, wavelengths):
    """生成Lumerical基础脚本"""
    script = f"""
# 多层膜优化基础脚本
# 清除工作空间
switchtolayout;
selectall;
delete;

# 设置仿真区域
addfdtd;
set("dimension", "2D");
set("x", 0);
set("x span", 20e-6);
set("y", 0);
set("y span", 0);
set("z", 0);
set("z span", 15e-6);
set("mesh accuracy", 3);

# 添加平面波光源
addplane;
set("name", "source");
set("injection axis", "z");
set("direction", "Backward");
set("x", 0);
set("x span", 30e-6);
set("y", 0);
set("y span", 0);
set("z", 10e-6);
set("wavelength start", {wavelengths[0]});
set("wavelength stop", {wavelengths[-1]});

# 添加反射监视器
addpower;
set("name", "R");
set("monitor type", "2D z-normal");
set("x", 0);
set("x span", 30e-6);
set("y", 0);
set("y span", 0);
set("z", 9e-6);

# 添加透射监视器
addpower;
set("name", "T");
set("monitor type", "2D z-normal");
set("x", 0);
set("x span", 30e-6);
set("y", 0);
set("y span", 0);
set("z", -5e-6);

# 添加场监视器（用于伴随优化）
addpower;
set("name", "opt_fields");
set("monitor type", "3D");
set("x", 0);
set("x span", 20e-6);
set("y", 0);
set("y span", 0);
set("z min", -1e-6);
set("z max", 8e-6);

# 添加基底
addrect;
set("name", "substrate");
set("material", "SiO2 (Glass) - Palik");
set("x", 0);
set("x span", 40e-6);
set("y", 0);
set("y span", 10e-6);
set("z min", -10e-6);
set("z max", 0);

# 边界条件
setglobalsource("wavelength start", {wavelengths[0]});
setglobalsource("wavelength stop", {wavelengths[-1]});
"""

    return script


def run_optimization():
    """主优化函数"""

    # 定义波长范围
    wavelengths = np.linspace(400e-9, 10000e-9, 51)

    # 创建目标吸收光谱
    target_absorption = create_dual_peak_target(wavelengths)

    # 定义初始多层膜结构
    # 策略：交替使用金属和介质实现双峰
    materials_sequence = [
        'Cr',  # 粘附层
        'Au',  # 等离激元层（可见光吸收）
        'Al2O3',  # 间隔层
        'Si',  # 高折射率层（中红外腔）
        'Al2O3',  # 低折射率层
        'Si',  # 部分反射层
        'Ag'  # 背反射镜
    ]

    # 初始厚度（纳米）
    initial_thicknesses = np.array([
        5e-9,  # Cr: 5nm
        20e-9,  # Au: 20nm
        50e-9,  # Al2O3: 50nm
        500e-9,  # Si: 500nm
        800e-9,  # Al2O3: 800nm
        300e-9,  # Si: 300nm
        100e-9  # Ag: 100nm
    ])

    # 厚度边界
    bounds = [
        (2e-9, 20e-9),  # Cr
        (10e-9, 50e-9),  # Au
        (20e-9, 200e-9),  # Al2O3
        (200e-9, 1000e-9),  # Si
        (400e-9, 1500e-9),  # Al2O3
        (100e-9, 600e-9),  # Si
        (50e-9, 200e-9)  # Ag
    ]

    # 创建几何结构
    geometry = MultilayerGeometry(materials_sequence, initial_thicknesses, bounds)

    # 创建lumopt波长对象
    lumopt_wavelengths = Wavelengths(start=400e-9, stop=10000e-9, points=51)

    # 创建几何优化对象
    polygon_func = geometry.create_polygon_function()
    lumopt_geometry = FunctionDefinedPolygon(
        func=polygon_func,
        initial_params=initial_thicknesses,
        bounds=bounds,
        z=0.0,
        depth=10e-6,
        eps_out=1.0,  # 空气
        eps_in=3.5,  # 平均介电常数
        edge_precision=5,
        dx=1e-9
    )

    # 创建品质因子
    fom = AbsorptionFOM(target_absorption, wavelengths)

    # 创建优化器
    optimizer = ScipyOptimizers(
        max_iter=100,
        method='L-BFGS-B',
        scaling_factor=1e9,  # 将厚度缩放到纳米量级
        pgtol=1e-5,
        ftol=1e-12,
        target_fom=0.95,
        scale_initial_gradient_to=0.1
    )

    # 生成基础脚本
    base_script = generate_lumerical_script(geometry, wavelengths)
    with open('multilayer_base.lsf', 'w') as f:
        f.write(base_script)

    # 创建监控器
    monitor = OptimizationMonitor(wavelengths)

    # 创建优化对象
    opt = Optimization(
        base_script='multilayer_base.lsf',
        wavelengths=lumopt_wavelengths,
        fom=fom,
        geometry=lumopt_geometry,
        optimizer=optimizer,
        hide_fdtd_cad=False,
        use_deps=True,
        plot_history=True,
        store_all_simulations=True
    )

    # 运行优化
    print("开始优化过程...")
    print(f"材料序列: {materials_sequence}")
    print(f"初始厚度 (nm): {initial_thicknesses * 1e9}")
    print(f"目标：可见光-中红外双峰吸收")
    print("-" * 50)

    # 执行优化
    opt.run()

    # 获取最终结果
    final_params = opt.optimizer.current_params
    final_fom = opt.optimizer.current_fom

    print("\n优化完成!")
    print(f"最终品质因子: {final_fom:.4f}")
    print(f"最终厚度 (nm): {final_params * 1e9}")

    # 绘制结果
    monitor.plot_convergence()

    # 保存结果
    monitor.save_results('optimization_results.npz')

    # 绘制最终光谱对比
    plt.figure(figsize=(10, 6))
    plt.plot(wavelengths * 1e9, target_absorption, 'k--', linewidth=2, label='目标光谱')
    plt.plot(wavelengths * 1e9, monitor.history['spectra'][-1], 'r-', linewidth=2, label='优化结果')
    plt.xlabel('波长 (nm)', fontsize=12)
    plt.ylabel('吸收率', fontsize=12)
    plt.title('双峰吸收光谱优化结果', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    return opt, monitor


if __name__ == "__main__":
    # 运行优化
    optimization, monitor = run_optimization()

    print("\n优化过程完成！")
    print("结果已保存至 'optimization_results.npz'")