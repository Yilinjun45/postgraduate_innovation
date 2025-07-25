#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
多层光学薄膜逆向优化系统 - 直接API版本
使用Lumerical Python API直接构建仿真模型
用于设计双波段红外屏蔽薄膜（1.5-2.5μm和5-7μm）
"""

import numpy as np
import matplotlib.pyplot as plt
import lumapi
from lumopt.optimizers import ScipyOptimizers
from lumopt.figures_of_merit import FigureOfMerit
from lumopt.optimization import Optimization

# 可用材料列表（直接使用FDTD中的名称）
MATERIALS = [
    'SiO2 (Glass) - Palik',
    'Si (Silicon) - Palik',
    'Al2O3 - Palik',
    'Cr (Chromium) - Palik',
    'Ag (Silver) - CRC',
    'Au (Gold) - Palik',
    'Cu (Copper) - Palik',
    'Al (Aluminium) - Palik'
]


class IRShieldingFOM(FigureOfMerit):
    '''红外屏蔽品质因数'''

    def __init__(self, wavelengths, monitor_name='T'):
        super().__init__()
        self.wavelengths = wavelengths
        self.wavelengths_nm = wavelengths * 1e9
        self.band1 = (1500, 2500)  # 第一屏蔽波段(nm)
        self.band2 = (5000, 7000)  # 第二屏蔽波段(nm)
        self.monitor_name = monitor_name
        self.target_transmission = self._create_target_transmission()

    def _create_target_transmission(self):
        '''创建目标透射率曲线：屏蔽波段透射率为0，其余为1'''
        target = np.ones_like(self.wavelengths_nm)
        for i, wl in enumerate(self.wavelengths_nm):
            if (self.band1[0] <= wl <= self.band1[1]) or (self.band2[0] <= wl <= self.band2[1]):
                target[i] = 0.0
        return target

    def initialize(self, sim):
        """初始化FOM（Lumopt要求）"""
        pass

    def get_fom(self, sim, params) -> float:
        """计算品质因数"""
        # 获取透射率数据
        T_data = sim.fdtd.getresult(self.monitor_name, 'T')
        T = np.array(T_data).flatten()

        # 计算均方误差
        mse = np.mean((T - self.target_transmission) ** 2)
        fom = 1.0 / (1.0 + mse)

        # 计算屏蔽波段内的平均透射率
        band1_mask = (self.wavelengths_nm >= self.band1[0]) & (self.wavelengths_nm <= self.band1[1])
        band2_mask = (self.wavelengths_nm >= self.band2[0]) & (self.wavelengths_nm <= self.band2[1])

        band1_avg_T = np.mean(T[band1_mask]) if np.any(band1_mask) else 1.0
        band2_avg_T = np.mean(T[band2_mask]) if np.any(band2_mask) else 1.0

        # 添加屏蔽性能奖励因子
        shielding_bonus = (1.0 - band1_avg_T) * (1.0 - band2_avg_T)
        return fom * (1.0 + shielding_bonus)

    def get_gradients(self, sim, params) -> np.ndarray:
        """梯度计算（数值近似）"""
        return None


def create_ir_shielding_simulation(fdtd, materials, thicknesses, wavelengths):
    """直接使用Lumerical API创建仿真结构"""
    # 清除现有结构
    fdtd.switchtolayout()
    fdtd.selectall()
    fdtd.deleteall()

    # 定义单位
    um = 1e-6
    nm = 1e-9

    # 创建FDTD仿真区域
    fdtd.addfdtd()
    fdtd.set("dimension", "2D")
    fdtd.set("x span", 1 * um)
    fdtd.set("y span", 0)
    fdtd.set("z span", 15 * um)
    fdtd.set("mesh accuracy", 3)
    fdtd.set("simulation time", 1000e-15)
    fdtd.set("x min bc", "periodic")
    fdtd.set("x max bc", "periodic")
    fdtd.set("z min bc", "PML")
    fdtd.set("z max bc", "PML")

    # 添加光源
    fdtd.addplane()
    fdtd.set("name", "source")
    fdtd.set("injection axis", "z")
    fdtd.set("direction", "Backward")
    fdtd.set("x span", 1.1 * um)
    fdtd.set("y span", 0)
    fdtd.set("z", 6.5 * um)
    fdtd.set("wavelength start", min(wavelengths))
    fdtd.set("wavelength stop", max(wavelengths))

    # 添加反射率监视器
    fdtd.addpower()
    fdtd.set("name", "R")
    fdtd.set("monitor type", "2D Z-normal")
    fdtd.set("x span", 1 * um)
    fdtd.set("y span", 0)
    fdtd.set("z", 5.5 * um)

    # 添加透射率监视器
    fdtd.addpower()
    fdtd.set("name", "T")
    fdtd.set("monitor type", "2D Z-normal")
    fdtd.set("x span", 1 * um)
    fdtd.set("y span", 0)
    fdtd.set("z", -5.5 * um)

    # 添加基底
    fdtd.addrect()
    fdtd.set("name", "substrate")
    fdtd.set("material", "SiO2 (Glass) - Palik")
    fdtd.set("x span", 2 * um)
    fdtd.set("y span", 0)
    fdtd.set("z min", -7.5 * um)
    fdtd.set("z max", 0)

    # 添加薄膜层
    z_start = 0
    for i, (material, thickness) in enumerate(zip(materials, thicknesses)):
        fdtd.addrect()
        fdtd.set("name", f"layer_{i + 1}")
        fdtd.set("material", material)
        fdtd.set("x span", 2 * um)
        fdtd.set("y span", 0)
        fdtd.set("z min", z_start)
        fdtd.set("z max", z_start + thickness)
        z_start += thickness

    # 设置波长扫描
    fdtd.addsweep(0)
    fdtd.set("name", "sweep")
    fdtd.set("number of points", len(wavelengths))
    fdtd.set("wavelength start", min(wavelengths))
    fdtd.set("wavelength stop", max(wavelengths))

    # 保存初始设计
    fdtd.save("initial_design.fsp")


def plot_shielding_performance(wavelengths, transmission, reflection=None):
    wavelengths_um = wavelengths * 1e6
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.plot(wavelengths_um, transmission * 100, 'b-', linewidth=2, label='Transmission')
    plt.axvspan(1.5, 2.5, alpha=0.2, color='red', label='Shielding Band 1')
    plt.axvspan(5.0, 7.0, alpha=0.2, color='orange', label='Shielding Band 2')
    plt.xlabel('Wavelength (μm)')
    plt.ylabel('Transmission (%)')
    plt.title('Infrared Shielding Film Performance')
    plt.grid(True)
    plt.legend()
    plt.ylim(0, 100)

    if reflection is not None:
        plt.subplot(2, 1, 2)
        plt.plot(wavelengths_um, reflection * 100, 'r-', linewidth=2, label='Reflection')
        plt.axvspan(1.5, 2.5, alpha=0.2, color='red')
        plt.axvspan(5.0, 7.0, alpha=0.2, color='orange')
        plt.xlabel('Wavelength (μm)')
        plt.ylabel('Reflection (%)')
        plt.title('Reflection Characteristics')
        plt.grid(True)
        plt.legend()
        plt.ylim(0, 100)

    plt.tight_layout()
    plt.savefig('shielding_performance.png')
    plt.show()


def run_optimization():
    wavelengths = np.linspace(1.0e-6, 8.0e-6, 71)

    # 使用可用材料定义10层薄膜结构
    materials_sequence = [
        'Ag (Silver) - CRC',
        'SiO2 (Glass) - Palik',
        'Si (Silicon) - Palik',
        'SiO2 (Glass) - Palik',
        'Al2O3 - Palik',
        'SiO2 (Glass) - Palik',
        'Si (Silicon) - Palik',
        'SiO2 (Glass) - Palik',
        'Al (Aluminium) - Palik',
        'Al2O3 - Palik'
    ]

    initial_thicknesses = np.array([
        100e-9, 400e-9, 250e-9, 600e-9, 350e-9,
        800e-9, 300e-9, 500e-9, 400e-9, 200e-9
    ])

    bounds = [
        (50e-9, 200e-9), (200e-9, 800e-9), (100e-9, 500e-9),
        (300e-9, 1200e-9), (150e-9, 700e-9), (400e-9, 1600e-9),
        (150e-9, 600e-9), (250e-9, 1000e-9), (200e-9, 800e-9),
        (100e-9, 400e-9)
    ]

    try:
        # 创建FDTD会话
        fdtd = lumapi.FDTD()

        # 直接创建仿真结构
        create_ir_shielding_simulation(fdtd, materials_sequence, initial_thicknesses, wavelengths)

        # 定义更新几何形状的函数
        def update_geometry(params):
            z_start = 0
            for i, thickness in enumerate(params):
                layer_name = f"layer_{i + 1}"
                fdtd.select(layer_name)
                fdtd.set("z min", z_start)
                fdtd.set("z max", z_start + thickness)
                z_start += thickness
            return params

        # 创建FOM对象
        fom = IRShieldingFOM(wavelengths)

        # 创建优化器
        optimizer = ScipyOptimizers(
            max_iter=50,
            method='L-BFGS-B',
            scaling_factor=1e9,
            pgtol=1e-5,
            ftol=1e-6,
            scale_initial_gradient_to=1
        )

        # 创建优化对象
        optimization = Optimization(
            fom=fom,
            geometry=None,  # 使用参数更新函数而不是几何对象
            optimizer=optimizer,
            use_var_fdtd=False,
            hide_fdtd_cad=False
        )

        # 设置参数和边界
        optimization.params = initial_thicknesses
        optimization.bounds = bounds
        optimization.update_geometry = update_geometry

        print("\nStarting optimization...")
        # 运行优化
        optimized_thicknesses, final_fom = optimization.run(fdtd=fdtd)

        print("\nOptimization completed! Final thicknesses:")
        for i, t in enumerate(optimized_thicknesses):
            print(f"Layer {i + 1} {materials_sequence[i]}: {t * 1e9:.1f} nm")
        print(f"Final FOM: {final_fom:.4f}")

        # 更新到优化后的厚度
        update_geometry(optimized_thicknesses)

        # 运行最终优化后的仿真
        fdtd.run()
        fdtd.save('optimized_design.fsp')

        # 获取结果
        T_data = fdtd.getresult("T", "T")
        T = np.array(T_data).flatten()
        R_data = fdtd.getresult("R", "R")
        R = np.array(R_data).flatten()

        # 绘制性能曲线
        plot_shielding_performance(wavelengths, T, R)

        # 关闭FDTD会话
        fdtd.close()

    except Exception as e:
        print(f"\n❌ Optimization error: {e}")
        import traceback
        traceback.print_exc()
        print("Please ensure Lumerical FDTD is running and Lumopt is properly configured")


if __name__ == "__main__":
    print("=" * 50)
    print("Multilayer Infrared Shielding Film Optimization System")
    print("Using Direct Lumerical API")
    print("=" * 50)
    run_optimization()