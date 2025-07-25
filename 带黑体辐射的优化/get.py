import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import lumapi
import os
import json
import pickle
import hashlib
import gc
import time
from datetime import datetime
from functools import lru_cache


# ================== 配置参数 ==================
class Config:
    """配置类，存储所有参数"""
    # 遗传算法参数
    MIN_LAYERS = 2
    MAX_LAYERS = 10
    DNA_SIZE_LAYERS = 4  # 层数编码位数
    DNA_SIZE_THICKNESS = 10  # 厚度编码位数
    DNA_SIZE_MATERIAL = 4  # 材料编码位数
    POP_SIZE = 20
    CROSSOVER_RATE = 0.8
    MUTATION_RATE = 0.02
    N_GENERATIONS = 10
    THICKNESS_BOUND = [10, 1000]  # 10nm到1000nm

    # 光学参数
    WAVELENGTH_START = 5.0  # μm
    WAVELENGTH_STOP = 8.0  # μm
    FREQUENCY_POINTS = 1000

    # 材料惩罚参数
    METAL_PENALTY_PER_LAYER = 0.02  # 每个金属层的惩罚
    MAX_METAL_LAYERS_PREFERRED = 2  # 首选最多这么多金属层
    LAYER_COUNT_PENALTY = 0.00001  # 每层惩罚，倾向于更简单的设计

    # 文件路径
    MATERIAL_DB_PATH = r"D:\桌面\大创\大创优化\112.mdf"
    OUTPUT_DIR = "optimization_results"
    BASE_SIMULATION_FILE = r"D:\桌面\大创\大创优化\不带黑体辐射的优化\Multilayer_Absorber.fsp"  # 基础仿真模板文件

    # 材料列表
    MATERIALS = [
        'SiO2 (Glass) - Palik', 'Si (Silicon) - Palik', 'Al2O3 - Palik',
        'Cr (Chromium) - Palik', 'Ag (Silver) - CRC', 'Au (Gold) - Palik',
        'Cu (Copper) - Palik', 'Al (Aluminium) - Palik', 'Nb2O3 - Custom'
    ]

    DIELECTRIC_MATERIALS = {
        'SiO2 (Glass) - Palik', 'Si (Silicon) - Palik', 'Al2O3 - Palik', 'Nb2O3 - Custom'
    }

    # 优化配置
    CACHE_SIZE = 1000
    MEMORY_CHECK_INTERVAL = 50  # 每50次评估检查一次内存
    SESSION_REFRESH_INTERVAL = 200  # 每200次评估刷新一次会话

    @property
    def DNA_SIZE(self):
        return self.DNA_SIZE_LAYERS + self.MAX_LAYERS * (self.DNA_SIZE_THICKNESS + self.DNA_SIZE_MATERIAL)


config = Config()


# ================== 缓存系统 ==================
class SimulationCache:
    """仿真结果缓存系统"""

    def __init__(self, cache_dir="./cache", max_memory_cache=100):
        self.cache_dir = cache_dir
        self.max_memory_cache = max_memory_cache
        os.makedirs(cache_dir, exist_ok=True)
        self.memory_cache = {}

    def _hash_parameters(self, params):
        """为参数集生成唯一哈希"""
        if isinstance(params, dict):
            param_str = json.dumps(params, sort_keys=True)
        else:
            param_str = json.dumps(params.tolist() if hasattr(params, 'tolist') else params)
        return hashlib.md5(param_str.encode()).hexdigest()

    def get_cached_result(self, params):
        """获取缓存结果"""
        param_hash = self._hash_parameters(params)

        # 首先检查内存缓存
        if param_hash in self.memory_cache:
            return self.memory_cache[param_hash]

        # 检查磁盘缓存
        cache_file = os.path.join(self.cache_dir, f"{param_hash}.pkl")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    result = pickle.load(f)
                # 添加到内存缓存
                if len(self.memory_cache) < self.max_memory_cache:
                    self.memory_cache[param_hash] = result
                return result
            except:
                pass

        return None

    def store_result(self, params, result):
        """存储结果到缓存"""
        param_hash = self._hash_parameters(params)

        # 存储到内存缓存
        if len(self.memory_cache) < self.max_memory_cache:
            self.memory_cache[param_hash] = result

        # 存储到磁盘缓存
        cache_file = os.path.join(self.cache_dir, f"{param_hash}.pkl")
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(result, f)
        except:
            pass

    def clear_cache(self):
        """清空缓存"""
        self.memory_cache.clear()


# ================== 内存管理器 ==================
class MemoryManager:
    """内存管理器"""

    def __init__(self, cleanup_interval=50):
        self.cleanup_interval = cleanup_interval
        self.evaluation_count = 0

    def check_and_cleanup(self):
        """检查并清理内存"""
        self.evaluation_count += 1

        if self.evaluation_count % self.cleanup_interval == 0:
            gc.collect()  # 强制垃圾回收
            print(f"内存清理完成 (评估次数: {self.evaluation_count})")


# ================== 仿真模板管理器 ==================
class SimulationTemplateManager:
    """仿真模板管理器"""

    def __init__(self, base_file):
        self.base_file = base_file
        self.is_template_ready = False

    def setup_template(self, fdtd):
        """设置多层结构模板"""
        if self.is_template_ready:
            return

        nm = 1e-9
        um = 1e-6
        a = 0.1 * um  # 横向尺寸

        try:
            # 导入材料数据库
            if os.path.exists(config.MATERIAL_DB_PATH):
                fdtd.importmaterialdb(config.MATERIAL_DB_PATH)

            # 添加衬底
            fdtd.addrect(
                name='substrate',
                x=0, y=0,
                x_span=a, y_span=a,
                z_min=0, z_max=200 * nm,
                material='SiO2 (Glass) - Palik'
            )

            # 创建最大数量的层模板
            for i in range(config.MAX_LAYERS):
                layer_name = f'layer_{i}'
                fdtd.addrect(
                    name=layer_name,
                    x=0, y=0,
                    x_span=a, y_span=a,
                    z_min=200 * nm + i * 50 * nm,
                    z_max=200 * nm + (i + 1) * 50 * nm,
                    material='etch',  # 默认透明材料
                    enabled=0  # 默认禁用
                )

            # 添加FDTD区域
            total_height = 200 * nm + config.MAX_LAYERS * 200 * nm
            fdtd.addfdtd(
                dimension='3D',
                x=0, y=0,
                x_span=a / 2, y_span=a / 2,
                z_min=-1 * um, z_max=total_height + 1 * um,
                x_min_bc='periodic', y_min_bc='periodic',
                mesh_accuracy=3
            )

            # 添加平面波源
            fdtd.addplane(
                injection_axis='z',
                direction='backward',
                polarization_angle=0,
                x=0, y=0,
                x_span=a, y_span=a,
                z=total_height + 0.5 * um,
                wavelength_start=config.WAVELENGTH_START * um,
                wavelength_stop=config.WAVELENGTH_STOP * um
            )

            # 添加监视器
            fdtd.addpower(
                name="R",
                x=0, y=0, z=total_height + 0.8 * um,
                x_span=a / 2, y_span=a / 2,
                monitor_type="2D Z-Normal",
                frequency_points=config.FREQUENCY_POINTS
            )

            fdtd.addpower(
                name="T",
                x=0, y=0, z=-0.5 * um,
                x_span=a / 2, y_span=a / 2,
                monitor_type="2D Z-Normal",
                frequency_points=config.FREQUENCY_POINTS
            )

            # 保存模板
            fdtd.save(config.BASE_SIMULATION_FILE)
            self.is_template_ready = True
            print("仿真模板设置完成")

        except Exception as e:
            print(f"模板设置错误: {e}")
            raise

    def update_structure(self, fdtd, n_layers, params):
        """更新多层结构"""
        nm = 1e-9

        try:
            # 首先禁用所有层
            for i in range(config.MAX_LAYERS):
                fdtd.setnamed(f'layer_{i}', 'enabled', 0)

            # 配置活动层
            z_current = 200 * nm
            for i in range(n_layers):
                thickness, mat_idx = params[i]
                layer_name = f'layer_{i}'
                material = config.MATERIALS[mat_idx]

                # 启用并配置层
                fdtd.setnamed(layer_name, 'enabled', 1)
                fdtd.setnamed(layer_name, 'z min', z_current)
                fdtd.setnamed(layer_name, 'z max', z_current + thickness * nm)
                fdtd.setnamed(layer_name, 'material', material)

                z_current += thickness * nm

            return True

        except Exception as e:
            print(f"结构更新错误: {e}")
            return False


# ================== 优化的光学仿真函数 ==================
class OptimizedSimulation:
    """优化的仿真类"""

    def __init__(self):
        self.cache = SimulationCache()
        self.memory_manager = MemoryManager()
        self.template_manager = SimulationTemplateManager(config.BASE_SIMULATION_FILE)
        self.fdtd_session = None
        self.evaluation_count = 0

    def __enter__(self):
        """进入上下文管理器"""
        self.fdtd_session = lumapi.FDTD()
        self.template_manager.setup_template(self.fdtd_session)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """退出上下文管理器"""
        if self.fdtd_session:
            self.fdtd_session.close()
            self.fdtd_session = None

    def get_spectrum(self, n_layers, params):
        """获取吸收光谱"""
        # 检查缓存
        param_key = {
            'n_layers': n_layers,
            'params': params
        }

        cached_result = self.cache.get_cached_result(param_key)
        if cached_result is not None:
            return cached_result['wavelengths'], cached_result['absorption']

        try:
            # 更新结构
            if not self.template_manager.update_structure(self.fdtd_session, n_layers, params):
                return self._get_default_spectrum()

            # 运行仿真
            self.fdtd_session.run()

            # 获取结果
            R = self.fdtd_session.transmission('R')
            T = self.fdtd_session.transmission('T')
            A = 1 - R - T

            # 获取波长数据
            freq = self.fdtd_session.getdata('R', 'f')
            c = 3e8  # 光速
            wavelengths = c / freq / 1e-6  # 转换为μm

            # 缓存结果
            result = {
                'wavelengths': wavelengths,
                'absorption': A
            }
            self.cache.store_result(param_key, result)

            # 内存管理
            self.evaluation_count += 1
            if self.evaluation_count % config.MEMORY_CHECK_INTERVAL == 0:
                self.memory_manager.check_and_cleanup()

            # 定期刷新会话以防止内存泄漏
            if self.evaluation_count % config.SESSION_REFRESH_INTERVAL == 0:
                self._refresh_session()

            return wavelengths, A

        except Exception as e:
            print(f"仿真错误: {e}")
            return self._get_default_spectrum()

    def _get_default_spectrum(self):
        """返回默认光谱"""
        wavelengths = np.linspace(config.WAVELENGTH_START, config.WAVELENGTH_STOP, config.FREQUENCY_POINTS)
        return wavelengths, np.zeros(config.FREQUENCY_POINTS)

    def _refresh_session(self):
        """刷新FDTD会话"""
        print("刷新FDTD会话...")
        if self.fdtd_session:
            self.fdtd_session.load(config.BASE_SIMULATION_FILE)
        gc.collect()


# ================== 遗传算法函数 ==================
def translateDNA(pop):
    """将二进制DNA转换为实际参数"""
    decoded_pop = []

    for individual in pop:
        # 解码层数
        layer_bits = individual[:config.DNA_SIZE_LAYERS]
        n_layers_raw = layer_bits.dot(2 ** np.arange(config.DNA_SIZE_LAYERS)[::-1])
        n_layers = config.MIN_LAYERS + (n_layers_raw % (config.MAX_LAYERS - config.MIN_LAYERS + 1))

        # 解码层参数
        params = []
        for layer in range(n_layers):
            start_idx = config.DNA_SIZE_LAYERS + layer * (config.DNA_SIZE_THICKNESS + config.DNA_SIZE_MATERIAL)

            # 解码厚度
            thickness_bits = individual[start_idx:start_idx + config.DNA_SIZE_THICKNESS]
            thickness_normalized = thickness_bits.dot(2 ** np.arange(config.DNA_SIZE_THICKNESS)[::-1]) / (
                    2 ** config.DNA_SIZE_THICKNESS - 1)
            thickness = thickness_normalized * (config.THICKNESS_BOUND[1] - config.THICKNESS_BOUND[0]) + \
                        config.THICKNESS_BOUND[0]

            # 解码材料
            material_bits = individual[
                            start_idx + config.DNA_SIZE_THICKNESS:start_idx + config.DNA_SIZE_THICKNESS + config.DNA_SIZE_MATERIAL]
            material_idx = material_bits.dot(2 ** np.arange(config.DNA_SIZE_MATERIAL)[::-1])
            material_idx = min(material_idx, len(config.MATERIALS) - 1)

            params.append((thickness, material_idx))

        decoded_pop.append((n_layers, params))

    return decoded_pop


def get_fitness(pop, simulation):
    """计算种群的适应度"""
    fitness = np.zeros(config.POP_SIZE)
    decoded_pop = translateDNA(pop)

    for i in range(config.POP_SIZE):
        n_layers, params = decoded_pop[i]

        try:
            # 获取吸收光谱
            wavelengths, absorption = simulation.get_spectrum(n_layers, params[:n_layers])

            # 计算平均吸收
            avg_absorption = np.mean(absorption)

            # 计算金属层数
            metal_layer_count = sum(
                1 for j in range(n_layers)
                if config.MATERIALS[params[j][1]] not in config.DIELECTRIC_MATERIALS
            )

            # 材料偏好因子
            material_preference_factor = -config.METAL_PENALTY_PER_LAYER * metal_layer_count

            # 过多金属层的额外惩罚
            if metal_layer_count > config.MAX_METAL_LAYERS_PREFERRED:
                excess_metals = metal_layer_count - config.MAX_METAL_LAYERS_PREFERRED
                material_preference_factor -= 0.05 * excess_metals

            # 计算总适应度
            fitness[i] = avg_absorption + material_preference_factor - config.LAYER_COUNT_PENALTY * n_layers
            fitness[i] = max(0, fitness[i])  # 确保非负适应度

        except Exception as e:
            print(f"计算个体 {i} 适应度时出错: {e}")
            fitness[i] = 0

        # 进度显示
        if (i + 1) % 5 == 0:
            print(f"已评估 {i + 1}/{config.POP_SIZE} 个个体")

    return fitness


def crossover_and_mutation(pop):
    """应用交叉和变异算子"""
    new_pop = []

    for father in pop:
        child = father.copy()

        # 交叉
        if np.random.rand() < config.CROSSOVER_RATE:
            mother = pop[np.random.randint(config.POP_SIZE)]
            cross_points = np.random.randint(0, config.DNA_SIZE)
            child[cross_points:] = mother[cross_points:]

        # 变异
        mutation(child)
        new_pop.append(child)

    return new_pop


def mutation(child):
    """对个体应用变异"""
    n_mutations = np.random.poisson(config.DNA_SIZE * config.MUTATION_RATE)
    for _ in range(n_mutations):
        mutate_point = np.random.randint(0, config.DNA_SIZE)
        child[mutate_point] = child[mutate_point] ^ 1  # 翻转位


def select(pop, fitness):
    """锦标赛选择"""
    selected = []

    for _ in range(config.POP_SIZE):
        # 大小为3的锦标赛选择
        tournament_idx = np.random.choice(config.POP_SIZE, 3, replace=False)
        tournament_fitness = fitness[tournament_idx]
        winner_idx = tournament_idx[np.argmax(tournament_fitness)]
        selected.append(pop[winner_idx])

    return np.array(selected)


# ================== 工具函数 ==================
def create_output_directory():
    """创建输出目录"""
    if not os.path.exists(config.OUTPUT_DIR):
        os.makedirs(config.OUTPUT_DIR)


def save_results(n_layers, params, wavelengths, absorption, best_fitness_history, avg_fitness_history):
    """保存优化结果到文件"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 保存配置
    config_dict = {
        'n_layers': n_layers,
        'layers': []
    }

    for i in range(n_layers):
        thickness, mat_idx = params[i]
        config_dict['layers'].append({
            'thickness_nm': thickness,
            'material': config.MATERIALS[mat_idx]
        })

    with open(f"{config.OUTPUT_DIR}/best_design_{timestamp}.json", 'w', encoding='utf-8') as f:
        json.dump(config_dict, f, indent=4, ensure_ascii=False)

    # 保存光谱数据
    spectrum_data = np.column_stack((wavelengths, absorption))
    np.savetxt(f"{config.OUTPUT_DIR}/spectrum_{timestamp}.txt", spectrum_data,
               header='Wavelength(um) Absorption', fmt='%.6f')

    # 保存适应度历史
    fitness_data = np.column_stack((best_fitness_history, avg_fitness_history))
    np.savetxt(f"{config.OUTPUT_DIR}/fitness_history_{timestamp}.txt", fitness_data,
               header='Best_Fitness Average_Fitness', fmt='%.6f')

    print(f"\n结果已保存到 {config.OUTPUT_DIR}/")


def plot_results(wavelengths, absorption, best_fitness_history, avg_fitness_history,
                 n_layers, params):
    """绘制优化结果"""
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # 绘制吸收光谱
    ax1.plot(wavelengths, absorption, 'b-', linewidth=2)
    ax1.set_xlabel('波长 (μm)')
    ax1.set_ylabel('吸收率')
    ax1.set_title(f'优化的多层吸收器 ({n_layers} 层)')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1])

    # 添加层信息
    layer_info = "层配置:\n"
    for i in range(n_layers):
        thickness, mat_idx = params[i]
        layer_info += f"第{i + 1}层: {thickness:.1f}nm - {config.MATERIALS[mat_idx]}\n"

    ax1.text(0.02, 0.98, layer_info, transform=ax1.transAxes,
             verticalalignment='top', fontsize=8,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # 绘制适应度演化
    generations = range(len(best_fitness_history))
    ax2.plot(generations, best_fitness_history, 'r-', label='最佳', linewidth=2)
    ax2.plot(generations, avg_fitness_history, 'b--', label='平均', linewidth=1)
    ax2.set_xlabel('代数')
    ax2.set_ylabel('适应度')
    ax2.set_title('适应度演化')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f"{config.OUTPUT_DIR}/optimization_results_{timestamp}.png", dpi=300)
    plt.show()


# ================== 主优化函数 ==================
def optimize_multilayer_absorber():
    """主优化函数"""
    # 创建输出目录
    create_output_directory()

    # 初始化种群
    pop = np.random.randint(2, size=(config.POP_SIZE, config.DNA_SIZE))

    # 跟踪最佳适应度历史
    best_fitness_history = []
    avg_fitness_history = []

    print("开始优化...")
    print(f"种群大小: {config.POP_SIZE}, 代数: {config.N_GENERATIONS}")

    # 使用优化的仿真类
    with OptimizedSimulation() as simulation:
        # 进化循环
        for generation in range(config.N_GENERATIONS):
            print(f"\n第 {generation + 1}/{config.N_GENERATIONS} 代:")

            # 应用遗传算子
            pop = np.array(crossover_and_mutation(pop))

            # 计算适应度
            fitness = get_fitness(pop, simulation)

            # 选择
            pop = select(pop, fitness)

            # 跟踪统计信息
            best_fitness = np.max(fitness)
            avg_fitness = np.mean(fitness)
            best_fitness_history.append(best_fitness)
            avg_fitness_history.append(avg_fitness)

            # 打印进度
            print(f"最佳适应度: {best_fitness:.4f}, 平均适应度: {avg_fitness:.4f}")

    # 获取最佳解
    best_idx = np.argmax(fitness)
    decoded_pop = translateDNA(pop)
    best_n_layers, best_params = decoded_pop[best_idx]

    # 获取最终光谱
    with OptimizedSimulation() as simulation:
        wavelengths, absorption = simulation.get_spectrum(best_n_layers, best_params[:best_n_layers])

    # 保存结果
    save_results(best_n_layers, best_params, wavelengths, absorption,
                 best_fitness_history, avg_fitness_history)

    # 绘制结果
    plot_results(wavelengths, absorption, best_fitness_history, avg_fitness_history,
                 best_n_layers, best_params)

    return best_n_layers, best_params, wavelengths, absorption


# ================== 主执行部分 ==================
if __name__ == "__main__":
    try:
        print("=== 多层吸收器遗传算法优化 ===")
        print("使用优化的仿真策略，避免重复创建文件")

        # 运行优化
        best_n_layers, best_params, wavelengths, absorption = optimize_multilayer_absorber()

        # 打印最终结果
        print(f"\n优化完成!")
        print(f"最佳设计有 {best_n_layers} 层:")
        for i in range(best_n_layers):
            thickness, mat_idx = best_params[i]
            print(f"  第 {i + 1} 层: {thickness:.1f} nm - {config.MATERIALS[mat_idx]}")
        print(f"平均吸收率: {np.mean(absorption):.4f}")

    except Exception as e:
        print(f"优化过程中出现错误: {e}")
        import traceback

        traceback.print_exc()

    finally:
        print("程序结束")