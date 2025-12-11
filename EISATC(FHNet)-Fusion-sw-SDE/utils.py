# utils.py
import os
import numpy as np
import pandas as pd # 用于表格数据处理
import torch
# import torch.utils.data # 在此文件中未直接使用 DataLoader 相关功能
from sklearn.metrics import mean_absolute_error, mean_squared_error # 用于可能的外部评估，但不在核心图工具中使用
# from .metrics import masked_mape_np # 原来的相对导入
try:
    from metrics import masked_mape_np # 尝试绝对/同级导入
except ImportError:
    # 如果 metrics.py 不存在或无法导入，定义一个占位符或移除依赖此函数的代码
    print("[警告] utils.py: 无法从 metrics.py 导入 masked_mape_np。依赖此函数的代码可能会失败。") # 中文警告
    def masked_mape_np(y_true, y_pred, null_val=np.nan): # 占位符函数
        print("[错误] masked_mape_np 未实现或无法加载。") # 中文错误
        return float('inf')

from scipy.sparse.linalg import eigs # 用于计算拉普拉斯矩阵的最大特征值
from scipy.linalg import eigvalsh # 用于计算对称矩阵的特征值
from scipy.linalg import fractional_matrix_power # 用于计算矩阵的分数幂

# 可视化库，在此文件中主要用于被注释掉的 predict_and_save_results_mstgcn
# import matplotlib.pyplot as plt
# import matplotlib.ticker as ticker

# 归一化相关的函数 (如果其他模块需要，可以保留)
def re_normalization(x, mean, std): #
    x = x * std + mean #
    return x #


def max_min_normalization(x, _max, _min): #
    x = 1. * (x - _min)/(_max - _min) #
    x = x * 2. - 1. #
    return x #


def re_max_min_normalization(x, _max, _min): #
    x = (x + 1.) / 2. #
    x = 1. * x * (_max - _min) + _min #
    return x #

# 构建邻接矩阵的函数 (如果数据格式需要，这些是重要的工具)
def get_adjacency_matrix(distance_df_filename, num_of_vertices, id_filename=None): #
    '''
    从CSV文件或NPY文件加载邻接矩阵。
    Parameters
    ----------
    distance_df_filename: str, 包含边信息的CSV文件路径或NPY文件路径
    num_of_vertices: int, 顶点数量
    id_filename: str, optional, 包含节点ID到索引映射的文件路径

    Returns
    ----------
    A: np.ndarray, 邻接矩阵
    distaneA: np.ndarray or None, 距离矩阵 (如果从CSV加载且包含距离)
    '''
    if 'npy' in distance_df_filename: #
        print(f"[信息] 从.npy文件加载邻接矩阵: {distance_df_filename}") # 中文信息
        adj_mx = np.load(distance_df_filename) #
        return adj_mx, None # .npy通常只存一个矩阵

    else: #
        import csv # 仅在需要时导入
        print(f"[信息] 从CSV文件构建邻接矩阵: {distance_df_filename}") # 中文信息
        A = np.zeros((int(num_of_vertices), int(num_of_vertices)), dtype=np.float32) #
        distaneA = np.zeros((int(num_of_vertices), int(num_of_vertices)), dtype=np.float32) #

        id_dict = None #
        if id_filename: #
            print(f"[信息] 使用ID映射文件: {id_filename}") # 中文信息
            with open(id_filename, 'r') as f: #
                id_dict = {int(i): idx for idx, i in enumerate(f.read().strip().split('\n'))}  # 把节点id（idx）映射成从0开始的索引

        with open(distance_df_filename, 'r') as f: #
            f.readline()  # 跳过表头
            reader = csv.reader(f) #
            for row in reader: #
                if len(row) != 3: #
                    continue #
                i_orig, j_orig, val = int(row[0]), int(row[1]), float(row[2]) #
                
                i = id_dict[i_orig] if id_dict else i_orig #
                j = id_dict[j_orig] if id_dict else j_orig #

                if 0 <= i < num_of_vertices and 0 <= j < num_of_vertices: # 边界检查
                    A[i, j] = 1 # 假设CSV的第三列是权重或仅用于指示连接性
                    distaneA[i, j] = val # 存储原始值（例如距离）
                else: #
                    print(f"[警告] utils.py: get_adjacency_matrix - 索引 ({i}, {j}) 超出范围 ({num_of_vertices})。跳过此行：{row}") # 中文警告

        return A, distaneA #

def get_adjacency_matrix2(distance_df_filename, num_of_vertices, type_='connectivity', id_filename=None): #
    '''
    从CSV文件构建对称的邻接矩阵 (A[i,j]=1, A[j,i]=1)。
    Parameters
    ----------
    distance_df_filename: str, 包含边信息的CSV文件路径
    num_of_vertices: int, 顶点数量
    type_: str, {'connectivity', 'distance'} - 如果是distance，则权重为1/distance
    id_filename: str, optional, 包含节点ID到索引映射的文件路径

    Returns
    ----------
    A: np.ndarray, 对称的邻接矩阵
    '''
    import csv #
    print(f"[信息] 从CSV文件构建对称邻接矩阵: {distance_df_filename}, 类型: {type_}") # 中文信息
    A = np.zeros((int(num_of_vertices), int(num_of_vertices)), dtype=np.float32) #

    id_dict = None #
    if id_filename: #
        print(f"[信息] 使用ID映射文件: {id_filename}") # 中文信息
        with open(id_filename, 'r') as f: #
            id_dict = {int(i): idx for idx, i in enumerate(f.read().strip().split('\n'))} #

    with open(distance_df_filename, 'r') as f: #
        f.readline()  # 跳过表头
        reader = csv.reader(f) #
        for row in reader: #
            if len(row) != 3: #
                continue #
            i_orig, j_orig, val = int(row[0]), int(row[1]), float(row[2]) #
            
            i = id_dict[i_orig] if id_dict else i_orig #
            j = id_dict[j_orig] if id_dict else j_orig #

            if not (0 <= i < num_of_vertices and 0 <= j < num_of_vertices): # 边界检查
                print(f"[警告] utils.py: get_adjacency_matrix2 - 索引 ({i}, {j}) 超出范围 ({num_of_vertices})。跳过此行：{row}") # 中文警告
                continue #

            if type_ == 'connectivity': #
                A[i, j] = 1 #
                A[j, i] = 1 # 确保对称
            elif type_ == 'distance': #
                if val == 0: # 防止除以零
                    print(f"[警告] utils.py: get_adjacency_matrix2 - 距离为零，无法计算倒数。边 ({i},{j}) 权重设为0。") # 中文警告
                    A[i, j] = 0 #
                    A[j, i] = 0 #
                else: #
                    A[i, j] = 1.0 / val #
                    A[j, i] = 1.0 / val # 确保对称
            else: #
                raise ValueError("type_ 参数错误, 必须是 'connectivity' 或 'distance'!") #
    return A #

# --- DSTAGNN 核心图计算函数 ---
def scaled_Laplacian(W): #
    '''
    计算标准化的拉普拉斯算子 \tilde{L} = 2L/\lambda_{max} - I
    Parameters
    ----------
    W: np.ndarray, 邻接矩阵, shape (N, N)

    Returns
    ----------
    scaled_Laplacian: np.ndarray, 标准化的拉普拉斯算子, shape (N, N)
    '''
    if not isinstance(W, np.ndarray): # 类型检查
        raise TypeError("输入 W 必须是 NumPy 数组。") #
    if W.ndim != 2 or W.shape[0] != W.shape[1]: # 形状检查
        raise ValueError(f"输入 W 必须是方阵 (N,N)，但得到形状 {W.shape}。") #
    
    N = W.shape[0] #
    if N == 0: return np.array([]) # 处理空矩阵

    assert W.shape[0] == W.shape[1] #

    D = np.diag(np.sum(W, axis=1)) # 度矩阵
    L = D - W # 组合拉普拉斯算子
    
    try: #
        lambda_max = eigs(L, k=1, which='LR', return_eigenvectors=False)[0].real # 计算最大特征值
    except Exception as e: #
        print(f"[错误] utils.py: scaled_Laplacian - 计算特征值失败: {e}。可能矩阵L有问题。") # 中文错误
        # 尝试使用 eigvalsh (如果L是对称的) 作为备选，尽管eigs更适合稀疏或大型矩阵
        try: #
            eigenvalues = eigvalsh(L) #
            lambda_max = np.max(eigenvalues) #
            print(f"[信息] utils.py: scaled_Laplacian - 使用 eigvalsh 成功计算 lambda_max: {lambda_max}") # 中文信息
        except Exception as e_alt: #
            print(f"[错误] utils.py: scaled_Laplacian - eigvalsh 也失败: {e_alt}。返回单位阵作为安全备选。") # 中文错误
            return -np.identity(N) # 返回一个与公式结构相似的矩阵，尽管这可能不是最佳的

    if lambda_max == 0: # 防止除以零
        print("[警告] utils.py: scaled_Laplacian - 拉普拉斯算子的最大特征值为零。返回 -I。") # 中文警告
        return -np.identity(N) # 对应于 2L/lambda_max - I 如果 lambda_max -> 0 (或接近0)

    return (2 * L) / lambda_max - np.identity(N) #


def cheb_polynomial(L_tilde, K): #
    '''
    计算切比雪夫多项式列表 T_0 到 T_{K-1}
    Parameters
    ----------
    L_tilde: np.ndarray, 标准化的拉普拉斯算子, shape (N, N)
    K: int, 切比雪夫多项式的最大阶数

    Returns
    ----------
    cheb_polynomials: list[np.ndarray], 长度为 K, 从 T_0 到 T_{K-1}
    '''
    if not isinstance(L_tilde, np.ndarray): # 类型检查
        raise TypeError("输入 L_tilde 必须是 NumPy 数组。") #
    if L_tilde.ndim != 2 or L_tilde.shape[0] != L_tilde.shape[1]: # 形状检查
        raise ValueError(f"输入 L_tilde 必须是方阵 (N,N)，但得到形状 {L_tilde.shape}。") #
    if not isinstance(K, int) or K < 1: # K值检查
        raise ValueError(f"K 必须是大于等于1的整数，但得到 {K}。") #

    N = L_tilde.shape[0] #
    if N == 0: return [] # 处理空拉普拉斯算子

    cheb_polynomials = [] #
    if K >= 1: #
        cheb_polynomials.append(np.identity(N)) # T_0 = I
    if K >= 2: #
        cheb_polynomials.append(L_tilde.copy()) # T_1 = L_tilde
    
    for i in range(2, K): #
        # T_k = 2 * L_tilde * T_{k-1} - T_{k-2}
        next_cheb = 2 * L_tilde @ cheb_polynomials[i - 1] - cheb_polynomials[i - 2] # 使用 @ 进行矩阵乘法
        cheb_polynomials.append(next_cheb) #

    return cheb_polynomials #

# --- 以下函数 (calculate_laplacian_matrix, load_graphdata_channel1, 相关评估函数) ---
# --- 在当前 Simple_Trans.py 的流程中不直接使用，但可以保留作为工具或用于其他目的 ---
# --- 我将它们注释掉，以保持此文件的简洁性，并专注于当前流程所需的核心功能 ---

"""
def calculate_laplacian_matrix(adj_mat, mat_type): # 备用的拉普拉斯计算函数
    n_vertex = adj_mat.shape[0]
    id_mat = np.asmatrix(np.identity(n_vertex))

    deg_mat_row = np.asmatrix(np.diag(np.sum(adj_mat, axis=1)))
    deg_mat = deg_mat_row
    adj_mat = np.asmatrix(adj_mat)
    wid_adj_mat = adj_mat + id_mat
    wid_deg_mat = deg_mat + id_mat
    com_lap_mat = deg_mat - adj_mat

    if mat_type == 'id_mat':
        return id_mat
    elif mat_type == 'com_lap_mat':
        return com_lap_mat

    if (mat_type == 'sym_normd_lap_mat') or (mat_type == 'wid_sym_normd_lap_mat') or (mat_type == 'hat_sym_normd_lap_mat'):
        deg_mat_inv_sqrt = fractional_matrix_power(deg_mat, -0.5)
        deg_mat_inv_sqrt[np.isinf(deg_mat_inv_sqrt)] = 0.
        wid_deg_mat_inv_sqrt = fractional_matrix_power(wid_deg_mat, -0.5)
        wid_deg_mat_inv_sqrt[np.isinf(wid_deg_mat_inv_sqrt)] = 0.
        sym_normd_lap_mat = id_mat - np.matmul(np.matmul(deg_mat_inv_sqrt, adj_mat), deg_mat_inv_sqrt)
        sym_max_lambda = max(eigvalsh(sym_normd_lap_mat))
        wid_sym_normd_lap_mat = 2 * sym_normd_lap_mat / sym_max_lambda - id_mat if sym_max_lambda != 0 else -id_mat
        hat_sym_normd_lap_mat = np.matmul(np.matmul(wid_deg_mat_inv_sqrt, wid_adj_mat), wid_deg_mat_inv_sqrt)
        if mat_type == 'sym_normd_lap_mat': return sym_normd_lap_mat
        elif mat_type == 'wid_sym_normd_lap_mat': return wid_sym_normd_lap_mat
        elif mat_type == 'hat_sym_normd_lap_mat': return hat_sym_normd_lap_mat

    elif (mat_type == 'rw_normd_lap_mat') or (mat_type == 'wid_rw_normd_lap_mat') or (mat_type == 'hat_rw_normd_lap_mat'):
        try:
            deg_mat_inv = np.linalg.inv(deg_mat)
        except np.linalg.LinAlgError:
            print(f'[警告] utils.py: calculate_laplacian_matrix - 度矩阵是奇异的。无法计算随机游走归一化拉普拉斯矩阵。')
            return None # 或者返回一个错误指示
        else:
            deg_mat_inv[np.isinf(deg_mat_inv)] = 0.
        # ... (随机游走部分可以类似地完成)
    return None # 如果mat_type无效或计算失败

def load_graphdata_channel1(graph_signal_matrix_filename, num_of_hours, num_of_days, num_of_weeks, DEVICE, batch_size, shuffle=True):
    '''
    此函数为原始DSTAGNN/MSTGCN类模型的数据加载方式，
    在 Simple_Trans.py 的当前流程中不直接使用。
    '''
    # ... (函数体) ...
    print("[信息] load_graphdata_channel1 (原始DSTAGNN流程) 被调用，但在当前 Simple_Trans.py 流程中通常不使用。")
    pass

def compute_val_loss_mstgcn(net, val_loader, criterion, sw, epoch, limit=None):
    '''
    此函数为原始DSTAGNN/MSTGCN类模型的验证损失计算方式，
    在 Simple_Trans.py 的当前流程中不直接使用。
    '''
    # ... (函数体) ...
    print("[信息] compute_val_loss_mstgcn (原始DSTAGNN流程) 被调用，但在当前 Simple_Trans.py 流程中通常不使用。")
    pass

def evaluate_on_test_mstgcn(net, test_loader, test_target_tensor, sw, epoch, _mean, _std):
    '''
    此函数为原始DSTAGNN/MSTGCN类模型的测试评估方式，
    在 Simple_Trans.py 的当前流程中不直接使用。
    '''
    # ... (函数体) ...
    print("[信息] evaluate_on_test_mstgcn (原始DSTAGNN流程) 被调用，但在当前 Simple_Trans.py 流程中通常不使用。")
    pass

def predict_and_save_results_mstgcn(net, data_loader, data_target_tensor, global_step, _mean, _std, params_path, type):
    '''
    此函数为原始DSTAGNN/MSTGCN类模型的预测和结果保存方式 (包括其内部的注意力可视化)，
    在 Simple_Trans.py 的当前流程中不直接使用。
    '''
    # ... (函数体，包含复杂的注意力图绘制) ...
    print("[信息] predict_and_save_results_mstgcn (原始DSTAGNN流程) 被调用，但在当前 Simple_Trans.py 流程中通常不使用。")
    pass
"""