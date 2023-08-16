import torch
import matplotlib.pyplot as plt
# sz = 3
# mask = torch.zeros([sz, sz], dtype=torch.float32)

print(mask)
def test_math_ops():
    """数学运算:单算子，向量到另一个向量，元素级别"""
    x = torch.linspace(-6, 6, 100)
    for ops in ['sin', 'cos', 'tan', 'asin', 'acos', 'atan', 'abs', 'exp', 'log', 'sqrt', 'floor', 'ceil', 'round',
                'trunc', 'neg', 'sigmoid']:
        y = getattr(torch, ops)(x)
        print(y)
        plt.plot(x, y)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(f'torch.{ops}')
        # plt.grid(True)
        plt.show()


def test_tensor_ops():
    """数学运算:二元运算"""
    import matplotlib.colors as mcolors
    cmap_white = mcolors.ListedColormap(['white'])
    r = torch.tensor([[0, 10, 20],
                      [10, 11, 12],
                      [20, 21, 22], ])
    l = torch.tensor([[0.0, 0.1, 0.2],
                      [1.0, 1.1, 1.2],
                      [2.0, 2.1, 2.2], ])

    for ops in ['add', 'sub', 'mul', 'div', 'pow', 'lt', 'le', 'gt', 'ge', 'eq', 'ne']:
        matrix = getattr(torch, ops)(l, r)
        print(matrix)
        fig, (ax1,ax2,ax) = plt.subplots(1,3)
        plot_tensor(matrix, cmap_white, ax)
        plot_tensor(l, cmap_white, ax1)
        plot_tensor(r, cmap_white, ax2)
        plt.show()


def plot_tensor(matrix, cmap_white, ax):
    im = ax.imshow(matrix, cmap=cmap_white, interpolation='nearest')
    # 显示数值
    rows, cols = matrix.shape
    for i in range(rows):
        for j in range(cols):
            val = '{:.2f}'.format(matrix[i, j].item())
            text = ax.text(j, i, val, ha='center', va='center', color='black')
    # 隐藏坐标轴上的刻度和标签
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    # 添加网格线
    ax.set_xticks(torch.arange(-0.5, matrix.shape[1], 1), minor=True)
    ax.set_yticks(torch.arange(-0.5, matrix.shape[0], 1), minor=True)
    # 添加颜色条
    ax.annotate('=', xy=(-1, 1), xytext=(-1, 1), ha='right', va='center', fontsize=12)
    # cbar = ax.figure.colorbar(im)
    plt.grid(True, linewidth=1, color='black', linestyle='--', which='minor')


# test_math_ops()
# test_tensor_ops()
# torch.abs
def test_abs():
    """返回一个张量中每个元素的绝对值"""
    x = torch.linspace(-1, 1, 100)
    y = torch.asin(x)
    print(y)
    plt.plot(x, y)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Absolute Value')
    plt.grid(True)
    plt.show()


# test_abs()
#
# torch.acos
# torch.absolute
# torch.acos
# torch.add
def test_addmv():
    """用于将一个矩阵与一个向量相乘，并将结果与另一个向量相加"""
    matrix = torch.tensor([[1, 2, 3], [4, 5, 6]])
    vector = torch.tensor([7, 8, 9])
    # 定义另一个向量（用于相加）
    bias = torch.tensor([10, 11])
    # 使用torch.addmv进行矩阵向量相乘并相加
    result = torch.addmv(bias, matrix, vector)
    print(result)


def test_addr():
    """用于执行向量相乘的外积操作，并将结果与另外两个向量相加"""
    # 定义两个输入向量
    vector1 = torch.tensor([1, 2, 3])
    vector2 = torch.tensor([4, 5, 6])

    # 定义一个矩阵（用于存储结果）
    matrix = torch.ones(3, 3)

    # 使用torch.addr进行外积操作并相加
    x = torch.addr(matrix, vector1, vector2, out=matrix)
    print(x)
    print(matrix)


# torch.all
# torch.allclose

def test_allclose():
    """用于检查两个张量是否在一定误差范围内相等"""
    # 定义两个输入张量
    tensor1 = torch.tensor([1.0001, 2.0002, 3.0003])
    tensor2 = torch.tensor([1.0, 2.0, 3.0])

    # 使用torch.allclose检查两个张量是否相等
    result = torch.allclose(tensor1, tensor2)

    print(result)


# torch.any
# torch.arange
def test_arange():
    """使用torch.arange创建一个一维张量"""
    # 创建一个从0到4的一维张量
    tensor = torch.arange(5)

    print(tensor)


# torch.arccos
# torch.arcsin
# torch.arctan
# torch.argmax
def test_argmax():
    """使用torch.argmax查找张量中的最大值索引"""
    # 创建一个二维张量
    tensor = torch.tensor([[1, 2, 3],
                           [4, 5, 6],
                           [7, 8, 9]])

    # 沿着指定维度查找最大值索引
    max_indices = torch.argmax(tensor, dim=1)

    print(max_indices)


# torch.argmin
# torch.asin
# torch.as_tensor
def test_as_tensor():
    """将已有数据转换为张量"""
    # 创建一个NumPy数组
    arr = [1, 2, 3, 4, 5]

    # 将NumPy数组转换为张量
    tensor = torch.as_tensor(arr)

    print(tensor)


# torch.atan
# torch.baddbmm
def test_baddbmm():
    """使用 torch.baddbmm 函数进行矩阵乘法和加法
    input：形状为 (B, N, M) 的输入张量，其中 B 是批次大小（batch size），N 和 M 分别是矩阵的行数和列数。
    batch1：形状为 (B, N, K) 的批次矩阵，表示第一个批次的矩阵。
    batch2：形状为 (B, K, M) 的批次矩阵，表示第二个批次的矩阵。
    beta：用于控制输出矩阵的乘法因子，默认为 1。
    alpha：用于控制输入矩阵的乘法因子，默认为 1。
    首先，将输入参数 input 复制到结果张量 output 中。

    对于每个批次中的矩阵：

    将第一个批次矩阵 batch1 和第二个批次矩阵 batch2 进行矩阵乘法操作得到临时张量。
    将临时张量乘以 alpha 的值，并将结果与 output 相加。
    最后，将结果乘以 beta 的值并保存在 output 中。
    返回结果张量 output。
    """
    # 创建输入张量和两个批次的矩阵
    input = torch.randn(3, 5, 4)
    batch1 = torch.randn(3, 5, 2)
    batch2 = torch.randn(3, 2, 4)

    # 执行批量矩阵乘法和加法
    output = torch.baddbmm(input, batch1, batch2)

    print(output)


# torch.bernoulli
# torch.bincount
def test_bincount():
    """使用 torch.bincount 函数统计一维整数张量中每个值的出现次数"""
    # 创建输入张量
    input = torch.tensor([0, 1, 1, 2, 5, 1, 1])

    # 统计值的出现次数
    result = torch.bincount(input)

    print(result)


# torch.broadcast_tensors

def test_broadcast_tensors():
    """使用 torch.broadcast_tensors 函数将输入的张量进行扩展"""
    # 创建输入张量
    tensor1 = torch.tensor([[1, 2, 3], [1, 2, 3]])
    tensor2 = torch.tensor([4, 5, 6])

    # 扩展张量
    result = torch.broadcast_tensors(tensor1, tensor2)

    for tensor in result:
        print(tensor)


# torch.broadcast_to


def test_broadcast_to():
    """使用 torch.broadcast_to 函数将输入张量广播到目标形状"""
    # 创建输入张量
    input = torch.tensor([1, 2, 3])

    # 广播张量到目标形状
    result = torch.broadcast_to(input, (2, 3))

    print(result)


# test_broadcast_to()

# torch.cat

def test_cat():
    """使用 torch.cat 函数将多个张量进行连接"""
    # 创建输入张量
    tensor1 = torch.tensor([[1, 2], [3, 4]])  # 可以理解成在某个维度的括号相消
    tensor2 = torch.tensor([[5, 6], [7, 8]])

    # 在行维度上连接张量
    result = torch.cat((tensor1, tensor2), dim=1)
    # tensor([[1, 2, 5, 6],
    #         [3, 4, 7, 8]])
    result = torch.cat((tensor1, tensor2), dim=0)
    # [[1, 2], [3, 4],
    #  [5, 6], [7, 8]]

    a = torch.tensor([[[1, 2], [3, 4]], [[1, 2], [3, 4]]])
    b = torch.tensor([[[5, 6], [7, 8]], [[5, 6], [7, 8]]])
    result = torch.cat((a, b), dim=2)
    # [[[1, 2,5,6], [3, 4,7,8]], [[1, 2,5,6], [3, 4,7,8]]]
    # dim = 1
    # [[1, 2], [3, 4],[5, 6], [7, 8]],[[1, 2], [3, 4],[5,6], [7, 8]]]
    # dim = 0
    # [[[1, 2], [3, 4]], [[1, 2], [3, 4]]],[[[5, 6], [7, 8]], [[5,6], [7, 8]]]
    print(result)


# torch.ceil
# torch.chain_matmul
# torch.cholesky_inverse
# torch.cholesky_solve
# torch.chunk
def test_chunk():
    """指定的维度将输入张量分割成相等大小的块。如果不能均匀分割，最后一个块的大小会小于其他块的大小。"""
    x = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9])
    chunks = torch.chunk(x, 3, dim=0)
    print(chunks)
    x = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    chunks = torch.chunk(x, 3, dim=0)
    print(chunks)
    x = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    chunks = torch.chunk(x, 3, dim=1)
    print(chunks)


# torch.clamp()
def test_clamp():
    """用于对张量进行限幅操作"""
    # 定义输入张量
    tensor = torch.tensor([1, 2, 3, 4, 5])

    # 使用 torch.clamp 对张量进行限幅操作
    clamped_tensor = torch.clamp(tensor, min=2, max=4)

    print(clamped_tensor)
    # 打印结果: tensor([2, 2, 3, 4, 4])


# test_clamp()

# torch.conj
# torch.corrcoef


def test_corrcoef():
    """用于计算张量的相关系数矩阵"""
    # 定义一个输入张量
    tensor = torch.tensor([[1, 2, 3], [4, 5, 9]])

    # 使用 torch.corrcoef 计算相关系数矩阵
    corr_matrix = torch.corrcoef(tensor)

    print(corr_matrix)
    # 打印结果: tensor([[1., 1.],
    #                  [1., 1.]])


# test_corrcoef()

# torch.cos
# torch.cosh
# torch.count_nonzero
# torch.cov
# torch.det
# torch.diag


def test_diag_create():
    """使用给定对角线元素创建对角矩阵"""
    # 定义一个输入张量
    diagonal = torch.tensor([1, 2, 3, 4])

    # 使用 torch.diag 创建对角矩阵
    diag_matrix = torch.diag(diagonal)

    print(diag_matrix)
    # 打印结果:
    # tensor([[1, 0, 0, 0],
    #         [0, 2, 0, 0],
    #         [0, 0, 3, 0],
    #         [0, 0, 0, 4]])


# test_diag_create()


def test_diag_extract():
    """从方阵中提取对角元素"""
    # 定义一个输入方阵
    matrix = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

    # 使用 torch.diag 提取对角元素
    diagonal = torch.diag(matrix)

    print(diagonal)
    # 打印结果: tensor([1, 5, 9])


# test_diag_extract()
# torch.diagflat


def test_diagflat():
    """使用一维张量创建对角矩阵"""
    # 定义一个输入一维张量
    diagonal = torch.tensor([1, 2, 3, 4])

    # 使用 torch.diagflat 创建对角矩阵
    diag_matrix = torch.diagflat(diagonal)

    print(diag_matrix)
    # 打印结果:
    # tensor([[1, 0, 0, 0],
    #         [0, 2, 0, 0],
    #         [0, 0, 3, 0],
    #         [0, 0, 0, 4]])


# test_diagflat()

# torch.diff
# torch.dist


def test_dist():
    """计算两个张量之间的欧氏距离"""
    # 定义两个输入张量
    x = torch.tensor([1.0, 2, 3])
    y = torch.tensor([4.0, 5, 6])

    # 使用 torch.dist 计算欧氏距离
    distance = torch.dist(x, y)

    print(distance)
    # 打印结果: tensor(5.1962)


# test_dist()

# torch.div
# torch.divide


def test_div():
    """执行张量之间的元素级除法"""
    # 定义两个输入张量
    x = torch.tensor([10, 20, 30])
    y = torch.tensor([2, 4, 6])

    # 使用 torch.div 执行元素级除法
    result = torch.div(x, y)

    print(result)
    # 打印结果: tensor([ 5,  5,  5])


# test_div()

# torch.empty
# torch.empty_like
# torch.equal
# torch.erf
# torch.exp
# torch.eye
# torch.flatten
# torch.flip


def test_flip():
    """反转张量的维度"""
    # 定义输入张量
    x = torch.tensor([[1, 2, 3],
                      [4, 5, 6]])

    # 使用 torch.flip 反转张量的维度
    result = torch.flip(x, dims=[0, 1])

    print(result)
    # 打印结果: tensor([[6, 5, 4],
    #                   [3, 2, 1]])


# test_flip()

# torch.fliplr
# torch.flipud
# torch.floor
# torch.floor_divide
# torch.fmod


def test_fmod():
    """计算元素级别的fu浮点数取模"""
    # 定义输入张量
    x = torch.tensor([12.5, 15.3, 20.8])
    y = torch.tensor([3.2, 2.5, 4.1])

    # 使用 torch.fmod 计算元素级别的取模
    result = torch.fmod(x, y)

    print(result)
    # 打印结果: tensor([0.5000, 0.3000, 0.5000])


# test_fmod()

# torch.from_numpy
# torch.full
# torch.full_like
# torch.gather


def test_gather():
    """按照给定维度索引从输入张量中收集值"""
    # 定义输入张量
    x = torch.tensor([[1, 2],
                      [3, 4],
                      [5, 6]])

    # 定义索引张量
    indices = torch.tensor([[0, 1],
                            [1, 0],
                            [0, 0]])

    # 使用 torch.gather 进行值的收集
    result = torch.gather(x, dim=1, index=indices)

    print(result)
    # 打印结果: tensor([[1, 2],
    #                   [4, 3],
    #                   [5, 5]])


# test_gather()

# torch.imag
# torch.index_select
# 


def test_index_select():
    """按照给定索引从输入张量中选择元素"""
    # 定义输入张量
    x = torch.tensor([[1, 2, 3],
                      [4, 5, 6],
                      [7, 8, 9]])

    # 定义索引张量
    indices = torch.tensor([0, 2])

    # 使用 torch.index_select 进行元素选择
    result = torch.index_select(x, dim=1, index=indices)

    print(result)
    # 打印结果: tensor([[1, 3],
    #                   [4, 6],
    #                   [7, 9]])


# test_index_select()

# torch.isclose


def test_isclose():
    """检查两个张量或标量是否在元素级别上接近"""
    # 定义输入张量
    x = torch.tensor([1.0, 2.0, 3.0])
    y = torch.tensor([1.1, 2.2, 2.9])

    # 使用 torch.isclose 进行接近性检查
    result = torch.isclose(x, y, rtol=0.1, atol=0.1)
    # rtol 是相对容忍度（relative tolerance），表示相对误差的最大允许值，而 atol 是绝对容忍度（absolute tolerance），表示绝对误差的最大允许值。
    print(result)
    # 打印结果: tensor([True, True, True])


# test_isclose()

# torch.isfinite
# torch.isinf
# torch.isnan


def test_isfinite_isinf_isnan():
    """检查张量中的元素是否为有限数、无穷大或NaN"""
    # 定义输入张量
    x = torch.tensor([1.0, float('inf'), float('-inf'), float('nan')])

    # 使用 torch.isfinite 判断有限数
    is_finite = torch.isfinite(x)
    print(is_finite)
    # 打印结果: tensor([ True, False, False, False])

    # 使用 torch.isinf 判断无穷大
    is_inf = torch.isinf(x)
    print(is_inf)
    # 打印结果: tensor([False,  True,  True, False])

    # 使用 torch.isnan 判断NaN
    is_nan = torch.isnan(x)
    print(is_nan)
    # 打印结果: tensor([False, False, False,  True])


# test_isfinite_isinf_isnan()

# torch.istft
# torch.is_complex
# torch.is_floating_point
# torch.is_tensor
# torch.kthvalue


def test_kthvalue():
    """找到输入张量中第 k 小的值和对应的索引"""
    x = torch.tensor([9, 2, 5, 1, 7, 4, 8, 3, 6])

    # 使用 torch.kthvalue 找到第 3 小的值和对应的索引
    value, indices = torch.kthvalue(x, k=3)

    print(value.item())  # 打印结果: 3
    print(indices.item())  # 打印结果: 7


# test_kthvalue()

# torch.linspace


def test_linspace():
    """生成等间隔的一维张量"""
    start = 0.0
    end = 1.0
    num_steps = 5

    # 使用 torch.linspace 生成从起始值到结束值之间的等间隔一维张量
    result = torch.linspace(start, end, steps=num_steps)

    print(result)  # 打印结果: tensor([0.0000, 0.2500, 0.5000, 0.7500, 1.0000])


# test_linspace()

# torch.load
# torch.log

# torch.logical_and
# torch.logical_not
# torch.logical_or
# torch.logical_xor
# torch.masked_select


def test_masked_select():
    """根据掩码条件选择输入张量中的元素"""
    x = torch.tensor([1, 2, 3, 4, 5])
    mask = torch.tensor([True, False, True, False, True])

    # 使用 torch.masked_select 根据掩码条件选择输入张量中的元素
    result = torch.masked_select(x, mask)

    print(result)  # 打印结果: tensor([1, 3, 5])


# test_masked_select()

# torch.mm
# torch.matmul 矩阵乘法

# torch.max
# torch.median
# torch.min
# 


def test_max_median_min():
    """计算张量的最大值、中位数和最小值"""
    x = torch.tensor([[1, 2, 3],
                      [4, 5, 6],
                      [7, 8, 9]])

    # 计算沿指定维度的最大值
    max_value = torch.max(x, dim=1)

    # 计算沿指定维度的中位数
    median_value = torch.median(x, dim=0)

    # 计算沿指定维度的最小值
    min_value = torch.min(x, dim=1)

    print("最大值:", max_value.values)  # 打印结果: tensor([3, 6, 9])
    print("最大值索引:", max_value.indices)  # 打印结果: tensor([2, 2, 2])

    print("中位数:", median_value.values)  # 打印结果: tensor([4, 5, 6])
    print("中位数索引:", median_value.indices)  # 此函数没有返回索引

    print("最小值:", min_value.values)  # 打印结果: tensor([1, 4, 7])
    print("最小值索引:", min_value.indices)  # 打印结果: tensor([0, 0, 0])


# test_max_median_min()


# torch.moveaxis


def test_moveaxis():
    """交换张量的维度顺序"""
    # 创建一个三维张量
    x = torch.zeros((2, 3, 4))

    # 使用 torch.moveaxis 交换维度顺序，0变2,1变0,2变1
    y = torch.moveaxis(x, [0, 1, 2], [2, 0, 1])

    print("原始张量形状:", x.shape)  # 打印结果: torch.Size([2, 3, 4])
    print("交换维度后的张量形状:", y.shape)  # 打印结果: torch.Size([3, 4, 2])


def test_transpose():
    """使用 torch.transpose 进行维度交换"""
    x = torch.zeros((2, 3, 4))

    y = torch.transpose(x, 2, 0)
    z = torch.transpose(x, 1, 2)

    print("原始张量形状:", x.shape)  # 打印结果: torch.Size([2, 3, 4])
    print("交换维度后的张量形状(y):", y.shape)  # 打印结果: torch.Size([4, 3, 2])
    print("交换维度后的张量形状(z):", z.shape)  # 打印结果: torch.Size([2, 4, 3])


# torch.movedim
# torch.mul

# torch.multiply


# torch.multinomial
def test_multinomial():
    """从给定概率分布中进行多项式抽样"""
    probs = torch.tensor([0.1, 0.2, 0.3, 0.4])

    samples = torch.multinomial(probs, 5, replacement=True)

    print("概率分布:", probs)  # 打印结果: tensor([0.1000, 0.2000, 0.3000, 0.4000])
    print("抽样结果:", samples)  # 打印结果: tensor([2, 3, 2, 2, 0])


# test_multinomial()

# torch.mv


def test_mv():
    """执行矩阵-向量乘法"""
    A = torch.tensor([[1.0, 2, 3], [4, 5, 6]])
    b = torch.tensor([0.1, 0.2, 0.3])

    c = torch.mv(A, b)

    print("矩阵 A:", A)  # 打印结果: tensor([[1, 2, 3],
    #              [4, 5, 6]])
    print("向量 b:", b)  # 打印结果: tensor([0.1000, 0.2000, 0.3000])
    print("A * b:", c)  # 打印结果: tensor([1.4000, 3.5000])


# test_mv()

# torch.nanmedian
# torch.narrow


def test_narrow():
    """在张量的指定维度上选择切片"""
    x = torch.tensor([[1, 2, 3, 4],
                      [5, 6, 7, 8],
                      [9, 10, 11, 12]])

    # 在第1维上选择从索引1开始的2个元素
    y = torch.narrow(x, dim=1, start=1, length=2)

    print("原始张量 x:")
    print(x)
    print("切片结果 y:")
    print(y)


# test_narrow()

# torch.nonzero
# 


def test_nonzero():
    """找到张量中非零元素的索引"""
    x = torch.tensor([[0, 1, 0],
                      [2, 0, 3],
                      [0, 4, 0]])

    indices = torch.nonzero(x)

    print("原始张量 x:")
    print(x)
    print("非零元素的索引:")
    print(indices)


# test_nonzero()

# torch.normal


def test_normal():
    """生成符合正态分布的随机数"""
    mean = 0.0
    std = 1.0

    # 生成一个形状为 (3, 4) 的张量，其中的元素服从正态分布
    x = torch.normal(mean, std, size=(3, 4))

    print("生成的随机数:")
    print(x)


# test_normal()

# torch.numel


def test_numel():
    """计算张量中元素的总数"""
    x = torch.tensor([[1, 2, 3],
                      [4, 5, 6]])

    num_elements = torch.numel(x)

    print("原始张量 x:")
    print(x)
    print("元素的总数:")
    print(num_elements)


# test_numel()

# torch.ones
# torch.ones_like
# torch.permute


def test_permute():
    """对张量进行维度重排,注意形状是不会发生任何变化的"""
    x = torch.tensor([[[1, 2],
                       [3, 4]],
                      [[5, 6],
                       [7, 8]]])
    # [[[1,5],[3,7]],[[2,6],[4,8]]]
    permuted = x.permute(2, 1, 0)
    # 参数 (2, 0, 1) 指定了新的维度顺序，其中 2 表示原来的维度 2 移到最前面，0 表示原来的维度 0 移到中间，1 表示原来的维度 1 移到最后面。
    print("原始张量 x:")
    print(x)
    print("重排后的张量:")
    print(permuted)


# test_permute()

# torch.pinverse 阵的伪逆（pseudo-inverse
# torch.pow
# torch.prod 元素积
# torch.rand
# torch.randint
# torch.randn
# torch.randperm


def test_randperm():
    """生成一个随机排列的整数序列"""
    length = 5

    randperm_tensor = torch.randperm(length)

    print("生成的随机整数序列:")
    print(randperm_tensor)


# test_randperm()

# torch.ravel
# torch.real
# torch.reciprocal


def test_reciprocal():
    """计算张量中每个元素的倒数"""
    tensor = torch.tensor([-2, 0, 4], dtype=torch.float32)

    reciprocal_tensor = torch.reciprocal(tensor)
    # 如果张量中包含零元素，则倒数为无穷大(inf)
    print("计算倒数后的张量:")
    print(reciprocal_tensor)


# test_reciprocal()

# torch.remainder


def test_remainder():
    """计算两个张量或一个张量和一个标量之间的元素级取模运算"""
    tensor1 = torch.tensor([5, 9, 12])
    tensor2 = torch.tensor([2, 3, 4])

    remainder_tensor = torch.remainder(tensor1, tensor2)
    # 请注意，对于负数的取模运算，torch.remainder 的行为与 Python 语言的取模运算符 % 以及 NumPy 中的 np.remainder 函数有所不同。
    # 具体来说，torch.remainder 的结果始终具有与除数相同的符号，而不是与被除数相同的符号。
    print("取模运算后的张量:")
    print(remainder_tensor)


# test_remainder()

# torch.reshape


def test_reshape():
    """改变张量的形状"""
    tensor = torch.tensor([[1, 2, 3], [4, 5, 6]])

    reshaped_tensor = torch.reshape(tensor, (3, 2))

    print("改变形状后的张量:")
    print(reshaped_tensor)


# test_reshape()


def test_reshape_negative_one():
    """改变张量的形状，并通过-1自动推断维度大小"""
    tensor = torch.tensor([[1, 2, 3], [4, 5, 6]])

    reshaped_tensor = torch.reshape(tensor, (-1, 3))

    print("改变形状后的张量:")
    print(reshaped_tensor)


# test_reshape_negative_one()

# torch.roll

def test_roll():
    """沿指定维度对张量进行循环滚动"""
    tensor = torch.tensor([[1, 2, 3], [4, 5, 6]])

    rolled_tensor = torch.roll(tensor, shifts=1, dims=1)

    print("滚动后的张量:")
    print(rolled_tensor)


# test_roll()

# torch.round
# torch.rsqrt


def test_rsqrt():
    """计算张量元素的倒数的平方根"""
    tensor = torch.tensor([4.0, 16.0, 36.0])

    result = torch.rsqrt(tensor)

    print("计算结果:")
    print(result)


# test_rsqrt()

# torch.save
# torch. # fixme

# torch.sign

def test_sign():
    """计算输入张量的符号函数并进行输出"""
    x = torch.tensor([-2.3, 0, 5.6])

    result = torch.sign(x)

    print("输入张量:")
    print(x)
    print("\n符号函数结果:")
    print(result)


# test_sign()

# torch.sin
# torch.sinh

# torch.split
# torch.sqrt
# torch.squeeze


def test_squeeze():
    """对张量进行维度压缩并输出结果"""
    # 创建一个具有尺寸为1的维度的张量
    x = torch.tensor([[[1, 2, 3]]])

    # 使用 torch.squeeze 进行维度压缩
    result = torch.squeeze(x)

    # 打印结果
    print("压缩前的张量:")
    print(x)
    print("\n压缩后的张量:")
    print(result)


# test_squeeze()

# torch.stack


def test_stack():
    """对张量进行拼接并输出结果"""
    # 创建两个示例张量
    x1 = torch.tensor([1, 2, 3])
    x2 = torch.tensor([4, 5, 6])

    # 使用 torch.stack 进行拼接
    result = torch.stack([x1, x2], dim=1)

    # 打印结果
    print("拼接前的张量:")
    print(x1)
    print(x2)
    print("\n拼接后的张量:")
    print(result)
    # torch.stack 还可以接受一个 dim 参数，用于指定在哪个维度进行拼接，默认为 0。可以根据需要调整 dim 参数来实现不同的拼接方式。


# test_stack()

# torch.std_mean


def test_std_mean():
    """计算张量的标准差和平均值并输出结果"""
    # 创建一个示例张量
    x = torch.tensor([1, 2, 3, 4, 5])

    # 使用 torch.std_mean 计算标准差和平均值
    std, mean = torch.std_mean(x)

    # 打印结果
    print("张量:")
    print(x)
    print("\n标准差:", std.item())
    print("平均值:", mean.item())


# test_std_mean()

# torch.sub
# torch.subtract
# torch.sum
# torch.take


def test_take():
    """按索引从张量中取出元素并输出结果"""
    # 创建一个示例张量
    x = torch.tensor([[1, 2, 3],
                      [4, 5, 6],
                      [7, 8, 9]])

    # 按索引从张量中取出元素
    indices = torch.tensor([0, 4, 8])
    result = torch.take(x, indices)

    # 打印结果
    print("原始张量:")
    print(x)
    print("\n按索引取出的元素:")
    print(result)


# test_take()

# torch.tensor
# torch.tile

def test_tile():
    """在指定维度上重复张量的元素并输出结果,如同贴地板"""
    # 创建一个示例张量
    x = torch.tensor([[1, 2],
                      [3, 4]])

    # 在指定维度上重复张量的元素
    result = torch.tile(x, (2, 3))

    # 打印结果
    print("原始张量:")
    print(x)
    print("\n重复元素后的张量:")
    print(result)


# test_tile()

# torch.trace


def test_trace():
    """计算二维张量的迹并输出结果，对角线和"""
    # 创建一个示例张量
    x = torch.tensor([[1, 2, 3],
                      [4, 5, 6],
                      [7, 8, 9]])

    # 计算二维张量的迹
    result = torch.trace(x)

    # 打印结果
    print("原始张量:")
    print(x)
    print("\n迹的值:")
    print(result)


# test_trace()

# torch.transpose


def test_transpose():
    """交换张量的维度并输出结果"""
    # 创建一个示例张量
    x = torch.tensor([[1, 2, 3],
                      [4, 5, 6]])

    # 交换张量的维度
    result = torch.transpose(x, 0, 1)

    # 打印结果
    print("原始张量:")
    print(x)
    print("\n交换维度后的张量:")
    print(result)


# test_transpose()

# torch.unbind


def test_unbind():
    """按维度拆分张量并输出结果"""
    # 创建一个示例张量
    x = torch.tensor([[1, 2, 3],
                      [4, 5, 6]])

    # 按维度拆分张量
    result = torch.unbind(x, dim=1)

    # 打印结果
    print("原始张量:")
    print(x)
    print("\n按维度拆分后的张量:")
    for tensor in result:
        print(tensor)


# test_unbind()


# torch.unique
# torch.unique_consecutive


def test_unique():
    """查找张量的唯一值并输出结果"""
    # 创建一个示例张量
    x = torch.tensor([1, 2, 3, 2, 1, 4, 4])

    # 查找张量的唯一值
    result = torch.unique(x)

    # 打印结果
    print("原始张量:")
    print(x)
    print("\n唯一值结果:")
    print(result)


# test_unique()

# torch.unsqueeze


def test_unsqueeze():
    """在指定维度上扩展张量的维度"""
    # 创建一个示例张量
    x = torch.tensor([1, 2, 3])

    # 在维度 0 上扩展张量
    result = torch.unsqueeze(x, dim=0)

    # 打印结果
    print("原始张量:")
    print(x)
    print("\n扩展后的张量:")
    print(result)


# test_unsqueeze()

# torch.var_mean
# torch.view_as_complex
# torch.view_as_real
# torch.zeros
# torch.zeros_like
# torch.meshgrid


def test_meshgrid():
    """生成多维网格坐标"""
    # 创建示例输入张量
    x = torch.tensor([1, 2, 3])
    y = torch.tensor([4, 5, 6])

    # 生成多维网格坐标
    X, Y = torch.meshgrid(x, y)

    # 打印结果
    print("X 坐标:")
    print(X)
    print("\nY 坐标:")
    print(Y)
    # X 坐标:
    # tensor([[1, 1, 1],
    #         [2, 2, 2],
    #         [3, 3, 3]])
    #
    # Y 坐标:
    # tensor([[4, 5, 6],
    #         [4, 5, 6],
    #         [4, 5, 6]])


# test_meshgrid()

# torch.sigmoid
# torch.set_default_dtype
# torch.get_default_dtype
# torch.t 同 transpose
# torch.where


def test_where():
    """根据条件选择张量中的元素"""
    # 创建示例张量和条件张量
    condition = torch.tensor([[True, False], [False, True]])
    A = torch.tensor([[1, 2], [3, 4]])
    B = torch.tensor([[5, 6], [7, 8]])

    # 根据条件选择张量中的元素
    result = torch.where(condition, A, B)

    # 打印结果
    print("条件张量:")
    print(condition)
    print("\n输入张量A:")
    print(A)
    print("\n输入张量B:")
    print(B)
    print("\n根据条件选择的结果:")
    print(result)


# test_where()


# torch.manual_seed
# torch.no_grad
# torch.set_grad_enabled
# torch.diag_embed
# torch.is_grad_enabled
# torch.nansum

def test_nansum():
    """计算张量中的非 NaN 值的和"""
    # 创建示例张量
    x = torch.tensor([[1.0, 2.0, float("nan")], [3.0, float("nan"), 4.0]])

    # 计算非 NaN 值的和
    result = torch.nansum(x)

    # 打印结果
    print("原始张量:")
    print(x)
    print("\n非 NaN 值的和:")
    print(result.item())


# test_nansum()

# torch.symeig


def test_symeig():
    """计算对称矩阵的特征值和特征向量"""
    # 创建示例对称矩阵
    x = torch.tensor([[1.0, 2.0, 3.0],
                      [2.0, 4.0, 5.0],
                      [3.0, 5.0, 6.0]])

    # 计算特征值和特征向量
    eigenvalues, eigenvectors = torch.symeig(x, eigenvectors=True)

    # 打印结果
    print("原始对称矩阵:")
    print(x)
    print("\n特征值:")
    print(eigenvalues)
    print("\n特征向量:")
    print(eigenvectors)


# test_symeig()

# torch.get_rng_state
# torch.heaviside #fixme


# torch.is_nonzero
# torch.polar


def test_polar():
    """使用 torch.polar()"""
    # 创建示例张量
    magnitude = torch.tensor([2.0, 1.0, 3.0])
    phase = torch.tensor([0.0, 3.1416 / 2, 3.1416])

    # 执行极坐标转换
    result = torch.polar(magnitude, phase)

    # 打印结果
    print("magnitude:")
    print(magnitude)
    print("\nphase:")
    print(phase)
    print("\npolar 结果:")
    print(result)


# test_polar()

# torch.rand_like
# torch.row_stack

# torch.seed
# torch.set_printoptions
# torch.set_rng_state

# torch.swapaxes
# torch.swapdims


def test_swapaxes():
    """使用 torch.swapaxes()"""
    # 创建示例张量
    tensor = torch.tensor([[1, 2, 3], [4, 5, 6]])

    # 执行轴交换
    result = torch.swapaxes(tensor, 0, 1)

    # 打印结果
    print("tensor:")
    print(tensor)
    print("\nswapaxes 结果:")
    print(result)


# test_swapaxes()


def test_swapdims():
    """使用 torch.swapdims()"""
    # 创建示例张量
    tensor = torch.tensor([[1, 2, 3], [4, 5, 6]])

    # 执行维度交换
    result = torch.swapdims(tensor, 0, 1)

    # 打印结果
    print("tensor:")
    print(tensor)
    print("\nswapdims 结果:")
    print(result)


# test_swapdims()


# torch.vstack
# torch.clip 同clamp

# torch.degrad
# torch.digamma
# torch.erfinv
# torch.expm

# torch.fix 向零取整
# torch.trunc 截断取整

# torch.acosh
# torch.arccosh
# torch.randint_like
# torch.dsplit
# torch.logspace
# torch.gcd

def test_gcd():
    """最大公约数"""
    # 创建示例张量
    input = torch.tensor([12, 24, 36])
    other = torch.tensor([8, 16, 28])

    # 计算最大公约数
    result = torch.gcd(input, other)

    # 打印结果
    print("input:")
    print(input)
    print("\nother:")
    print(other)
    print("\n最大公约数:")
    print(result)


# test_gcd()

# torch.histc


def test_histc():
    """直方图"""
    # 创建示例张量
    input = torch.tensor([1.0, 2.0, 1.5, 3.2, 2.5])

    # 计算直方图
    hist = torch.histc(input, bins=4, min=1.0, max=3.5)

    # 打印结果
    print("input:")
    print(input)
    print("\nhist:")
    print(hist)


# test_histc()

# torch.kron 一种积
# torch.lcm 最小公倍数
# torch.logcumsumexp
# torch.renorm 重新正则化
# torch.repeat_interleave


def test_repeat_interleave():
    """input（输入张量）、repeats（要重复的次数）和可选的 dim 参数（指定在哪个维度上进行重复）。
    它返回一个新的张量，其中每个元素都被重复了指定次数。"""
    # 创建示例张量
    input = torch.tensor([1, 2, 3])
    repeats = torch.tensor([2, 3, 1])

    # 对张量进行重复
    result = torch.repeat_interleave(input, repeats)

    # 打印结果
    print("input:")
    print(input)
    print("\nrepeats:")
    print(repeats)
    print("\n重复后的张量:")
    print(result)


# test_repeat_interleave()

# torch.searchsorted


def test_searchsorted():
    """sorted_sequence（已排序的一维张量）和 values（要搜索的一维张量或标量）。
    它返回一个新的张量，表示每个 values 元素在 sorted_sequence 中应该被插入的索引位置。"""
    # 创建示例张量
    sorted_sequence = torch.linspace(1, 10, 10)
    values = torch.tensor([3.2, 6.7, 9.5])

    # 在已排序的张量中搜索插入位置
    result = torch.searchsorted(sorted_sequence, values)

    # 打印结果
    print("sorted_sequence:")
    print(sorted_sequence)
    print("\nvalues:")
    print(values)
    print("\n插入位置:")
    print(result)


# test_searchsorted()

# torch.frac


def test_frac():
    """用于计算张量中元素的小数部分。"""
    # 创建示例张量
    input = torch.tensor([3.14159, -2.71828, 2.5])

    # 计算张量元素的小数部分
    result = torch.frac(input)

    # 打印结果
    print("input:")
    print(input)
    print("\n小数部分:")
    print(result)


# test_frac()

# torch.lerp
# torch.lgamma
# torch.logit
# torch.nan_to_num
# torch.neg
# torch.negative
# torch.raddeg
# torch.fmax
# torch.fmin

# torch.greater


def test_greater():
    """使用 torch.greater()"""
    # 创建示例张量
    input = torch.tensor([1, 2, 3])
    other = torch.tensor([2, 1, 3])

    # 执行元素级别的大于比较操作
    result = torch.greater(input, other)
    result1 = torch.gt(input, other)
    # 打印结果
    print("input:")
    print(input)
    print("\nother:")
    print(other)
    print("\n大于比较结果:")
    print(result)
    print(result1)


# test_greater()


# torch.le
# torch.less
# torch.less_equal
# torch.lt

# torch.maximum
# torch.minimum
# torch.addmm
# torch.bmm
# torch.cholesky
# torch.tensordot

# torch.tril
# torch.tril_indices
# torch.triu
# torch.triu_indices

# torch.dot = inner
# torch.ger = outer
# torch.inner


def test_inner():
    """使用 torch.inner()"""
    # 创建示例向量
    a = torch.tensor([1, 2, 3])
    b = torch.tensor([4, 5, 6])

    # 计算内积
    result = torch.inner(a, b)

    # 打印结果
    print("a:")
    print(a)
    print("\nb:")
    print(b)
    print("\ninner 结果:")
    print(result)


# torch.inverse

def test_inverse():
    """使用 torch.inverse()"""
    # 创建示例矩阵
    A = torch.tensor([[1, 2], [3, 4]])

    # 计算逆矩阵
    A_inv = torch.inverse(A)

    # 打印结果
    print("A:")
    print(A)
    print("\nA 的逆矩阵:")
    print(A_inv)


# test_inverse()

# torch.lu
# torch.lu_unpack
# torch.matrix_power
# torch.outer

def test_outer():
    """外积"""
    # 创建示例向量
    a = torch.tensor([1, 2, 3])
    b = torch.tensor([4, 5, 6])

    # 计算外积
    result = torch.outer(a, b)

    # 打印结果
    print("a:")
    print(a)
    print("\nb:")
    print(b)
    print("\nouter 结果:")
    print(result)


# test_outer()

# torch.qr
# torch.cross
def test_cross():
    """使用 torch.cross()"""
    # 创建示例张量
    a = torch.tensor([1, 2, 3])
    b = torch.tensor([4, 5, 6])

    # 计算叉积
    result = torch.cross(a, b)

    # 打印结果
    print("a:")
    print(a)
    print("\nb:")
    print(b)
    print("\ncross 结果:")
    print(result)


# test_cross()

# torch.clone

def test_clone():
    """使用 torch.clone()"""
    # 创建示例张量
    tensor = torch.tensor([[1, 2, 3], [4, 5, 6]])

    # 执行克隆操作
    cloned_tensor = tensor.clone()

    # 修改克隆张量
    cloned_tensor[0, 0] = 9

    # 打印原始张量和克隆张量
    print("tensor:")
    print(tensor)
    print("\ncloned_tensor:")
    print(cloned_tensor)


# test_clone()

# torch.cumprod
# torch.cumsum


def test_cumsum():
    """使用 torch.cumsum()"""
    # 创建示例张量
    input = torch.tensor([[1, 2, 3], [4, 5, 6]])

    # 在指定维度上计算累积和
    result = torch.cumsum(input, dim=1)

    # 打印结果
    print("input:")
    print(input)
    print("\n累积和:")
    print(result)
    #     [[ 1,  3,  6],
    #      [ 4,  9, 15]]


# test_cumsum()


# torch.ne
# torch.not_equal
# torch.sort
# torch.stft
# torch.topk


def test_topk():
    """input（输入张量）、k（要获取的元素个数）、dim（指定的维度，默认为 None，表示在整个张量中获取）、
    largest（是否获取最大的 k 个元素，默认为 True）、sorted（是否以降序排序输出，默认为 True）和
    可选的 out 参数（输出张量或元组）。
    它返回一个包含两个张量的元组，第一个张量表示最大（或最小）的 k 个元素，第二个张量表示这些元素在原始张量中的索引。
    """

    # 创建示例张量
    input = torch.tensor([[5, 2, 9, 1], [7, 3, 8, 4]])

    # 获取最大的两个元素及其索引
    values, indices = torch.topk(input, k=2)

    # 打印结果
    print("input:")
    print(input)
    print("\n最大的两个元素:")
    print(values)
    print("\n最大的两个元素的索引:")
    print(indices)

# test_topk()

# torch.frexp
# torch.nanmean
# torch.take_along_dim
# torch.geqrf
# torch.bitwise_right_shift
# torch.is_conj

# torch.vsplit
# torch.hsplit
# torch.histogram
