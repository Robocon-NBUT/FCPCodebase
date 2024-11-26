import numpy as np


def run_mlp(obs: np.ndarray, weights: list, activation_function: str = "tanh"):
    """
    使用 numpy 运行 (MLP)

    Parameters
    ----------
    obs: ndarray
        一个 float32 类型的数组，包含神经网络的输入数据
    weights: list
        包含MLP各层权重的列表，每层权重由一对 (bias, kernel) 组成
    activation_function: str
        hidden layers 使用的激活函数类型
        将其设置为 none 来禁用激活函数
    """

    # 确保观测值obs的数据类型为float32
    obs = obs.astype(np.float32, copy=False)
    out = obs

    # 遍历所有 hidden layer
    for w in weights[:-1]:
        out = np.matmul(w[1], out) + w[0]
        if activation_function == "tanh":
            np.tanh(out, out=out)
        elif activation_function != "none":
            raise NotImplementedError
    return np.matmul(weights[-1][1], out) + weights[-1][0]  # final layer
