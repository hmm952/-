# 第五章 模型架构

## 5.1 Transformer 模型
### 5.1.1 输入编码

输入序列 $\boldsymbol{u} = [u_1, u_2, ..., u_T]$ 处理流程：
$$
\boldsymbol{v}_t = \text{Embedding}(u_t)
，\boldsymbol{p}_t^{(i)} = 
\begin{cases}
\sin(pos/10000^{2i/H}) & \text{偶索引} \\
\cos(pos/10000^{2i/H}) & \text{奇索引}
\end{cases}，
\boldsymbol{x}_t = \boldsymbol{v}_t + \boldsymbol{p}_t，
$$


### 5.1.2 多头自注意力机制
缩放点积注意力公式：
$$
\text{Attention}(Q,K,V) = \text{softmax}(\frac{QK^T}{\sqrt{d_k}})V，
多头拼接公式：
\text{MultiHead}(Q,K,V) = \text{Concat}(head_1,...,head_h)W^O，
head_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)
$$


### 5.1.3 前馈网络层

$$
\text{FFN}(\boldsymbol{x}) = \max(0, \boldsymbol{x}W_1 + b_1)W_2 + b_2
$$



### 5.1.4 编码器结构
编码器层计算流程：
$$
\boldsymbol{X}'_l = \text{LayerNorm}(\boldsymbol{X}_{l-1} + \text{MHA}(\boldsymbol{X}_{l-1}))，
\boldsymbol{X}_l = \text{LayerNorm}(\boldsymbol{X}'_l + \text{FFN}(\boldsymbol{X}'_l))
$$


### 5.1.5 解码器结构
解码器层计算流程：
$$
\boldsymbol{Y}'_l = \text{LayerNorm}(\boldsymbol{Y}_{l-1} + \text{MaskedMHA}(\boldsymbol{Y}_{l-1}))，
\boldsymbol{Y}''_l = \text{LayerNorm}(\boldsymbol{Y}'_l + \text{CrossMHA}(\boldsymbol{Y}'_l, \boldsymbol{X}_L))，
\boldsymbol{Y}_l = \text{LayerNorm}(\boldsymbol{Y}''_l + \text{FFN}(\boldsymbol{Y}''_l))，
$$


## 5.2 详细配置
### 5.2.1 归一化方法
**LayerNorm**:
$$
\mu = \frac{1}{H}\sum_{i=1}^H x_i，
\sigma = \sqrt{\frac{1}{H}\sum_{i=1}^H (x_i-\mu)^2}，
\text{LayerNorm}(\boldsymbol{x}) = \gamma \odot \frac{\boldsymbol{x}-\mu}{\sigma + \epsilon} + \beta
$$


**RMSNorm**:
$$
\text{RMS}(\boldsymbol{x}) = \sqrt{\frac{1}{H}\sum_{i=1}^H x_i^2}，
\text{RMSNorm}(\boldsymbol{x}) = \frac{\boldsymbol{x}}{\text{RMS}(\boldsymbol{x}) + \epsilon} \odot \gamma
$$
()

**DeepNorm**:
$$
\text{DeepNorm}(\boldsymbol{x}) = \text{LayerNorm}(\alpha \cdot \boldsymbol{x} + \text{Sublayer}(\boldsymbol{x}))
$$


### 5.2.2 归一化模块位置
三种配置对比：
| 类型          | 公式                    | 特点               |
| ------------- | ----------------------- | ------------------ |
| Post-Norm     | LN(x + Sublayer(x))     | 原始配置，易不稳定 |
| Pre-Norm      | x + Sublayer(LN(x))     | 训练稳定           |
| Sandwich-Norm | x + LN(Sublayer(LN(x))) | 平衡方案           |

### 5.2.3 激活函数
常用激活函数实现：
```python
# GELU实现
def gelu(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2/math.pi) * (x + 0.044715*x**3)))

# SwiGLU实现
class SwiGLU(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.wg = nn.Linear(dim, dim)
        self.wu = nn.Linear(dim, dim)
        
    def forward(self, x):
        return F.silu(self.wg(x)) * self.wu(x)
```

### 5.2.4 注意力优化

**Flash Attention 实现示例**：



```python
import torch.nn.functional as F

def flash_attention(q, k, v, mask=None):
    scale = q.size(-1)**0.5
    scores = torch.matmul(q, k.transpose(-2, -1)) / scale
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    attn = F.softmax(scores, dim=-1)
    return torch.matmul(attn, v)
```

## 5.3 主流架构

### 5.3.1 解码器架构（GPT系列）



```python
class GPTBlock(nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()
        self.attn = MultiHeadAttention(dim, n_heads)
        self.ffn = SwiGLU(dim)
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)
        
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x
```

## 5.4 长上下文模型

### 5.4.1 位置插值

RoPE插值公式：
RoPE(xm,m)→RoPE(xm,m/λ)RoPE(*x**m*​,*m*)→RoPE(*x**m*​,*m*/*λ*)

```python
def apply_rope(q, k, scale=1.0):
    # 实现位置插值
    seq_len = q.size(1)
    positions = torch.linspace(0, 1/scale, seq_len)
    # ... RoPE计算逻辑
```

## 5.5 创新架构

### 5.5.1 混合专家（MoE）

```python
class MoELayer(nn.Module):
    def __init__(self, dim, num_experts=8):
        super().__init__()
        self.experts = nn.ModuleList([SwiGLU(dim) for _ in range(num_experts)])
        self.gate = nn.Linear(dim, num_experts)
        
    def forward(self, x):
        gates = F.softmax(self.gate(x), dim=-1)  # [B, T, E]
        outputs = torch.stack([expert(x) for expert in self.experts], dim=-1)
        return torch.einsum('bte,bt...e->bt...', gates, outputs)
```

# 5.2.4 位置编码

## 核心概念
Transformer的自注意力模块具有置换不变性，需通过位置编码注入序列顺序信息[1,3](@ref)。主要实现方式：

## 一、绝对位置编码
### 1.1 基础公式

$$
\boldsymbol{x}_t = \boldsymbol{v}_t + \boldsymbol{p}_t，
$$

其中$\boldsymbol{p}_t$为位置嵌入，$\boldsymbol{v}_t$为词向量

### 1.2 正余弦编码（原始Transformer）

$$
p_{t,2i} = \sin\left(\frac{t}{10000^{2i/d}}\right)，p_{t,2i+1} = \cos\left(\frac{t}{10000^{2i/d}}\right)
$$


其中$d$为嵌入维度，$i$为维度索引[1,4](@ref)

### 1.3 可学习编码（如BERT）
通过可训练参数矩阵生成位置嵌入，适用于固定长度序列

## 二、相对位置编码
### 2.1 Transformer-XL公式

$$
\begin{aligned}
A_{ij} &= \underbrace{\boldsymbol{x}_i\boldsymbol{W}^Q(\boldsymbol{x}_j\boldsymbol{W}^K)^\top}_{\text{内容相关}} \\
&+ \underbrace{\boldsymbol{x}_i\boldsymbol{W}^Q(\boldsymbol{r}_{i-j}\boldsymbol{U}^K)^\top}_{\text{位置偏移}} \\
&+ \underbrace{\boldsymbol{u}(\boldsymbol{x}_j\boldsymbol{W}^K)^\top}_{\text{全局内容偏置}} \\
&+ \underbrace{\boldsymbol{v}(\boldsymbol{r}_{i-j}\boldsymbol{U}^K)^\top}_{\text{全局位置偏置}}
\end{aligned}
$$



### 2.2 T5简化版

$$
A_{ij} = \boldsymbol{q}_i\boldsymbol{k}_j^\top + r_{|i-j|}
$$


其中$r_{|i-j|}$为可学习的相对位置标量[4](@ref)

## 三、旋转位置编码(RoPE)
### 3.1 核心公式
旋转矩阵定义：
$$
\boldsymbol{R}_{\theta,t} = \begin{pmatrix}
\cos t\theta & -\sin t\theta \\
\sin t\theta & \cos t\theta
\end{pmatrix}
$$


注意力计算：
$$
\begin{aligned}
\boldsymbol{q}_i &= \boldsymbol{x}_i\boldsymbol{W}^Q\boldsymbol{R}_{\theta,i} \\
\boldsymbol{k}_j &= \boldsymbol{x}_j\boldsymbol{W}^K\boldsymbol{R}_{\theta,j} \\
A_{ij} &= (\boldsymbol{x}_i\boldsymbol{W}^Q\boldsymbol{R}_{\theta,i-j})(\boldsymbol{x}_j\boldsymbol{W}^K)^\top
\end{aligned}
$$


### 3.2 代码实现（LLaMA）
```python
def rotate_half(x):
    x1 = x[..., :x.shape[-1]//2]
    x2 = x[..., x.shape[-1]//2:]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    cos = cos[position_ids].unsqueeze(1)
    sin = sin[position_ids].unsqueeze(1)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed
```

## 四、ALiBi位置编码

### 4.1 核心公式

*A**ij*=**x***i***W***Q*(**W***K*)⊤**x***j*⊤−*m*⋅∣*i*−*j*∣
其中*m*为头特定的斜率参数，按几何序列设置

### 4.2 代码实现（BLOOM）

```python
def build_alibi_tensor(attention_mask, num_heads, dtype):
    batch_size, seq_length = attention_mask.shape
    closest_power_of_2 = 2**math.floor(math.log2(num_heads))
    base = torch.tensor(2**(-(2**-(math.log2(closest_power_of_2)-3))), 
                      device=attention_mask.device)
    slopes = torch.pow(base, torch.arange(1, 1+closest_power_of_2))
    # ...完整计算过程见原始代码...
    return alibi.reshape(batch_size*num_heads, 1, seq_length).to(dtype)
```

## 五、扩展案例

### 5.1 外推能力验证

使用RoPE在PG19数据集（长度>2048）的测试结果：

|      模型       | 困惑度(1k) | 困惑度(2k) | 困惑度(4k) |
| :-------------: | :--------: | :--------: | :--------: |
| 原始Transformer |    18.2    |    43.1    |    OOM     |
|   RoPE-LLaMA    |    17.8    |    19.3    |    21.7    |

## 六、最佳实践

1. **短序列任务**：优先选择可学习绝对编码（如BERT）
2. **长序列建模**：推荐RoPE或ALiBi
3. **外推需求**：ALiBi在zero-shot外推表现最佳
4. **计算效率**：T5式相对编码适合资源受限场景