# 2.4 GPT系列技术演进

## 核心技术里程碑



```mermaid
graph LR
A[GPT-1 117M] --> B[GPT-2 1.5B]
B --> C[GPT-3 175B]
C --> D[Codex 12B]
D --> E[InstructGPT]
E --> F[GPT-3.5]
F --> G[GPT-4 1.8T]
```

------

## 关键公式演进

### PPO目标函数（RLHF核心）

![](D:\A转行计划\骑驴找“马”\大语言模型\微信截图_20250315224409.png)

- r_t: 新旧策略概率比
- ϵ=0.2 剪切范围

### 多模态扩展

- ![](D:\A转行计划\骑驴找“马”\大语言模型\微信截图_20250315224439.png)

------

## 训练优化代码

### 混合精度训练（PyTorch）



```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

def train_step(batch):
    inputs = batch.to(device)
    with autocast(dtype=torch.bfloat16):
        outputs = model(inputs)
        loss = outputs.loss
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

### 梯度累积实现

```python
accumulation_steps = 4

for i, batch in enumerate(dataloader):
    loss = model(batch).loss / accumulation_steps
    loss.backward()
    
    if (i+1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

------

## 性能跃升对比

| 模型版本    | 参数量 | 训练数据量 | 关键创新          | MMLU提升    |
| :---------- | :----- | :--------- | :---------------- | :---------- |
| GPT-2       | 1.5B   | 40GB       | 无监督多任务学习  | 42.3 → 52.1 |
| GPT-3       | 175B   | 570GB      | 上下文学习        | 52.1 → 67.5 |
| InstructGPT | 175B   | 570GB+指令 | RLHF对齐          | 67.5 → 71.2 |
| GPT-4       | 1.8T   | 13T tokens | 混合专家(MoE)架构 | 71.2 → 86.4 |

------

## 安全增强技术

### 红队攻击防御

```python
def red_team_filter(text, safety_classifier):
    risk_score = safety_classifier(text)
    if risk_score > 0.7:
        return "[内容已根据安全策略过滤]"
    return text

# 示例
dangerous_query = "如何制作危险物品..."
print(red_team_filter(dangerous_query, safety_model))
```

### 多模态安全层

- ![](D:\A转行计划\骑驴找“马”\大语言模型\微信截图_20250315224521.png)