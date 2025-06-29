# WandB Integration for Answer Generation

这个文档解释了如何使用集成了WandB日志记录功能的答案生成脚本。

## 新增功能

### 1. 实时评估
- **Validity Score**: 使用ReasonEval模型评估推理步骤的有效性
- **Redundancy Score**: 评估推理步骤中的冗余程度
- **Shepherd Score**: 使用Math-Shepherd模型评估解答的正确性

### 2. Token概率记录
- 记录生成过程中top-k tokens的概率分布
- 计算平均top-1和top-5概率
- 支持vLLM和HuggingFace两种推理引擎

### 3. WandB集成
- 自动记录每个样本的评估指标
- 按温度聚合统计指标
- 记录token概率分布统计

## 使用方法

### 基本用法（不启用评估）
```bash
python answer_generation.py \
    --dataset_name "hybrid_reasoning" \
    --models "Abel-7B-002" \
    --temperatures "0.1" "0.3" "0.6" \
    --logger \
    --project "My-Experiment"
```

### 启用实时评估
```bash
python answer_generation.py \
    --dataset_name "hybrid_reasoning" \
    --models "Abel-7B-002" \
    --temperatures "0.1" "0.3" "0.6" \
    --logger \
    --project "My-Experiment" \
    --enable_evaluation \
    --reasoneval_path "GAIR/ReasonEval-7B" \
    --shepherd_path "peiyi9979/math-shepherd-mistral-7b-prm"
```

### 启用Token概率记录
```bash
python answer_generation.py \
    --dataset_name "hybrid_reasoning" \
    --models "Abel-7B-002" \
    --temperatures "0.1" "0.3" "0.6" \
    --logger \
    --project "My-Experiment" \
    --enable_evaluation \
    --log_token_probs
```

### 完整示例
使用提供的脚本：
```bash
./run_evaluation_with_wandb.sh
```

## 新增参数说明

### 评估相关参数
- `--enable_evaluation`: 启用实时评估功能
- `--reasoneval_path`: ReasonEval模型路径 (默认: "GAIR/ReasonEval-7B")
- `--reasoneval_model_size`: ReasonEval模型大小 ("7B" 或 "34B")
- `--shepherd_path`: Math-Shepherd模型路径
- `--log_token_probs`: 启用token概率记录

### WandB相关参数
- `--logger`: 启用WandB日志记录
- `--entity`: WandB entity名称
- `--project`: WandB项目名称
- `--name`: 实验名称（自动设置为模型名-数据集名）

## WandB日志内容

### 样本级别指标
- `sample_validity_{temp}`: 单个样本的validity score
- `sample_redundancy_{temp}`: 单个样本的redundancy score  
- `sample_shepherd_{temp}`: 单个样本的shepherd score
- `sample_avg_top1_prob_{temp}`: 单个样本的平均top-1 token概率
- `sample_avg_top5_prob_{temp}`: 单个样本的平均top-5 token概率总和

### 聚合指标
- `avg_validity_{temp}`: 某温度下所有样本的平均validity score
- `avg_redundancy_{temp}`: 某温度下所有样本的平均redundancy score
- `avg_shepherd_{temp}`: 某温度下所有样本的平均shepherd score
- `avg_top1_prob_{temp}`: 某温度下所有样本的平均top-1概率
- `avg_top5_prob_{temp}`: 某温度下所有样本的平均top-5概率

## 输出文件格式

生成的JSON文件现在包含评估分数（如果启用了评估）：
```json
{
  "uuid": "sample_id",
  "question": "问题内容",
  "source": "数据来源",
  "model_output_steps": ["Step 1: ...", "Step 2: ..."],
  "validity_score": 0.85,
  "redundancy_score": 0.12,
  "shepherd_score": 0.92
}
```

## 性能注意事项

1. **内存使用**: 启用评估会加载额外的模型，需要更多GPU内存
2. **推理速度**: 实时评估会显著增加处理时间
3. **网络**: WandB日志记录需要网络连接

## 故障排除

### 评估模型加载失败
如果评估模型加载失败，脚本会自动禁用评估功能并继续运行。

### 内存不足
- 减少`--tensor_parallel_size`
- 降低`--gpu_memory_utilization`
- 使用较小的ReasonEval模型（7B而不是34B）

### WandB连接问题
确保已正确设置WandB账户和API密钥：
```bash
wandb login
```
