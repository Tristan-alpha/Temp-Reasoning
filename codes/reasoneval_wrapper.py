import argparse
import torch
from transformers import AutoTokenizer
import numpy as np

# 导入ReasonEval模型类
from examples import ReasonEval_7B, ReasonEval_34B

class ReasonEvalWrapper:
    """
    ReasonEval的包装类，提供简化的接口进行推理评估
    """
    def __init__(self, model_path, model_size='7B'):
        """
        初始化ReasonEval评估器
        
        参数:
            model_path: ReasonEval模型路径
            model_size: 模型大小，'7B'或'34B'
        """
        self.model_path = model_path
        self.model_size = model_size
        
        # 加载模型和tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if model_size == '7B':
            self.model = ReasonEval_7B.from_pretrained(model_path)
        elif model_size == '34B':
            self.model = ReasonEval_34B.from_pretrained(model_path)
        else:
            raise ValueError(f"不支持的模型大小: {model_size}")
        
        # 如果有GPU可用则使用GPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        
    def evaluate(self, question, reasoning_steps):
        """
        评估推理过程
        
        参数:
            question: 问题文本
            reasoning_steps: 推理步骤列表
            
        返回:
            包含评估分数的字典
        """
        # 准备输入
        PROMPT_FORMAT = "Question:\n{input}\nAnswer:\nLet's think step by step.\n"
        step_separator = f"{self.tokenizer.pad_token}"
        combined_steps = ""
        for steps in reasoning_steps:
            combined_steps += steps + step_separator
        prompt = PROMPT_FORMAT.format(input=question)
        tokenized_result = self.tokenizer(prompt + step_separator + combined_steps)['input_ids']
        
        # 分离标签并调整token ID
        separator_token_id = self.tokenizer(step_separator)['input_ids'][-1]
        labeled_token_indices = []
        adjusted_token_ids = []
        separator_count = 0
        
        for idx, token_id in enumerate(tokenized_result):
            if token_id == separator_token_id:
                labeled_token_indices.append(idx - 1 - separator_count)
                separator_count += 1
            else:
                adjusted_token_ids.append(token_id)
                
        if self.model_size == '7B':
            adjusted_token_ids = [1] + adjusted_token_ids
            adjusted_token_ids = torch.tensor([adjusted_token_ids]).to(self.device)
            labeled_token_indices = labeled_token_indices[2:]
        elif self.model_size == '34B':
            adjusted_token_ids = torch.tensor([adjusted_token_ids]).to(self.device)
            labeled_token_indices = labeled_token_indices[1:]
        
        attention_mask = adjusted_token_ids.new_ones(adjusted_token_ids.size(), dtype=torch.bool)
        
        # 评估推理步骤
        with torch.no_grad():
            reasoning_scores = self.model(adjusted_token_ids, attention_mask)[0, labeled_token_indices, :]
            scores = torch.softmax(reasoning_scores, dim=-1).tolist()
        
        # 计算有效性和冗余性分数
        step_level_validity_scores = [(score[1] + score[2]) for score in scores]
        step_level_redundancy_scores = [score[1] for score in scores]
        solution_level_validity_score = min(step_level_validity_scores)
        solution_level_redundancy_score = max(step_level_redundancy_scores)
        
        return {
            "step_level_validity_scores": step_level_validity_scores,
            "step_level_redundancy_scores": step_level_redundancy_scores,
            "solution_level_validity_score": solution_level_validity_score,
            "solution_level_redundancy_score": solution_level_redundancy_score,
            "raw_scores": scores
        }

def main():
    """用于测试ReasonEvalWrapper的简单示例"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default='GAIR/ReasonEval-7B')
    parser.add_argument("--model_size", type=str, choices=['7B', '34B'], default='7B')
    args = parser.parse_args()
    
    # 测试示例
    question = "Let $x,$ $y,$ and $z$ be positive real numbers such that $xyz(x + y + z) = 1.$  Find the minimum value of\n\\[(x + y)(y + z).\\]"
    reasoning_steps = [
        "1. The problem asks us to find the minimum value of $(x + y)(y + z)$ given that $x,$ $y,$ and $z$ are positive real numbers and $xyz(x + y + z) = 1$.", 
        "2. By the AM-GM inequality, we have $x + y + z \\geq 3\\sqrt[3]{xyz}$.",
        "3. Let's substitute this into the constraint. We have $xyz(x + y + z) = 1$.",
        "4. This means $(x + y + z) = \\frac{1}{xyz}$.",
        "5. Now we need to find the minimum value of $(x + y)(y + z)$."
    ]
    
    evaluator = ReasonEvalWrapper(args.model_path, args.model_size)
    result = evaluator.evaluate(question, reasoning_steps)
    
    print("\n评估结果:")
    print(f"步骤级有效性分数: {result['step_level_validity_scores']}")
    print(f"步骤级冗余性分数: {result['step_level_redundancy_scores']}")
    print(f"解决方案级有效性分数: {result['solution_level_validity_score']}")
    print(f"解决方案级冗余性分数: {result['solution_level_redundancy_score']}")

if __name__ == "__main__":
    main()
