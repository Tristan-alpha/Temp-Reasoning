import argparse
import json
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
import requests
from sklearn.metrics import roc_curve, auc
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

# 导入已有的评估函数
from examples import get_results
from mr_gsm8k_eval import calculate_f1_score

class TemperatureAnalyzer:
    def __init__(self, args):
        self.args = args
        self.temperatures = np.arange(args.temp_min, args.temp_max + args.temp_step, args.temp_step)
        self.temperatures = [round(t, 2) for t in self.temperatures]
        self.api_key = args.api_key
        self.model = args.model
        self.output_dir = args.output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 加载数据集
        self.questions = self._load_questions()
        
    def _load_questions(self):
        """加载要测试的问题"""
        questions = []
        if self.args.dataset_path.endswith('.json'):
            with open(self.args.dataset_path, 'r') as f:
                dataset = json.load(f)
                if isinstance(dataset, list):
                    # 假设每个问题都有一个'question'字段
                    if len(dataset) > self.args.max_samples:
                        dataset = dataset[:self.args.max_samples]
                    for item in dataset:
                        if isinstance(item, dict) and 'question' in item:
                            questions.append(item['question'])
                        elif isinstance(item, str):
                            questions.append(item)
        return questions
    
    def generate_reasoning(self, question, temperature):
        """使用API生成特定温度下的推理过程"""
        prompt = f"Question: {question}\n\nAnswer: Let's think step by step."
        
        # 这里使用OpenAI API作为示例，可以替换为其他API
        try:
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": temperature,
                    "max_tokens": 1000
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                reasoning = result["choices"][0]["message"]["content"]
                # 将推理拆分成步骤
                steps = self._split_into_steps(reasoning)
                return steps
            else:
                print(f"API错误: {response.status_code}")
                print(response.text)
                return []
        except Exception as e:
            print(f"生成推理时发生错误: {e}")
            return []
    
    def _split_into_steps(self, reasoning):
        """将推理文本拆分为单独的步骤"""
        # 简单的拆分方法，可以根据实际输出格式进行调整
        steps = []
        lines = reasoning.strip().split('\n')
        current_step = ""
        
        for line in lines:
            if line.strip() == "":
                continue
                
            # 检查是否是新的步骤开始
            if line.strip().startswith(str(len(steps) + 1) + ".") or line.strip().startswith("Step " + str(len(steps) + 1)):
                if current_step:
                    steps.append(current_step.strip())
                current_step = line
            else:
                current_step += "\n" + line
                
        if current_step:
            steps.append(current_step.strip())
            
        return steps
    
    def evaluate_reasoning(self, question, reasoning_steps, temperature):
        """使用ReasonEval评估推理步骤"""
        # 创建一个临时参数对象用于调用get_results函数
        class TempArgs:
            def __init__(self, model_name, model_size):
                self.model_name_or_path = model_name
                self.model_size = model_size
        
        temp_args = TempArgs(self.args.reasoneval_model, self.args.model_size)
        
        try:
            # 使用examples.py中的get_results函数评估推理
            get_results(temp_args, question, reasoning_steps)
            # 注意：这里需要修改get_results函数以返回评估结果，而不是打印
            # 这里我们假设get_results已被修改为返回评估结果
            return {
                "temperature": temperature,
                "step_level_validity_scores": step_level_validity_scores,
                "step_level_redundancy_scores": step_level_redundancy_scores,
                "solution_level_validity_score": solution_level_validity_score,
                "solution_level_redundancy_score": solution_level_redundancy_score
            }
        except Exception as e:
            print(f"评估错误: {e}")
            return None
    
    def analyze_temperature_impact(self):
        """分析温度对推理过程的影响"""
        results = {}
        
        for temp in self.temperatures:
            print(f"\n分析温度 {temp}...")
            temp_results = []
            
            for i, question in enumerate(tqdm(self.questions)):
                print(f"\n问题 {i+1}/{len(self.questions)}")
                
                # 生成推理
                reasoning_steps = self.generate_reasoning(question, temp)
                if not reasoning_steps:
                    continue
                    
                # 评估推理
                eval_result = self.evaluate_reasoning(question, reasoning_steps, temp)
                if eval_result:
                    temp_results.append(eval_result)
                    
                # 避免API限制
                time.sleep(1)
            
            results[temp] = temp_results
            
            # 保存中间结果
            self._save_results(results, f"temperature_analysis_until_{temp}.json")
        
        return results
    
    def _save_results(self, results, filename):
        """保存分析结果为JSON文件"""
        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"结果已保存到 {filepath}")
    
    def visualize_results(self, results):
        """可视化不同温度下的评估指标"""
        # 提取每个温度的平均指标
        temp_values = []
        validity_means = []
        redundancy_means = []
        sol_validity_means = []
        sol_redundancy_means = []
        
        for temp, temp_results in results.items():
            if not temp_results:
                continue
                
            temp_values.append(temp)
            
            # 计算步骤级指标的平均值
            all_validity = []
            all_redundancy = []
            for result in temp_results:
                all_validity.extend(result["step_level_validity_scores"])
                all_redundancy.extend(result["step_level_redundancy_scores"])
            
            validity_means.append(np.mean(all_validity))
            redundancy_means.append(np.mean(all_redundancy))
            
            # 计算解决方案级指标的平均值
            sol_validities = [r["solution_level_validity_score"] for r in temp_results]
            sol_redundancies = [r["solution_level_redundancy_score"] for r in temp_results]
            sol_validity_means.append(np.mean(sol_validities))
            sol_redundancy_means.append(np.mean(sol_redundancies))
        
        # 创建图表
        plt.figure(figsize=(16, 12))
        
        # 步骤级有效性和冗余性
        plt.subplot(2, 2, 1)
        plt.plot(temp_values, validity_means, 'b-o', label='步骤级有效性')
        plt.plot(temp_values, redundancy_means, 'r-o', label='步骤级冗余性')
        plt.xlabel('温度')
        plt.ylabel('平均分数')
        plt.title('温度对步骤级评估指标的影响')
        plt.legend()
        plt.grid(True)
        
        # 解决方案级有效性和冗余性
        plt.subplot(2, 2, 2)
        plt.plot(temp_values, sol_validity_means, 'b-o', label='解决方案级有效性')
        plt.plot(temp_values, sol_redundancy_means, 'r-o', label='解决方案级冗余性')
        plt.xlabel('温度')
        plt.ylabel('平均分数')
        plt.title('温度对解决方案级评估指标的影响')
        plt.legend()
        plt.grid(True)
        
        # 保存图表
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'temperature_impact_visualization.png'))
        plt.close()
        
        print(f"可视化结果已保存到 {self.output_dir}")
        
        # 将数据保存为CSV以便进一步分析
        data = {
            'Temperature': temp_values,
            'Step_Level_Validity': validity_means,
            'Step_Level_Redundancy': redundancy_means,
            'Solution_Level_Validity': sol_validity_means,
            'Solution_Level_Redundancy': sol_redundancy_means
        }
        df = pd.DataFrame(data)
        df.to_csv(os.path.join(self.output_dir, 'temperature_analysis_results.csv'), index=False)

    def run_analysis(self):
        """运行完整的分析流程"""
        print(f"开始分析温度范围 {min(self.temperatures)} 到 {max(self.temperatures)} 对推理过程的影响...")
        results = self.analyze_temperature_impact()
        self._save_results(results, "temperature_analysis_complete.json")
        self.visualize_results(results)
        print("温度分析完成！")

def main():
    parser = argparse.ArgumentParser(description='分析温度对推理过程的影响')
    parser.add_argument('--temp_min', type=float, default=0.1, help='最小温度值')
    parser.add_argument('--temp_max', type=float, default=2.0, help='最大温度值')
    parser.add_argument('--temp_step', type=float, default=0.2, help='温度步长')
    parser.add_argument('--api_key', type=str, required=True, help='API密钥')
    parser.add_argument('--model', type=str, default='gpt-3.5-turbo', help='要使用的模型')
    parser.add_argument('--dataset_path', type=str, required=True, help='数据集路径')
    parser.add_argument('--output_dir', type=str, default='./temperature_analysis_results', help='输出目录')
    parser.add_argument('--max_samples', type=int, default=10, help='最大样本数')
    parser.add_argument('--reasoneval_model', type=str, default='GAIR/ReasonEval-7B', help='评估模型路径')
    parser.add_argument('--model_size', type=str, choices=['7B', '34B'], default='7B', help='模型大小')
    
    args = parser.parse_args()
    
    analyzer = TemperatureAnalyzer(args)
    analyzer.run_analysis()

if __name__ == "__main__":
    main()
