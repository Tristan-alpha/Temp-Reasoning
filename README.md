Hi! This is the research I have done in Professor Jianguo Zhang's lab.

I analysis the reasoning ability of LLMs under the influence of temperature. The following is the whole pipeline. 

- First, I ask models in different temperatures to generate answers for problems in datasets. 
- Then, I use evaluators, such as ReasonEval and Math Shepherd, to evaluate the quality of the generated answers.

<img width="1434" height="1026" alt="image" src="https://github.com/user-attachments/assets/0f79f5af-a756-401a-90eb-a9624d514e59" />

Based on the figure, I scaled up the experiment. For example, 
- Datasets: I adopted several datasets such as AIME 22-24, MATH and MR-GSM8K.
- Models: I used Abel, Wizard and Qwen3 0.6B, 4B, 8B, 14B models to do large-scale experiment. And calling APIs of Deepseek-V3 (0324), OpenAI models(GPT-4o-mini) in a smaller scale.

