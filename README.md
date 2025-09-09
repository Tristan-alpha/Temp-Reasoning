Hi! This is the research I have done in Professor Jianguo Zhang's lab.

I analysis the reasoning ability of LLMs under the influence of temperature, particularly focusing on the influence on **reasoning steps**. The following is the whole pipeline. 

- First, I ask models in different temperatures to generate answers for problems in datasets. 
- Then, I use evaluators, such as ReasonEval and Math Shepherd, to evaluate the quality of the generated answers.

<img width="717" height="513" alt="image" src="https://github.com/user-attachments/assets/0f79f5af-a756-401a-90eb-a9624d514e59" />

Based on the figure, I scaled up the experiment. For example, 
- Datasets: I adopted several datasets such as AIME 22-24, MATH and MR-GSM8K.
- Models: I used Abel, Wizard and Qwen3 0.6B, 4B, 8B, 14B models to do large-scale experiment. And calling APIs of Deepseek-V3 (0324), OpenAI models(GPT-4o-mini) in a smaller scale.

At first, I use a simple setting to conduct experiments. The following is the result.

<img width="960" height="590" alt="image" src="https://github.com/user-attachments/assets/5c77747a-3791-4281-84e9-94f7d787a481" />

<img width="960" height="590" alt="image" src="https://github.com/user-attachments/assets/0b3d3a9d-dd37-45c1-bcf7-cb383050772a" />

Here, hybrid_reasoning is a combined dataset. **TODO** And Validity score, redundancy score, shepherd score are 3 metrics based on reasoning steps. From a simple view, the higher the validity and shepherd scores are, the better the reasoning is. The lower the redundancy score is, the clearer the reasoning is.

TODO: what I find to inspire me to do the following experiments(difficulty  of datasets, flat in  deepseek)

After initial experiments, I gain new interests in the following aspects.
- Whether the model size influence the result?
- Whether the dataset's difficulty influence the result?
- Whether the evaluator size influence the result?

With these questions, I continued to do more experiments.
