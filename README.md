Hi! This is the research I have done in Professor Jianguo Zhang's lab.

I analysis the reasoning ability of LLMs under the influence of temperature, particularly focusing on the influence on **reasoning steps**. The following is the whole pipeline. 

- First, I ask models in different temperatures to generate answers for problems in datasets. 
- Then, I use evaluators, such as ReasonEval and Math Shepherd, to evaluate the quality of the generated answers.

<img width="717" height="513" alt="image" src="https://github.com/user-attachments/assets/0f79f5af-a756-401a-90eb-a9624d514e59" />

Based on the figure, I scaled up the experiment. For example, 
- Datasets: I adopted several datasets such as AIME 22-24, MATH and MR-GSM8K.
- Models: I used Abel, Wizard and Qwen3 0.6B, 4B, 8B, 14B models to do large-scale experiment. And calling APIs of Deepseek-V3 (0324), OpenAI models(GPT-4o-mini) in a smaller scale.

At first, I use a simple setting to conduct experiments. The following is the result.

<img width="480" height="295" alt="image" src="https://github.com/user-attachments/assets/5c77747a-3791-4281-84e9-94f7d787a481" />

<img width="480" height="295" alt="image" src="https://github.com/user-attachments/assets/0b3d3a9d-dd37-45c1-bcf7-cb383050772a" />

Here, hybrid_reasoning is a combined dataset. I randomly pick 50 questions from each of the MATH level 1-5 and MR-GSM8K datasets, forming totally 300 questions. And Validity score, redundancy score, shepherd score are 3 metrics based on reasoning steps. From a simple view, the higher the validity and shepherd scores are, the better the reasoning is. The lower the redundancy score is, the clearer the reasoning is.

From the figure, I find that Deepseek-V3 and GPT-4o-mini significantly outperform Abel and Wizard, which are based on Llama models. To clarify the reason, I studied some instances of the answers. The followings are the answers for a specific problem.

<img width="1241" height="699" alt="image" src="https://github.com/user-attachments/assets/7c709010-67f7-45e5-a41f-9013eaa4f4cb" />

<img width="1241" height="697" alt="image" src="https://github.com/user-attachments/assets/2c49ed1f-108f-45d8-a9e9-e6c44ac708ae" />

From the answer we can see that Abel and Wizard models output some unrelated informations, such as dates and meaningless symbols. Even GPT-4o-mini uses some strange symbols to answer the question at a high temperature. Instead, Deepseek V3 performs well and steadily, generating ordered and helpful answers. It also proves the correctness of the evaluators.

TODO: what I find to inspire me to do the following experiments(difficulty  of datasets, flat in  deepseek)

After initial experiments, I gain new interests in the following aspects.
- Whether the model size influence the result?
- Whether the dataset's difficulty influence the result?
- Whether the evaluator size influence the result?

With these questions, I continued to do more experiments.
