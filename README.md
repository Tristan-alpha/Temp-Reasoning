Hi! This is the research I have done in Professor Jianguo Zhang's lab.

I analysis the reasoning ability of LLMs under the influence of temperature, particularly focusing on the influence on **reasoning steps**. The following is the whole pipeline. 

- First, I ask models in different temperatures to generate answers for problems in datasets. 
- Then, I use evaluators, such as ReasonEval and Math Shepherd, to evaluate the quality of the generated answers.

<img width="717" height="513" alt="image" src="https://github.com/user-attachments/assets/0f79f5af-a756-401a-90eb-a9624d514e59" />

Based on the figure, I scaled up the experiment. For example, 
- Datasets: I adopted several datasets such as AIME 22-24, MATH and MR-GSM8K.
- Models: I used Abel, Wizard and Qwen3 0.6B, 4B, 8B, 14B models to do large-scale experiment. And calling APIs of Deepseek-V3 (0324), OpenAI models(GPT-4o-mini) in a smaller scale.

At first, I use a simple setting to conduct experiments. The following is the result.

<img width="768" height="267" alt="image" src="https://github.com/user-attachments/assets/b738c413-339b-4fcf-8cbf-c0a6f1d52c84" />

<img width="480" height="295" alt="image" src="https://github.com/user-attachments/assets/0b3d3a9d-dd37-45c1-bcf7-cb383050772a" />

Here, hybrid_reasoning is a combined dataset. I randomly pick 50 questions from each of the MATH level 1-5 (1 is the easiest and 5 is the hardest) and MR-GSM8K datasets, forming totally 300 questions. And Validity score, redundancy score, shepherd score are 3 metrics based on reasoning steps. From a simple view, the higher the validity and shepherd scores are, the better the reasoning is. The lower the redundancy score is, the clearer the reasoning is. The Validity score and redundancy score come from ReasonEval, and shepherd score comes from MATH-Shepherd.

From the figure, I find that Deepseek-V3 and GPT-4o-mini significantly outperform Abel and Wizard, which are based on Llama models. To clarify the reason, I studied some instances of the answers. The followings are the answers for a specific problem.

<img width="1241" height="699" alt="image" src="https://github.com/user-attachments/assets/7c709010-67f7-45e5-a41f-9013eaa4f4cb" />

<img width="1241" height="697" alt="image" src="https://github.com/user-attachments/assets/2c49ed1f-108f-45d8-a9e9-e6c44ac708ae" />

From the answer we can see that Abel and Wizard models output some unrelated informations, such as dates and meaningless symbols. Even GPT-4o-mini uses some strange symbols to answer the question at a high temperature. Instead, Deepseek V3 performs well and steadily, generating ordered and helpful answers. It also proves the correctness of the evaluators.

I found that other 3 models's performances decrease with the increase of temperatures. But Deepseek V3 doesn't. I guess the reason is Deepseek has seen the datasets before and it's very easy for it, considering the difficulties of MATH and GSM8K are merely junior and high school level. To avoid this, I expand the datasets and adopts AIME 22-24, which are mathematical Olympiad (IMO) level. 

In addition, with the release of Qwen3 series models, I also employed Qwen3 models of different sizes to do the large-scale experiments. In order to make the following experiments clearer and make the content richer, I come up with the following 2 questions.

- Whether the model size influence the result?
- Whether the dataset's difficulty influence the result?

To answer these questions, the experiment is conducted through the following 2 aspects.
- Same datasets, Same evluators, Different models
- Same evluators, Same models, Different datasets

During the experiment, I find it hard to manage the result only with python. So I used WandB to restore the experiment's result. The following result is a representative, which uses the labels to clearly present the result according to the above 2 settings.

<img width="1230" height="302" alt="image" src="https://github.com/user-attachments/assets/1a7ebb1a-5b42-4c3b-9865-8170a7dff69b" />

The dataset is AIME and the ReasonEval size is 7B. The model size varies. From the figure we can see that roughly with the increase of model size, the model performs better.

<img width="1229" height="303" alt="image" src="https://github.com/user-attachments/assets/6c55c5d6-fa61-4f5f-92b3-1e229280b1ed" />

The model size is Qwen3 4B and the ReasonEval size is 7B. The difficulty of datasets vary. From the figure we can see that with the increase of difficulty, models perform worse. However, the result didn't support my guess, that the line will decrease harder with the increase of the difficulty. 

**TODO:** top 1 prob, 0.6B lowest, the most uncertain one. Uncertainty

