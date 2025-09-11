Hi! This is the research I have done in Professor Jianguo Zhang's lab.

# Motivation
The motivation is from this paper, "The Effect of Sampling Temperature on Problem Solving in Large Language Models". It claims that temperature doesn't influence model's performance when the temperature changes from 0.0 to 1.0. It actually violates my intuition. With the increase of temperature, I think the performance will continually decrease, considering the problem is a reasoning task and it doesn't require creativity. After closer inspection, I noticed that the work is based on MCQA (Multiple-Choice Question-and-Answer) problems and it mainly focuses on final accuracy. This gives me some inspirations: will it be different if I go deeper into the **reasoning steps**? 

# Pipeline
To find a good way to measure the reasoning steps' quality, I made a wide range of literature research and found some fine-tuned models that can score the steps based on validity and redundancy. With the help of these models (will be introduced in detail later), I build up the whole pipeline independently.

- First, I ask models in different temperatures to generate answers for problems in datasets. 
- Then, I use evaluators, such as ReasonEval and Math Shepherd, to evaluate the quality of the generated answers.

<img width="717" height="513" alt="image" src="https://github.com/user-attachments/assets/0f79f5af-a756-401a-90eb-a9624d514e59" />

The figure above is the initial plan. Then, I scaled up the experiment. For example, 
- Datasets: I adopted several datasets such as AIME 22-24, MATH and MR-GSM8K.
- Models: I used Abel, Wizard and Qwen3 0.6B, 4B, 8B, 14B models to do large-scale experiment. And calling APIs of Deepseek-V3 (0324), OpenAI models(GPT-4o-mini) in a smaller scale.

# Initial Attempt
At first, I use a simple setting to conduct experiments. The following is the result.

<img width="768" height="267" alt="image" src="https://github.com/user-attachments/assets/b738c413-339b-4fcf-8cbf-c0a6f1d52c84" />

<img width="480" height="295" alt="image" src="https://github.com/user-attachments/assets/0b3d3a9d-dd37-45c1-bcf7-cb383050772a" />

Here, hybrid_reasoning is a combined dataset. I randomly pick 50 questions from each of the MATH level 1-5 (1 is the easiest and 5 is the hardest) and MR-GSM8K datasets, forming totally 300 questions. And Validity score, redundancy score, shepherd score are 3 metrics based on reasoning steps. From a simple view, the higher the validity and shepherd scores are, the better the reasoning is. The lower the redundancy score is, the clearer the reasoning is. The Validity score and redundancy score come from ReasonEval, and shepherd score comes from MATH-Shepherd.

From the figure, I find that Deepseek-V3 and GPT-4o-mini significantly outperform Abel and Wizard, which are based on Llama models. To clarify the reason, I studied some instances of the answers. The followings are the answers for a specific problem.

<img width="1240" height="700" alt="image" src="https://github.com/user-attachments/assets/7c709010-67f7-45e5-a41f-9013eaa4f4cb" />

<img width="1240" height="700" alt="image" src="https://github.com/user-attachments/assets/2c49ed1f-108f-45d8-a9e9-e6c44ac708ae" />

From the answer we can see that Abel and Wizard models output some unrelated informations, such as dates and meaningless symbols. Even GPT-4o-mini uses some strange symbols to answer the question at a high temperature. Instead, Deepseek V3 performs well and steadily, generating ordered and helpful answers. It also proves the correctness of the evaluators.

# Further Experiments
I found that other 3 models's performances decrease with the increase of temperatures. But Deepseek V3 doesn't. I guess the reason is Deepseek has seen the datasets before and it's very easy for it, considering the difficulties of MATH and GSM8K are merely junior and high school level. To avoid this, I expand the datasets and adopts AIME 22-24, which are mathematical Olympiad (IMO) level. 

In addition, with the release of Qwen3 series models, I also employed Qwen3 models of different sizes to do the large-scale experiments. In order to make the following experiments clearer and make the content richer, I come up with the following 2 questions.

- Whether the model size influence the result?
- Whether the dataset's difficulty influence the result?

To answer these questions, the experiment is conducted through the following 2 aspects.
- Same datasets, Same evluators, Different models
- Same evluators, Same models, Different datasets

The setting table is as following. The evaluators are ReasonEval 7B, ReasonEval 34B and MATH-Shepherd. So there should be another dimension but I don't know how to present. :)

| Models     | math-1 | math-3 | math-5 | AIME |
| ---------- | :----: | :----: | :----: | :--: |
| Qwen3 0.6B |   ✅    |   ✅    |   ✅    |  ✅   |
| Qwen3 4B   |   ✅    |   ✅    |   ✅    |  ✅   |
| Qwen3 8B   |   ✅    |   ✅    |   ✅    |  ✅   |
| Qwen3 14B  |   ✅    |   ✅    |   ✅    |  ✅   |


During the experiment, I find it hard to manage the result only with python. So I used WandB to restore the experiment's result. The following is the total result.

<img width="1513" height="807" alt="image" src="https://github.com/user-attachments/assets/e938f93a-0e81-418e-ba9b-c56c7278c8aa" />

The following result is a representative, which uses the labels to clearly present the result according to the above 2 aspects.

<img width="1230" height="302" alt="image" src="https://github.com/user-attachments/assets/1a7ebb1a-5b42-4c3b-9865-8170a7dff69b" />

The dataset is AIME and the ReasonEval size is 7B. The model size varies. From the figure we can see that roughly with the increase of model size, the model performs better.

<img width="1229" height="303" alt="image" src="https://github.com/user-attachments/assets/6c55c5d6-fa61-4f5f-92b3-1e229280b1ed" />

The model size is Qwen3 4B and the ReasonEval size is 7B. The difficulty of datasets vary. From the figure we can see that with the increase of difficulty, models perform worse. However, the result didn't support my guess, that the line will decrease harder with the increase of the difficulty. 

During the exploration, I found an interesting phonomenon. In each of the datasets, Qwen3 0.6B's token probability drops the fastest! The evidences are listed below. I think this is because it has the least knowledge compared to other models, so it is the most **uncertain** about which token to generate. But there is one thing strange. In AIME, I tried Qwen3 32B, but it is also **unconfident**. The token probability is as low as Qwen3 0.6B. It's an interesting phonomenon remained to explore.

<img width="765" height="280" alt="image" src="https://github.com/user-attachments/assets/8c950cba-baa7-4764-ae47-d9e47c74233b" />
Dataset: MATH-1

<img width="765" height="280" alt="image" src="https://github.com/user-attachments/assets/1c5a56b9-fb4b-467b-b8dd-930342b0d388" />
Dataset: MATH-3

<img width="765" height="280" alt="image" src="https://github.com/user-attachments/assets/57726d3e-7296-492c-b5e2-470641454e1e" />
Dataset: MATH-5

<img width="765" height="280" alt="image" src="https://github.com/user-attachments/assets/65f6c908-786c-48e2-a14e-3cb0af50ab27" />
Dataset: AIME

# Conclusion
In general, the experiments' result support the conclusion of the paper "The Effect of Sampling Temperature on Problem Solving in Large Language Models", that temperature doesn't influence model's performance when the temperature changes from 0.0 to 1.0. In the process, I also found that there are some uncertainties using LLM to judge the reasoning steps. It's not easy to judge the quality of the output of LLM, not to mention using LLM to evaluate the output of other LLMs. The evaluators provide some disturbances, making some of the results hard to analysis. After all, it's a precious opportunity for me to study about LLM and I gained valuable engineering experiences in it.
