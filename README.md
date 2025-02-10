<div align="center">
<h1 style="font-size: 20px;"><img src="static/images/facinfinity.webp" height="40px" fontsize="12pt" align="top"/> GSM-Infinite: How Do Your LLMs Behave over Infinitly <br>Increasing Context Length and Reasoning Complexity?
</h1>
</div> 
GSM-Infinite is a reasoning benchmarks that is completely synthetic without LLMs in the loop, capable of generating problems of context length and reasoning complexity that are infinitely scalable. Inspired by <a href="https://arxiv.org/abs/2407.20311">Physics of Language Model 2.1</a>, we use abstract grade school level math problems in to computational graph and through graph manipulation and graph-language mapping to generate LLM-readable (also, Human-readable) problems. 

<div align="center">
<b><a href="https://github.com/YangZhou08">Yang Zhou*</a></b><sup>1</sup>,
<b><a href="">Hongyi Liu*</a></b><sup>1</sup>,
<b><a href="https://github.com/dreaming-panda">Zhuoming Chen</a></b><sup>1</sup>,
<b><a href="">Yuandong Tian</a></b><sup>2</sup>,
<b><a href="https://github.com/keroro824">Beidi Chen</a></b><sup>1</sup>,
</div> 
*Equal Contributions, order decided by a coin flip 
<div align="center">
<sup>1</sup>Carnegie Mellon University
<sup>2</sup>Meta AI 
</div>

<div align="center">
[<a href="">Paper</a>] | [<a href="https://infini-ai-lab.github.io/gsm_infinite/">Blog</a>] 
</div> 

<h2>Problem of Contextual Sparsity</h2> 
<div align="center">
<img src="static/images/probwebsite.png"/>
<figcaption>Contextual Sparsity Weakness in Complex Reasoning Tasks</figcaption> 
</div> 
In this paper, we evaluate Contextual Sparsity (CS) models comprehensively on various complex generation tasks. 
CS models are evaluated at their default sparsity (50% neuron sparsity). Across the evaluation, we present the following takeaways: 
<ol>
<li>CS models work well on prompt understanding tasks, e.g. text summarization (CNN/DailyMail) and conversation question answering (CoQA). </li>
<li><span style="font-weight: bold;">CS models significantly ill-perform on generation tasks that require complex reasoning </span> (GSM8K) or knowledge-based tasks (MMLU-FLAN-COT). </li>
<li>The problem in complex reasoning generation tasks escalates for more well-trained model, given the similar parameter count. </li>
</ol> 
