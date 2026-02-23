
<div align="center">
<h1>Learning to Reason in 4D: Dynamic Spatial Understanding for Vision Language Models</h1>
</div>

<div align="center">
<a href="https://arxiv.org/abs/2512.20557" target="_blank">
    <img alt="arXiv" src="https://img.shields.io/badge/arXiv-paper-red?logo=arxiv" height="20" />
</a>
<a href="https://huggingface.co/collections/TencentARC/dsr-suite" target="_blank">
    <img src="https://img.shields.io/badge/%F0%9F%A4%97%20_Hugging--face-Data & Model-ffc107?color=ffc107&logoColor=white" height="20" />
</a>
</div>

<div align="center">
    <a href="https://scholar.google.com/citations?user=Q5jj-TYAAAAJ&hl=en" target="_blank">Shengchao Zhou</a><sup>1</sup>,</span>
    <a href="https://scholar.google.com/citations?hl=en&user=dEm4OKAAAAAJ" target="_blank">Yuxin Chen</a><sup>2</sup>, </span>
    <a href="https://scholar.google.com/citations?user=hv1LiiEAAAAJ&hl=en&oi=ao" target="_blank">Yuying Ge</a><sup>2</sup>,</span>
    <a href="https://scholar.google.com/citations?user=rZVUlPAAAAAJ&hl=zh-CN" target="_blank">Wei Huang</a><sup>1</sup>,</span>
    <a href="https://scholar.google.com/citations?user=eSkDBYcAAAAJ&hl=zh-CN" target="_blank">Jiehong Lin</a><sup>1</sup>,</span>
    <a href="https://scholar.google.com/citations?hl=en&user=bGn0uacAAAAJ&view_op=list_works&sortby=pubdate" target="_blank">Xiaojuan Qi</a><sup>1</sup>,</span>
</div>

<div align="center">
    <sup>1</sup>The University of Hong Kong&emsp;
    <sup>2</sup>ARC Lab, Tencent PCG&emsp;
</div>

<div align="center">

[![Watch the video](assets/demo_video_first_frame.png)](https://www.youtube.com/watch?v=ps8wzm8kDC0)

</div>

We introduce **DSR Suite**, which consists of an automated data generation pipeline that constructs multiple-choice question answer pairs (QAs) for ***dynamic spatial reasoning*** task and a module that seamlessly integrates geometric priors from 3D foundation models into VLMs without hindering their general understanding capability. With in-the-wild videos, randomly selected viewpoints, target objects, predefined templates and answer derivation rules, the generated QAs can evaluate object- and scene-level 3D understanding, multi-object interactions, viewpoint transformations and fine-grained temporal reasoning in general dynamic scenes. Based on the pipeline, we generate: (1) **DSR-Train**, a training dataset with 50K QAs; (2) **DSR-Bench**, a benchmark with 1484 QAs further refined by human annotators. In addition to data, we introduce **Geometry Selection Module** (GSM) that is stacked with two Q-Formers, where the first Q-Former condenses question semantics and the second Q-Former extracts question-relevant knowledge from 3D foundation models into a compact set of geometry tokens that will be concatenated with vision tokens.  When integrated with our GSM, the spatial reasoning ability of a VLM will improve while its general understanding performance is preserved. 

<p align="center" width="100%">
<img src="assets/teaser.png" style="width: 100%; min-width: 300px; display: block; margin: auto;">
</p>

## Table of Contents
1. [News](#news)
2. [Data Generation](#data-generation)
3. [Model Training](#model-training)
4. [Citation](#citation)
5. [Acknowledgement](#acknowledgement)

## News
- [x] [2026.2.21] Our [Paper](https://arxiv.org/abs/2512.20557) is accepted by CVPR 2026!
- [x] [2025.12.25] We release [Paper](https://arxiv.org/abs/2512.20557) and this GitHub repo [DSR_Suite](https://github.com/TencentARC/DSR_Suite).


## Data Generation
Please refer to `data/README.md` for details of automated data generation pipeline. Since there are some differences between the packages of "data generation" and "model training", we recommend two different installations. 

## Model Training
Please refer to `model/README.md` for details of training the model with GSM. Since there are some differences between the packages of "model training" and "data generation", we recommend two different installations.

## Benchmark Evaluation
First download our constructed [DSR-Bench](https://huggingface.co/datasets/TencentARC/DSR_Suite-Data) as `benchmark.parquet`. Then modify `'PATH_TO_VIDEO_ROOT'` and `'PATH_TO_PARQUET'` in `./VLMEvalKit_mine/vlmeval/dataset/spatial_reasoning.py` to the path of directory containing videos and the path to `benchmark.parquet` respectively.

Our current evaluation framework is based on [VLMEvalKit](https://github.com/open-compass/VLMEvalKit) so that you can adopt your model for evaluation according to its [Development_Guide](https://github.com/open-compass/VLMEvalKit/blob/main/docs/en/Development.md). To evaluate your adopted model or those already supported by VLMEvalKit, run the following command:
```bash
cd VLMEvalKit_mine
CUDA_VISIBLE_DEVICES=0 python run.py --data Spatial-Reasoning --model YOUR_MODEL --work-dir spatial_reasoning
```

If with Qwen2.5-VL-7B-Instruct as example, it will be:
```bash
cd VLMEvalKit_mine
CUDA_VISIBLE_DEVICES=0 python run.py --data Spatial-Reasoning --model Qwen2.5-VL-7B-Instruct-ForVideo --work-dir spatial_reasoning
```
where `--work-dir` is the directory to save the prediction results. The prediction results will be saved in `./spatial_reasoning/{YOUR_MODEL}/{YOUR_MODEL}_Spatial-Reasoning_score.xlsx`, where the column of 'score' indicates whether the model prediction is the same as the correct answer.

To evaluate the model you trained before or [our trained model](https://huggingface.co/TencentARC/DSR_Suite-Model), modify the path of model's checkpoint `'PATH_TO_MODEL'` in `./VLMEvalKit_mine/vlmeval/config.py` to the evaluated one and run:
```bash
cd VLMEvalKit_mine
CUDA_VISIBLE_DEVICES=0 python run.py --data Spatial-Reasoning --model Qwen2.5-VL-7B-Instruct-ForVideo-Spatial --work-dir spatial_reasoning
```

## Citation
If you find our work useful, please consider citing:

```bibtex
@misc{zhou2025learning,
      title={Learning to Reason in 4D: Dynamic Spatial Understanding for Vision Language Models}, 
      author={Shengchao Zhou, Yuxin Chen, Yuying Ge, Wei Huang, Jiehong Lin, Ying Shan, Xiaojuan Qi},
      year={2025},
      eprint={2512.20557},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2512.20557}, 
}
```
## Acknowledgement
- [Qwen2.5-VL](https://github.com/QwenLM/Qwen3-VL): the model codebase we built upon.
- [VLMEvalKit](https://github.com/open-compass/VLMEvalKit): the evaluation framework we built upon.
- [Grounded SAM2](https://github.com/IDEA-Research/Grounded-SAM-2), [Orient Anything](https://github.com/SpatialVision/Orient-Anything), [π^3](https://github.com/yyfz/Pi3): the models used in our data generation pipeline to extract 3D cues.
- [Koala-36M](https://github.com/KlingTeam/Koala-36M): the video database we build QAs upon.
