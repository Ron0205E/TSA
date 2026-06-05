# TSA: Two-Stage Jailbreak Attack via Logical Consistency Induction for Large Language Models
## Abstract
Large language models (LLMs) are increasingly used in complex decision-making, raising serious security concerns, especially from jailbreak attacks that exploit inherent model behaviors rather than surface-level tricks. One key vulnerability is the tendency of LLMs to maintain logical consistency—once a model accepts a seemingly legitimate premise, it struggles to reject conclusions derived from it. This repository introduces TSA (Two-Stage Attack), a jailbreak method that first guides the model to generate a structured, compliant analysis of a harmful request (logic presetting), and then leads it to autonomously derive concrete, violative steps from that analysis (intent enhancement). Evaluated on nine mainstream LLMs, TSA achieves an average attack success rate (ASR) of 84.83% with only 3.61 queries per successful jailbreak (QPS), outperforming existing baselines. These results highlight how logical consistency can be exploited to bypass current safety alignment mechanisms. All code and experimental details are available in this repository.

## Method Overview

The overall workflow of TSA is illustrated below. TSA consists of two sequential stages: **Logic Presetting** and **Intent Enhancement**. In the first stage, the target LLM is guided to produce a seemingly compliant and structured semantic analysis of the input request. This response does not directly expose harmful content, but it establishes a latent reasoning context that contains background information, possible execution paths, risks, and limitations.

In the second stage, TSA leverages the structured output from the first stage to further enhance the hidden intent. Through semantic focusing, structural reorganization, and depth refinement, the model is encouraged to transform the abstract reasoning context into more specific and actionable content. This two-stage process exploits the model's tendency to maintain logical consistency across turns, ultimately increasing the likelihood of a successful jailbreak.

![TSA Workflow](static/Architecture.svg)

## Getting Started
This repository provides the code and mini benchmark data needed to reproduce TSA experiments. We recommend creating a fresh Python environment with Python 3.8 or later, then installing the required runtime packages with `pip install requests pyyaml`. The main TSA implementation and launch scripts are located under `baselines/TSA/`.

Before running an experiment, open `baselines/TSA/config_tsa.yaml` and configure both the target model and judge model. In particular, fill in `model_name_or_path`, the OpenAI-compatible `base_url`, and the API `token` fields. The repository includes mini versions of AdvBench and HarmBench in `dataset/miniAdvBench.json` and `dataset/miniHarmbench.csv`, while generated result snapshots and JSONL logs are written to `baselines/TSA/output/` by default.

To verify the setup, run a small subset first: `python baselines/TSA/run_tsa_AdvBench.py --num 5 --threads 5` for AdvBench, or `python baselines/TSA/run_tsa_HarmBench.py --num 5 --workers 5` for HarmBench. After confirming that the configuration works, remove `--num` to process the full mini dataset, or use `--start`, `--end`, and `--indices` to run selected examples.

## ⚠️ Disclaimer & Ethical Warning

**This repository is provided for academic and research purposes only.** By accessing this code, you agree not to use it for malicious purposes, including generating harmful content, attacking LLM services, or bypassing content filters. Users assume full responsibility for any outputs generated, and the authors disclaim any liability for misuse. Vulnerability findings will be responsibly disclosed to affected LLM providers.

> **Short Notice:** This code is for academic research only. Do not use for malicious purposes. Misuse may violate LLM providers' terms of service and applicable laws.
