# K/V adaptation in self-attention and cross-attention

A study of the interface between frozen SmolVLM features and the SmolVLA action expert.

SmolVLA combines a frozen vision-language model with a trainable action expert. Its default expert alternates cross-attention and self-attention layers. The paper reports that this interleaved architecture reaches 85.5% success on LIBERO, compared with 79.0% for cross-attention alone and 74.5% for self-attention alone.

While examining the public LeRobot implementation, I found that the SA and CA variants differ in more than their attention pattern. In CA layers, frozen VLM keys and values pass through trainable projections owned by the action expert. In SA layers, the expert instead attends to prefix keys and values produced directly by the frozen VLM, with no equivalent expert-owned transformation.

This report tests whether this asymmetric VLM-to-expert interface explains the poor SA result and reevaluates the interpretation of the attention layout ablation presented in the original paper. I test the interface in both directions by adding equivalent expert-owned prefix K/V projections to SA and removing the existing expert-owned prefix K/V projections from CA.

Across two training seeds, adding the projection raised pure SA from 77.85% to 84.13%, a gain of 6.28 percentage points. Removing the projection lowered pure CA from 81.80% to 77.13%, a drop of 4.68 points. On LIBERO Long, adding the projection raised SA from 56.5% to 72.7%, while removing it lowered CA from 69.0% to 57.2%. Together, these reciprocal interventions show that the asymmetric prefix interface explains the poor relative SA performance in the original comparison.

## What is being compared

SmolVLA encodes the images, language instruction and robot state as a prefix using the frozen SmolVLM, while the action expert processes a sequence of noisy action tokens. In a cross-attention layer, action queries attend only to the VLM prefix. Before that attention operation, the cached VLM keys and values pass through trainable 320 by 320 projections owned by the action expert.

In a self-attention layer, prefix and action K/V are combined in one causal attention operation, allowing action tokens to attend both to the prefix and to earlier action tokens. In this path, the prefix K/V are produced by the frozen VLM and enter the joint attention operation without an equivalent expert-owned transformation.

The original SA and CA variants therefore change two things at once: the attention topology and the trainable interface between the frozen VLM prefix and the action expert. The first ablation adds an analogous expert-owned prefix K/V transformation to SA layers, referred to below as the K/V adapter. The second removes the existing expert-owned prefix K/V transformations from CA layers. Both interventions leave the attention topology, action-token projections, attention masks, positions and training objective unchanged.



This distinction matters because the paper uses the SA, CA and interleaved comparison to support the conclusion that the two attention mechanisms have complementary strengths. However, the original variants also differ in whether the frozen VLM prefix receives an expert-owned projection. Adding the projection to SA and removing it from CA isolate this interface difference from attention topology.

## Hypothesis

Going into these experiments I had two initial hypotheses about the effect on model performance.

The first hypothesis was that SA is disadvantaged by receiving prefix K/V directly from the frozen VLM, without the expert-owned transformation available in the CA path. Adding an analogous trainable transformation should therefore improve SA. The reverse prediction was that removing the transformation from CA should reduce its performance to approximately the original SA level.

The second hypothesis was that removing this asymmetry might allow an all-SA expert to match the interleaved architecture. Joint self-attention can in principle learn both prefix conditioning and action-token interaction without assigning fixed roles to different layers.

## Experiment

Before training, I checked the new SA path with the K/V adapter initialized to identity. The adapted and original models used identical shared weights, the same fixed LIBERO batch and the same flow-matching noise and time samples. Their losses agreed within `rtol=1e-3` and `atol=1e-4`. A backward pass also confirmed finite, nonzero adapter gradients. Identity initialization was used only for this check. The adapters used the standard model initialization in every training run.

I compared six model variants:

| Variant               | Attention layout     | Prefix K/V interface            |
| --------------------- | -------------------- | ------------------------------- |
| CA                    | Cross-attention only | Expert-owned projection         |
| CA without projection | Cross-attention only | Direct frozen VLM K/V           |
| SA                    | Self-attention only  | Direct frozen VLM K/V           |
| SA + adapter          | Self-attention only  | Expert-owned projection         |
| SA+CA                 | Interleaved          | Projected in CA, direct in SA   |
| SA+CA + adapter       | Interleaved          | Expert-owned projection in both |

Each variant was trained with seeds `0` and `1` for 100,000 steps at batch size 64. Training used a frozen SmolVLM backbone and a trainable action expert on [`lerobot/libero`](https://huggingface.co/datasets/lerobot/libero) at revision `a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4`. The recipe follows the [SmolVLA simulation setup](https://arxiv.org/html/2506.01844) and the additional [reproduction settings discussed by the LeRobot community](https://github.com/huggingface/lerobot/issues/3287#issuecomment-5076885804).

Every checkpoint was evaluated on all 40 LIBERO tasks using evaluation seed `1000`, with 50 episodes per task and 2,000 episodes in total. Flow matching used 10 denoising steps. The policy executed 10 actions from each predicted chunk before taking a new observation. This reduces evaluation time and was the best-performing setting in the paper's action-execution ablation. It differs from the paper's main simulation protocol, which took a new observation after every action.

For the two leading variants, I repeated the complete 2,000-episode evaluation with seed `2000` for both training seeds. This checks whether their ranking depends on a particular set of evaluation conditions.

The implementation, training configurations, scripts and raw results are stored in [`ablation_1_kv_adapter_self`](ablation_1_kv_adapter_self) and [`ablation_2_no_kv_proj_cross`](ablation_2_no_kv_proj_cross).

## Results

### Reproducing the original ranking

Before examining the prefix K/V interface, I compared the three unmodified architectures with [Table 6 of the SmolVLA paper](https://arxiv.org/html/2506.01844). Our values are averages over training seeds `0` and `1`, with 2,000 evaluation episodes per checkpoint.

| Architecture | Source | Spatial | Object | Goal | Long | Average |
| ------------ | ------ | ------: | -----: | ---: | ---: | ------: |
| CA           | Paper  |    87.0 |   92.0 | 83.0 | 54.0 |   79.00 |
| CA           | Ours   |    81.8 |   88.6 | 87.8 | 69.0 |   81.80 |
| SA           | Paper  |    80.0 |   94.0 | 84.0 | 40.0 |   74.50 |
| SA           | Ours   |    79.8 |   86.7 | 88.4 | 56.5 |   77.85 |
| SA+CA        | Paper  |    86.0 |   99.0 | 90.0 | 67.0 |   85.50 |
| SA+CA        | Ours   |    86.5 |   93.5 | 90.4 | 70.1 |   85.13 |

The original ranking was preserved independently in both training runs. Seed `0` produced SA+CA at 84.45%, CA at 82.60% and SA at 76.70%. Seed `1` produced SA+CA at 85.80%, CA at 81.00% and SA at 79.00%.

The margins differ from the paper, but both seeds reproduce its main result: interleaving SA and CA performs best, CA comes second and pure SA performs worst. The two-seed interleaved average was 85.13%, compared with 85.5% reported in the paper.

This is a close qualitative reproduction rather than an exact numerical one. Our evaluation uses 50 instead of 10 episodes per task and executes 10 actions before updating the observation. The largest numerical difference appears on LIBERO Long. What matters for the ablations is that the relevant architectural ordering is recovered before changing the prefix K/V interface.

### Effect of the prefix K/V projection

The table below reports success rate in percent for evaluation seed `1000`, averaged over training seeds `0` and `1`. Each suite contains 10 tasks with 50 episodes per task, so every suite value averages 1,000 episodes across the two checkpoints. `Average` covers all 40 tasks and 4,000 episodes. Because the suites are equal in size, it is also the equally weighted mean of the four suite rates. Long denotes the 10 LIBERO-10 tasks.

| Architecture          |  Spatial |   Object |     Goal |     Long |   Average |
| --------------------- | -------: | -------: | -------: | -------: | --------: |
| CA                    |     81.8 |     88.6 |     87.8 |     69.0 |     81.80 |
| CA without projection |     78.1 |     86.3 |     86.9 |     57.2 |     77.13 |
| SA                    |     79.8 |     86.7 |     88.4 |     56.5 |     77.85 |
| SA + adapter          |     83.1 |     89.2 | **91.5** | **72.7** |     84.13 |
| SA+CA                 | **86.5** | **93.5** |     90.4 |     70.1 | **85.13** |
| SA+CA + adapter       |     85.6 |     89.3 |     89.7 |     70.9 |     83.88 |

Adding the projection raised SA from 77.85% to 84.13%, a gain of 6.28 points. The effect was present in both training runs. Seed `0` rose from 76.70% to 82.90%, a gain of 6.20 points. Seed `1` rose from 79.00% to 85.35%, a gain of 6.35 points. The projection improved SA on every suite, with the largest change on Long. SA + adapter outperformed CA overall and nearly matched SA+CA. It remained behind SA+CA on Spatial and Object, but moved ahead on Goal and Long.

Removing the projection lowered CA from 81.80% to 77.13%, a drop of 4.68 points. The effect was also present in both training runs. Seed `0` fell from 82.60% to 79.30%, a drop of 3.30 points. Seed `1` fell from 81.00% to 74.95%, a drop of 6.05 points. Removing the projection reduced CA on every suite, with the largest change on Long. After removal, CA at 77.13% closely matched the original SA result of 77.85% and followed a similar suite-level pattern.

Adding the projection to the SA layers of the interleaved model lowered its average from 85.13% to 83.88%, a drop of 1.25 points. The largest change appeared on Object, which fell from 93.5% to 89.3%. Unlike pure SA, the interleaved model already contains an expert-owned prefix transformation in every CA layer, so it was not missing the trainable VLM-to-expert interface supplied by the adapter.

Taken together, the two reciprocal interventions show that performance follows the prefix K/V interface rather than the attention topology alone. Adding the projection removes the original SA disadvantage, while removing it eliminates the CA advantage.

### Unexpected performance on Long tasks

Long performance was not the target of this study, but it produced the largest and most consistent interface effects. Each entry below is the success rate over 500 Long episodes for one checkpoint. The mean therefore covers 1,000 episodes.

| Architecture          | Training seed 0 | Training seed 1 |     Mean |
| --------------------- | --------------: | --------------: | -------: |
| CA                    |            69.2 |            68.8 |     69.0 |
| CA without projection |            56.6 |            57.8 |     57.2 |
| SA                    |            54.8 |            58.2 |     56.5 |
| **SA + adapter**      |        **71.6** |        **73.8** | **72.7** |
| SA+CA                 |            68.8 |            71.4 |     70.1 |
| SA+CA + adapter       |            71.4 |            70.4 |     70.9 |

Adding the projection raised SA on Long from 56.5% to 72.7%, a gain of 16.20 points. Removing the projection lowered CA from 69.0% to 57.2%, a drop of 11.80 points. These were the largest suite-level effects of both interventions. SA + adapter ranked first on Long for both training seeds.

One possible interpretation is that longer tasks depend more heavily on making the VLM context useful to the action expert throughout the trajectory. Both directions are consistent with this interpretation because Long performance rises sharply when the trainable interface is added to SA and falls sharply when it is removed from CA. These experiments do not directly test the underlying mechanism, so this remains a hypothesis raised by the results.

The consistency of the Long result suggests that pure SA + adapter should not be discarded in future comparisons, even though the original interleaved model remains slightly stronger overall.

### Robustness to evaluation seed

The two leading models were evaluated again with evaluation seed `2000`. Each `Overall` value below covers all 40 tasks and 2,000 episodes. Each `Long` value covers 10 tasks and 500 episodes. The mean row averages four evaluations per model, covering 8,000 overall episodes and 2,000 Long episodes.

| Training seed | Evaluation seed | SA + adapter Overall | SA+CA Overall | SA + adapter Long | SA+CA Long |
| ------------: | --------------: | -------------------: | ------------: | ----------------: | ---------: |
|             0 |            1000 |                82.90 |     **84.45** |          **71.6** |       68.8 |
|             0 |            2000 |                83.60 |     **84.55** |          **72.2** |       68.6 |
|             1 |            1000 |                85.35 |     **85.80** |          **73.8** |       71.4 |
|             1 |            2000 |            **85.50** |         85.10 |          **73.8** |       70.6 |
|          Mean |             N/A |                84.34 |     **84.98** |         **72.85** |      69.85 |

The evaluation seed changed the overall result of any checkpoint by at most 0.70 points. SA+CA ranked first overall in three comparisons, while SA + adapter ranked first once. Across the four evaluations, SA+CA averaged 84.98% and SA + adapter averaged 84.34%.

The suite pattern also remained similar. SA + adapter remained lower than SA+CA on Spatial and Object, but higher on Goal and Long. It won all four matched Long comparisons, averaging 72.85% compared with 69.85% for SA+CA.

## Conclusion

The original interleaved architecture remains the safest default. It reached 85.13% overall, compared with 84.13% for SA + adapter and retained a consistent advantage on Spatial and Object.

Adding the expert-owned prefix projection raised pure SA from 77.85% to 84.13%. Removing the corresponding projection lowered pure CA from 81.80% to 77.13%. CA without projection therefore closely matched the original SA model.

Together, the two ablations confirm that the asymmetric prefix K/V interface explains the poor relative SA performance in the original attention-layout comparison. Adding the projection removes the SA deficit, while removing it eliminates the CA advantage. The original comparison changed both attention topology and the trainable VLM-to-expert interface, so its ranking does not isolate the effect of attention topology.

The Long result makes the adapted SA variant especially worth following up. It reached 72.7% compared with 70.1% for the original interleaved model and ranked first for both training seeds. More seeds and other long-horizon benchmarks are needed to determine whether this is a general advantage.

## Repeatability

The implementation targets LeRobot commit `713a409faedd73bb5597481b8885f17fbee23330`. The SA adapter implementation and parity test are provided in [`ablation_1_kv_adapter_self/patches`](ablation_1_kv_adapter_self/patches). The CA projection removal is provided in [`ablation_2_no_kv_proj_cross/patches`](ablation_2_no_kv_proj_cross/patches).

The exact training commands, saved configurations and raw evaluation results are stored in [`ablation_1_kv_adapter_self`](ablation_1_kv_adapter_self) and [`ablation_2_no_kv_proj_cross`](ablation_2_no_kv_proj_cross).

Each evaluation JSON contains outcomes for all 40 tasks, with 50 episodes per task, together with per-suite and overall aggregates. The tables above can therefore be recomputed directly from the checked-in episode-level results.
