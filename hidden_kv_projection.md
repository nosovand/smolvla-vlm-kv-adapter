# Direct projection from VLM hidden states

A final ablation of the interface between frozen SmolVLM features and the SmolVLA action expert.

The previous experiments showed that the action expert benefits from adapting the pretrained VLM K/V representation. Adding an expert-owned projection improved SA, while removing the corresponding projection harmed CA. This raised a further question. Could the expert do better by extracting the prefix K/V directly from the full VLM hidden states instead of receiving keys and values already produced by the frozen VLM?

## Direct hidden-state projection

The final ablation replaces the previous two-stage interface with a single expert-owned projection. Rather than first passing the VLM hidden states through the frozen VLM K/V projections and then adapting those keys and values for the expert, the new path projects the VLM hidden states directly into expert prefix K/V.

This gives the action expert full control over which information it extracts from the VLM hidden states. It also removes the pretrained K/V representation that served as the input to the earlier adapter.

In pure SA, the direct projection is used in every expert layer. In SA+CA, it is used in the CA layers. The SA layers retain their original direct frozen VLM K/V interface.

Removing the frozen VLM K/V projections also required a choice about positional encoding. I tested assigning position zero to every projected prefix key and retaining the individual prefix-token positions through RoPE.

The shared position-zero variant has a natural interpretation in this setting. All VLM prefix features describe the same observation time, while the action tokens represent different future steps. Under this interpretation, the prefix acts as context for the complete action sequence rather than as a temporal sequence of its own. The individual-position variant instead preserves the standard tokenwise RoPE structure used by the VLM.

I also tested two initializations for the direct projections. Random initialization lets the expert learn the interface from scratch. VLM initialization copies the corresponding frozen VLM K/V projection weights into the new expert projections, allowing optimization to start from features already learned by the VLM.

## Experiment

I trained all combinations of the two positional variants and the two initialization variants for pure SA and SA+CA. Each model was trained for 100,000 steps with training seed `0` under the same training recipe used in the preceding interface experiments.

Every checkpoint was evaluated with seed `1000` on all 40 LIBERO tasks using 50 episodes per task. Each overall value therefore covers 2,000 episodes. Each Long value covers 500 episodes from the 10 LIBERO-10 tasks.

## Results

The table reports success rate in percent. `Individual RoPE` preserves the original position of each VLM prefix token. `Position zero` applies no positional rotation to the projected prefix keys. `VLM` initialization copies the pretrained VLM K/V projection weights, while `Random` uses the standard random initialization.

| Prefix key positions | Initialization | SA Overall | SA Long | SA+CA Overall | SA+CA Long |
| --- | --- | ---: | ---: | ---: | ---: |
| Position zero | Random | 76.75 | 58.0 | 77.45 | 54.4 |
| Position zero | VLM | 77.85 | 59.4 | 77.20 | 53.6 |
| Individual RoPE | Random | 79.50 | 60.6 | **81.50** | **63.0** |
| Individual RoPE | VLM | **81.25** | **61.8** | 78.20 | 58.2 |

Individual RoPE outperformed position zero in every matched overall comparison. The same ordering appeared on Long for both architectures and both initializations. Although treating all VLM features as context from one observation time made position zero conceptually appealing, preserving the individual prefix-token positions worked consistently better.

Initialization did not produce a consistent advantage across architectures. VLM initialization improved both SA variants, but reduced performance for both SA+CA variants. The best SA result used VLM initialization, while the best SA+CA result used random initialization. Starting from the pretrained VLM projection weights therefore did not provide a generally useful advantage and does not appear to be the main factor controlling performance in this ablation.

Even the strongest direct-projection variants remained below the earlier interfaces under the same training and evaluation seeds. For pure SA, the best direct projection reached 81.25% overall and 61.8% on Long. SA + adapter reached 82.90% overall and 71.6% on Long. For SA+CA, the best direct projection reached 81.50% overall and 63.0% on Long. The original interleaved model reached 84.45% overall and 68.8% on Long.

## Conclusion

Giving the action expert direct access to the VLM hidden states did not improve performance. The best variants preserved individual RoPE positions, even though assigning a shared position to features from the same observation time appeared conceptually reasonable. Projection initialization had no consistent effect across the two architectures.

The direct hidden-state path also performed worse than the earlier interfaces overall and produced particularly large losses on Long. More direct access to the VLM representation was therefore not beneficial under this setup.

These results indicate that the frozen VLM K/V projections are not merely an information bottleneck that restricts the action expert. They provide a useful intermediate representation. Among the tested interfaces, the stronger approach is to retain the pretrained VLM K/V representation and adapt it for the action expert rather than asking the expert to extract a new representation directly from the full hidden states.

This ablation used one training seed, so its exact margins are more exploratory than the two-seed SA and CA interface results. However, the consistent advantage of individual RoPE and the failure of every direct-projection variant to improve on the corresponding earlier interface support the same conclusion.

## Repeatability

The implementation targets LeRobot commit `713a409faedd73bb5597481b8885f17fbee23330`. The implementation patch, Slurm scripts, saved training configurations and raw evaluation results are stored in [`ablation_3_hidden_kv_proj`](ablation_3_hidden_kv_proj).

Each evaluation JSON contains outcomes for all 40 tasks, with 50 episodes per task, together with per-suite and overall aggregates. The table above can therefore be recomputed directly from the checked-in episode-level results.