# Model comparison

| action | comparison | metric | base | players | diff | players_smooth |
| --- | --- | --- | --- | --- | --- | --- |
| pick | base_vs_players | top1 | 0.0836066 | 0.129508 | 0.0459016 |  |
| pick | base_vs_players | top3 | 0.232787 | 0.303279 | 0.0704918 |  |
| pick | base_vs_players | top5 | 0.313115 | 0.431148 | 0.118033 |  |
| pick | base_vs_players | top10 | 0.47377 | 0.603279 | 0.129508 |  |
| pick | base_vs_players | mean_rank | 20.618 | 14.0721 | -6.5459 |  |
| pick | base_vs_players | mrr | 0.208062 | 0.275042 | 0.0669797 |  |
| pick | players_vs_players_smooth | top1 |  | 0.129508 | -0.0213115 | 0.108197 |
| pick | players_vs_players_smooth | top3 |  | 0.303279 | -0.0311475 | 0.272131 |
| pick | players_vs_players_smooth | top5 |  | 0.431148 | -0.0311475 | 0.4 |
| pick | players_vs_players_smooth | top10 |  | 0.603279 | -0.00819672 | 0.595082 |
| pick | players_vs_players_smooth | mean_rank |  | 14.0721 | 0.814754 | 14.8869 |
| pick | players_vs_players_smooth | mrr |  | 0.275042 | -0.0247355 | 0.250306 |
| ban | base_vs_players | top1 | 0.129977 | 0.194379 | 0.0644028 |  |
| ban | base_vs_players | top3 | 0.313817 | 0.368852 | 0.0550351 |  |
| ban | base_vs_players | top5 | 0.396956 | 0.497658 | 0.100703 |  |
| ban | base_vs_players | top10 | 0.539813 | 0.689696 | 0.149883 |  |
| ban | base_vs_players | mean_rank | 18.4157 | 11.0316 | -7.38407 |  |
| ban | base_vs_players | mrr | 0.265883 | 0.338816 | 0.0729329 |  |
| ban | players_vs_players_smooth | top1 |  | 0.194379 | -0.0480094 | 0.14637 |
| ban | players_vs_players_smooth | top3 |  | 0.368852 | -0.0316159 | 0.337237 |
| ban | players_vs_players_smooth | top5 |  | 0.497658 | -0.029274 | 0.468384 |
| ban | players_vs_players_smooth | top10 |  | 0.689696 | -0.0456674 | 0.644028 |
| ban | players_vs_players_smooth | mean_rank |  | 11.0316 | 0.542155 | 11.5738 |
| ban | players_vs_players_smooth | mrr |  | 0.338816 | -0.0376918 | 0.301124 |
