# Отчёт по первой ML-модели драфта, patch 60

## 1. Проверка таблиц

- draft_events: (8232, 24)

- draft_states: (8232, 27)

- draft_candidates_pick: (385189, 24)

- draft_candidates_ban: (565607, 24)


### Число действий в матчах


```text
 matches_count
           343
```


Неполных матчей не обнаружено.


## 2. Train / valid / test


### pick

- train_shape: (269520, 24)

- valid_shape: (57273, 24)

- test_shape: (58396, 24)

- train_states: 2400

- valid_states: 510

- test_states: 520

- train_matches: 240

- valid_matches: 51

- test_matches: 52


### ban

- train_shape: (395760, 24)

- valid_shape: (84099, 24)

- test_shape: (85748, 24)

- train_states: 3360

- valid_states: 714

- test_states: 728

- train_matches: 240

- valid_matches: 51

- test_matches: 52


## 3. Метрики CatBoost


### pick


```text
 states     top1     top3     top5    top10  mean_rank      mrr
    520 0.119231 0.263462 0.346154 0.490385  18.261538 0.241502
```


### ban


```text
 states     top1     top3     top5    top10  mean_rank      mrr
    728 0.104396 0.255495 0.322802 0.541209  15.081044 0.234338
```


## 4. Сравнение с baseline


### pick


```text
            model  states     top1     top3     top5    top10  mean_rank      mrr
           random     520 0.013462 0.026923 0.048077 0.101923  56.655769 0.051172
global_popularity     520 0.109615 0.196154 0.278846 0.428846  23.011538 0.209294
  team_popularity     520 0.059615 0.151923 0.219231 0.380769  23.034615 0.161600
         catboost     520 0.119231 0.263462 0.346154 0.490385  18.261538 0.241502
```


### ban


```text
            model  states     top1     top3     top5    top10  mean_rank      mrr
           random     728 0.016484 0.034341 0.050824 0.090659  59.431319 0.052613
global_popularity     728 0.096154 0.199176 0.270604 0.473901  18.054945 0.207613
  team_popularity     728 0.087912 0.192308 0.259615 0.407967  20.211538 0.192449
         catboost     728 0.104396 0.255495 0.322802 0.541209  15.081044 0.234338
```


## 5. Метрики по order


### pick


```text
 states     top1     top3     top5    top10  mean_rank      mrr  order
     52 0.173077 0.461538 0.634615 0.769231   7.769231 0.365227      8
     52 0.173077 0.423077 0.538462 0.653846  12.000000 0.335588      9
     52 0.173077 0.423077 0.519231 0.653846   9.903846 0.343035     13
     52 0.115385 0.403846 0.442308 0.538462  16.153846 0.291294     14
     52 0.134615 0.230769 0.288462 0.403846  18.807692 0.235864     15
     52 0.134615 0.230769 0.288462 0.384615  23.942308 0.215742     16
     52 0.076923 0.115385 0.153846 0.269231  24.826923 0.145067     17
     52 0.076923 0.115385 0.192308 0.423077  17.923077 0.168477     18
     52 0.057692 0.134615 0.211538 0.384615  26.134615 0.152933     23
     52 0.076923 0.096154 0.192308 0.423077  25.153846 0.161796     24
```


### ban


```text
 states     top1     top3     top5    top10  mean_rank      mrr  order
     52 0.134615 0.365385 0.403846 0.576923   9.788462 0.285589      1
     52 0.076923 0.288462 0.326923 0.557692  12.057692 0.231395      2
     52 0.211538 0.384615 0.423077 0.711538  10.269231 0.340505      3
     52 0.134615 0.442308 0.480769 0.653846   9.884615 0.328718      4
     52 0.057692 0.269231 0.384615 0.557692  15.230769 0.202629      5
     52 0.230769 0.384615 0.500000 0.750000   8.211538 0.373005      6
     52 0.211538 0.326923 0.384615 0.673077  12.903846 0.319505      7
     52 0.038462 0.173077 0.230769 0.442308  21.250000 0.156031     10
     52 0.057692 0.173077 0.230769 0.480769  21.480769 0.174268     11
     52 0.115385 0.230769 0.384615 0.615385  16.403846 0.240745     12
     52 0.019231 0.153846 0.173077 0.269231  17.461538 0.140223     19
     52 0.038462 0.076923 0.153846 0.461538  17.076923 0.141489     20
     52 0.057692 0.096154 0.211538 0.423077  19.000000 0.158573     21
     52 0.076923 0.211538 0.230769 0.403846  20.115385 0.188057     22
```


## 6. Метрики по draft_phase


### pick


```text
 states     top1     top3     top5    top10  mean_rank      mrr  draft_phase
     52 0.173077 0.461538 0.634615 0.769231   7.769231 0.365227            1
    260 0.146154 0.342308 0.415385 0.526923  16.161538 0.284305            2
    208 0.072115 0.115385 0.187500 0.375000  23.509615 0.157068            3
```


### ban


```text
 states     top1     top3     top5    top10  mean_rank      mrr  draft_phase
    364 0.151099 0.351648 0.414835 0.640110  11.192308 0.297335            1
    156 0.070513 0.192308 0.282051 0.512821  19.711538 0.190348            2
    208 0.048077 0.134615 0.192308 0.389423  18.413462 0.157086            3
```


## 7. Feature importance top-20


### pick


```text
                Feature Id  Importances
       candidate_pick_rate    25.394896
  candidate_matches_played    19.303710
                     order     9.195711
               draft_phase     6.929992
        candidate_ban_rate     6.306989
       n_enemy_bans_before     5.797928
candidate_pick_or_ban_rate     5.516220
         candidate_hero_id     4.133121
      n_enemy_picks_before     3.380061
            acting_team_id     3.331273
      available_hero_count     2.842803
         candidate_winrate     1.646365
               league_name     1.597676
        n_ally_bans_before     1.399356
               acting_side     1.393687
       n_ally_picks_before     1.372729
          opponent_team_id     0.457483
                     patch     0.000000
```


### ban


```text
                Feature Id  Importances
        candidate_ban_rate    26.073820
                     order    21.077767
candidate_pick_or_ban_rate    15.419030
      n_enemy_picks_before     6.158256
       n_ally_picks_before     5.996935
         candidate_hero_id     4.807882
         candidate_winrate     4.727354
  candidate_matches_played     3.405300
          opponent_team_id     3.033891
       candidate_pick_rate     2.353722
       n_enemy_bans_before     2.175825
               draft_phase     1.652342
        n_ally_bans_before     1.075942
            acting_team_id     0.988346
      available_hero_count     0.942135
               acting_side     0.092365
               league_name     0.019085
                     patch     0.000000
```
