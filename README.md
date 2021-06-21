# MCLYang-ORIE5250_project
### Colab link: 
```
https://colab.research.google.com/drive/1sSFy5spf6zPsYpEEQMNFNSwALKI15uTy?authuser=1
```

### Run assortment optimization and get the expected revenue:(Ex.clusters=3) 
```
python assortment.py --NUM_CLUSTERS=3
```
### Results:

```
(assortment.py:720): Gdk-CRITICAL **: 22:16:27.786: gdk_cursor_new_for_display: assertion 'GDK_IS_DISPLAY (display)' failed
Using license file /home/malcolm/gurobi.lic
Academic license - for non-commercial use only
Gurobi Optimizer version 9.0.2 build v9.0.2rc0 (linux64)
Optimize a model with 327 rows, 111 columns and 813 nonzeros
Model fingerprint: 0xce20f05e
Variable types: 84 continuous, 27 integer (27 binary)
Coefficient statistics:
  Matrix range     [1e-03, 1e+00]
  Objective range  [8e-02, 5e-01]
  Bounds range     [1e+00, 1e+00]
  RHS range        [9e-02, 9e-02]
Found heuristic solution: objective 0.0000000
Presolve removed 81 rows and 0 columns
Presolve time: 0.00s
Presolved: 246 rows, 111 columns, 771 nonzeros
Variable types: 84 continuous, 27 integer (27 binary)

Root relaxation: objective -4.946215e-02, 65 iterations, 0.00 seconds

    Nodes    |    Current Node    |     Objective Bounds      |     Work
 Expl Unexpl |  Obj  Depth IntInf | Incumbent    BestBd   Gap | It/Node Time

     0     0   -0.04946    0    6    0.00000   -0.04946      -     -    0s
H    0     0                      -0.0349727   -0.04946  41.4%     -    0s
     0     0   -0.04462    0    8   -0.03497   -0.04462  27.6%     -    0s
     0     0   -0.04462    0    6   -0.03497   -0.04462  27.6%     -    0s
H    0     0                      -0.0361598   -0.04462  23.4%     -    0s
H    0     0                      -0.0374148   -0.04352  16.3%     -    0s
H    0     0                      -0.0392840   -0.04352  10.8%     -    0s
     0     0   -0.04202    0    7   -0.03928   -0.04202  6.98%     -    0s
     0     0   -0.03928    0    7   -0.03928   -0.03928  0.00%     -    0s

Cutting planes:
  Gomory: 6
  MIR: 5
  Flow cover: 1

Explored 1 nodes (123 simplex iterations) in 0.02 seconds
Thread count was 8 (of 8 available processors)

Solution count 5: -0.039284 -0.0374148 -0.0361598 ... 0

Optimal solution found (tolerance 1.00e-04)
Best objective -3.928403242196e-02, best bound -3.928403242196e-02, gap 0.0000%
******************************
******************************
**********Assortment OPT,data1.csv,IP**********
******************************
******************************
displaye #0 item.
displaye #1 item.
displaye #2 item.
displaye #3 item.
Suppose unknown the customers type, the displayed canidates are: {0, 1, 2, 3}
Mixture Revenue: 78.91427132747182
Suppose known the customers from cluster0
Cluster center:
srch_booking_window         25.298003
srch_adults_count            2.393241
srch_children_count          0.632873
srch_room_count              2.000000
srch_saturday_night_bool     0.548387
Name: 0, dtype: float64
the displayed canidates are: {0, 1, 2, 3, 4, 5}
Revenue: 87.85705868756958
Revenue[Mixture canidates]: 63.53856799386406
Suppose known the customers from cluster1
Cluster center:
srch_booking_window         2.107061e+01
srch_adults_count           1.761949e+00
srch_children_count         2.602529e-01
srch_room_count             1.000000e+00
srch_saturday_night_bool    3.230749e-14
Name: 1, dtype: float64
the displayed canidates are: {0, 1, 2, 3, 4, 5}
Revenue: 89.32008698228921
Revenue[Mixture canidates]: 64.81520743258197
Suppose known the customers from cluster2
Cluster center:
srch_booking_window         20.824439
srch_adults_count            1.897534
srch_children_count          0.285650
srch_room_count              1.000000
srch_saturday_night_bool     1.000000
Name: 2, dtype: float64
the displayed canidates are: {0, 1, 2, 3}
Revenue: 91.41042538157642
Revenue[Mixture canidates]: 91.41042538157642
Gurobi Optimizer version 9.0.2 build v9.0.2rc0 (linux64)
Optimize a model with 351 rows, 119 columns and 873 nonzeros
Model fingerprint: 0xe80ed61c
Variable types: 90 continuous, 29 integer (29 binary)
Coefficient statistics:
  Matrix range     [3e-03, 2e+00]
  Objective range  [8e-02, 5e-01]
  Bounds range     [1e+00, 1e+00]
  RHS range        [5e-01, 5e-01]
Found heuristic solution: objective 0.0000000
Presolve removed 87 rows and 0 columns
Presolve time: 0.00s
Presolved: 264 rows, 119 columns, 786 nonzeros
Variable types: 90 continuous, 29 integer (29 binary)

Root relaxation: objective -2.953204e-01, 152 iterations, 0.00 seconds

    Nodes    |    Current Node    |     Objective Bounds      |     Work
 Expl Unexpl |  Obj  Depth IntInf | Incumbent    BestBd   Gap | It/Node Time

     0     0   -0.29532    0    6    0.00000   -0.29532      -     -    0s
H    0     0                      -0.1706713   -0.29532  73.0%     -    0s
H    0     0                      -0.2199724   -0.29532  34.3%     -    0s
     0     0   -0.28413    0    8   -0.21997   -0.28413  29.2%     -    0s
H    0     0                      -0.2468657   -0.28413  15.1%     -    0s
H    0     0                      -0.2666276   -0.28413  6.56%     -    0s
     0     0   -0.26663    0    8   -0.26663   -0.26663  0.00%     -    0s

Cutting planes:
  Gomory: 3
  MIR: 3
  Flow cover: 1

Explored 1 nodes (163 simplex iterations) in 0.01 seconds
Thread count was 8 (of 8 available processors)

Solution count 5: -0.266628 -0.246866 -0.219972 ... 0

Optimal solution found (tolerance 1.00e-04)
Best objective -2.666275778771e-01, best bound -2.666275778771e-01, gap 0.0000%
******************************
******************************
**********Assortment OPT,data2.csv,IP**********
******************************
******************************
displaye #0 item.
displaye #1 item.
displaye #2 item.
Suppose unknown the customers type, the displayed canidates are: {0, 1, 2}
Mixture Revenue: 140.83449996698465
Suppose known the customers from cluster0
Cluster center:
srch_booking_window         25.298003
srch_adults_count            2.393241
srch_children_count          0.632873
srch_room_count              2.000000
srch_saturday_night_bool     0.548387
Name: 0, dtype: float64
the displayed canidates are: {0, 1, 2}
Revenue: 110.25557144424836
Revenue[Mixture canidates]: 110.25557144424836
Suppose known the customers from cluster1
Cluster center:
srch_booking_window         2.107061e+01
srch_adults_count           1.761949e+00
srch_children_count         2.602529e-01
srch_room_count             1.000000e+00
srch_saturday_night_bool    3.230749e-14
Name: 1, dtype: float64
the displayed canidates are: {0, 1, 2}
Revenue: 112.36116215823003
Revenue[Mixture canidates]: 112.36116215823003
Suppose known the customers from cluster2
Cluster center:
srch_booking_window         20.824439
srch_adults_count            1.897534
srch_children_count          0.285650
srch_room_count              1.000000
srch_saturday_night_bool     1.000000
Name: 2, dtype: float64
the displayed canidates are: {0, 1, 2}
Revenue: 166.00174592709507
Revenue[Mixture canidates]: 166.00174592709507
Gurobi Optimizer version 9.0.2 build v9.0.2rc0 (linux64)
Optimize a model with 315 rows, 107 columns and 783 nonzeros
Model fingerprint: 0x7e182c8f
Variable types: 81 continuous, 26 integer (26 binary)
Coefficient statistics:
  Matrix range     [1e-03, 3e+00]
  Objective range  [8e-02, 5e-01]
  Bounds range     [1e+00, 1e+00]
  RHS range        [3e+00, 3e+00]
Found heuristic solution: objective 0.0000000
Presolve removed 78 rows and 0 columns
Presolve time: 0.00s
Presolved: 237 rows, 107 columns, 705 nonzeros
Variable types: 81 continuous, 26 integer (26 binary)

Root relaxation: objective -2.686297e-01, 43 iterations, 0.00 seconds

    Nodes    |    Current Node    |     Objective Bounds      |     Work
 Expl Unexpl |  Obj  Depth IntInf | Incumbent    BestBd   Gap | It/Node Time

     0     0   -0.26863    0   10    0.00000   -0.26863      -     -    0s
H    0     0                      -0.0912819   -0.26863   194%     -    0s
H    0     0                      -0.1790596   -0.24426  36.4%     -    0s
     0     0   -0.23191    0   12   -0.17906   -0.23191  29.5%     -    0s
     0     0   -0.22548    0   12   -0.17906   -0.22548  25.9%     -    0s
     0     0   -0.21658    0   10   -0.17906   -0.21658  21.0%     -    0s
     0     0   -0.21185    0   11   -0.17906   -0.21185  18.3%     -    0s
H    0     0                      -0.1790689   -0.21185  18.3%     -    0s
H    0     0                      -0.1813119   -0.19395  6.97%     -    0s
     0     0   -0.19353    0   10   -0.18131   -0.19353  6.74%     -    0s
     0     0   -0.19224    0   10   -0.18131   -0.19224  6.03%     -    0s
     0     2   -0.19224    0   10   -0.18131   -0.19224  6.03%     -    0s

Cutting planes:
  Gomory: 9
  Implied bound: 3
  MIR: 7
  Flow cover: 20
  RLT: 10
  Relax-and-lift: 1
  BQP: 3

Explored 38 nodes (435 simplex iterations) in 0.05 seconds
Thread count was 8 (of 8 available processors)

Solution count 5: -0.181312 -0.179069 -0.17906 ... 0

Optimal solution found (tolerance 1.00e-04)
Best objective -1.813119328283e-01, best bound -1.813119328283e-01, gap 0.0000%
******************************
******************************
**********Assortment OPT,data3.csv,IP**********
******************************
******************************
displaye #0 item.
displaye #1 item.
displaye #2 item.
displaye #3 item.
displaye #4 item.
displaye #5 item.
displaye #6 item.
displaye #7 item.
Suppose unknown the customers type, the displayed canidates are: {0, 1, 2, 3, 4, 5, 6, 7}
Mixture Revenue: 111.78087844069219
Suppose known the customers from cluster0
Cluster center:
srch_booking_window         25.298003
srch_adults_count            2.393241
srch_children_count          0.632873
srch_room_count              2.000000
srch_saturday_night_bool     0.548387
Name: 0, dtype: float64
the displayed canidates are: {0, 1, 2, 3, 4, 5, 6, 7}
Revenue: 98.75304998533511
Revenue[Mixture canidates]: 98.75304998533511
Suppose known the customers from cluster1
Cluster center:
srch_booking_window         2.107061e+01
srch_adults_count           1.761949e+00
srch_children_count         2.602529e-01
srch_room_count             1.000000e+00
srch_saturday_night_bool    3.230749e-14
Name: 1, dtype: float64
the displayed canidates are: {0, 1, 2, 3, 4, 5, 6, 7, 8}
Revenue: 98.53789599539786
Revenue[Mixture canidates]: 90.86364665964582
Suppose known the customers from cluster2
Cluster center:
srch_booking_window         20.824439
srch_adults_count            1.897534
srch_children_count          0.285650
srch_room_count              1.000000
srch_saturday_night_bool     1.000000
Name: 2, dtype: float64
the displayed canidates are: {0, 1, 2, 3, 4, 5, 6, 7}
Revenue: 128.8920217120758
Revenue[Mixture canidates]: 128.8920217120758
Gurobi Optimizer version 9.0.2 build v9.0.2rc0 (linux64)
Optimize a model with 327 rows, 111 columns and 813 nonzeros
Model fingerprint: 0x39dbacba
Variable types: 84 continuous, 27 integer (27 binary)
Coefficient statistics:
  Matrix range     [2e-03, 2e+00]
  Objective range  [8e-02, 5e-01]
  Bounds range     [1e+00, 1e+00]
  RHS range        [3e-01, 3e-01]
Found heuristic solution: objective 0.0000000
Presolve removed 243 rows and 72 columns
Presolve time: 0.00s
Presolved: 84 rows, 39 columns, 270 nonzeros
Variable types: 30 continuous, 9 integer (9 binary)

Root relaxation: objective -8.802178e-02, 12 iterations, 0.00 seconds

    Nodes    |    Current Node    |     Objective Bounds      |     Work
 Expl Unexpl |  Obj  Depth IntInf | Incumbent    BestBd   Gap | It/Node Time

     0     0   -0.08802    0    1    0.00000   -0.08802      -     -    0s
H    0     0                      -0.0818722   -0.08802  7.51%     -    0s

Explored 1 nodes (12 simplex iterations) in 0.00 seconds
Thread count was 8 (of 8 available processors)

Solution count 2: -0.0818722 0 

Optimal solution found (tolerance 1.00e-04)
Best objective -8.187224686816e-02, best bound -8.187224686816e-02, gap 0.0000%
******************************
******************************
**********Assortment OPT,data4.csv,IP**********
******************************
******************************
displaye #0 item.
Suppose unknown the customers type, the displayed canidates are: {0}
Mixture Revenue: 53.933452350668944
Suppose known the customers from cluster0
Cluster center:
srch_booking_window         25.298003
srch_adults_count            2.393241
srch_children_count          0.632873
srch_room_count              2.000000
srch_saturday_night_bool     0.548387
Name: 0, dtype: float64
the displayed canidates are: {0}
Revenue: 42.35694473942674
Revenue[Mixture canidates]: 42.35694473942674
Suppose known the customers from cluster1
Cluster center:
srch_booking_window         2.107061e+01
srch_adults_count           1.761949e+00
srch_children_count         2.602529e-01
srch_room_count             1.000000e+00
srch_saturday_night_bool    3.230749e-14
Name: 1, dtype: float64
the displayed canidates are: {0}
Revenue: 40.510779265013895
Revenue[Mixture canidates]: 40.510779265013895
Suppose known the customers from cluster2
Cluster center:
srch_booking_window         20.824439
srch_adults_count            1.897534
srch_children_count          0.285650
srch_room_count              1.000000
srch_saturday_night_bool     1.000000
Name: 2, dtype: float64
the displayed canidates are: {0}
Revenue: 65.38323604387477
Revenue[Mixture canidates]: 65.38323604387477
```
