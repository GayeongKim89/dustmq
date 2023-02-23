# RAKGE
This is the code of paper 
**Exploiting Relation-aware Attribute Representation Learning in Knowledge Graph Embedding for Numerical Reasoning**. 


## Experiment Environment
- python 3.7+
- torch 1.9+
- dgl 0.7


## Usage
    cd RAKGE

### Preprocessing
    python preprocessing_kg_num_lit.py --dataset {credit, spotify}

### Reproducing Paper's Experiments
#### RAKGE
    python run.py --gpu 0 --n_layer 0  --literal --init_dim 200 --att_dim 200 --head_num 5 --name RAKGE --scale 0.25 --data {credit, spotify} --input_drop 0.7 

#### TransE
    python run.py --gpu 0 --n_layer 0 --init_dim 200 --name lte --score_func transe --opn mult --x_ops "d" --hid_drop 0.7  --data {credit,spotify}
    
#### LiteralE
    python run.py --gpu 0 --n_layer 0 --literal --init_dim 200 --name TransELiteral_gate --data {credit, spotify} --input_drop 0.7 
   
#### R-GCN
    python run.py --gpu 0 --n_layer 1 --score_func transe --opn mult --gcn_dim 150 --init_dim 150 --num_base 5 --encoder rgcn --name repro --data {credit, spotify} --hid_drop 0.7
   
 


## Acknowledgement
We refer to the code of [LTE-KGC](https://github.com/MIRALab-USTC/GCN4KGC) and [LiteralE](https://github.com/SmartDataAnalytics/LiteralE). Thanks for their contributions.
