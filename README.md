# GAT-for-PPI
The code for [Graph Attention Networks](https://arxiv.org/pdf/1710.10903.pdf) with the script compatible with PPI. 

I notice that many ones show off their codes for GAT in tf or pytorch while few people implement the part for PPI. It is an important thing to validate this type of powerful GNNs on the large-scale dataset and know its limit, and make a progress. Based on this, I fork [GAT](https://github.com/PetarV-/GAT) in the tf-version and implement this part for others to use or compare.

## Dependencies
Similar to the original repository, the script has been tested running under Python 3.5.2, with the following packages installed (along with their dependencies):

- `numpy==1.14.1`
- `scipy==1.0.0`
- `networkx==2.1`
- `tensorflow-gpu==1.6.0`

In addition, CUDA 9.0 and cuDNN 7 have been used.

## Codes

1. execute.py is for Cora and Citeseer. (There might be memory error for Pubmed.)

2. execute_sparse.py is for Cora, Citeseer and Pubmed via the SparseTensor implementation in tf.

3. execute_inductive.py is for PPI, a large scale dataset.

#### Running demo.
```
CUDA_VISIBLE_DEVICES=0, python execute.py; python execute_sparse.py; python execute_inductive.py
```

#### How about the Reddit dataset?

According to reddit, it is quite hard to implement the experiment of the Reddit dataset based on GAT. Two following reasons,

1. You have to decompose a large Reddit graph into too many small graphs like DFS decomposition in GAT. And when you algin the size of the sub-graphs, you will find that it consumes too much memory. One trade-off is limit the size of each sub-graph without guaranttee of the isolated condition such as 10000 sub-graphs with the adj of size 1000x1000.

2. The train/val/test split is different from PPI since PPI does this in the graph level when reddit does this in the node level according to the timeline. You cannot reuse the preprocess script for PPI and also you have to change the code in the execute_inductive.py.


## Reference
```
@article{
  velickovic2018graph,
  title="{Graph Attention Networks}",
  author={Veli{\v{c}}kovi{\'{c}}, Petar and Cucurull, Guillem and Casanova, Arantxa and Romero, Adriana and Li{\`{o}}, Pietro and Bengio, Yoshua},
  journal={International Conference on Learning Representations},
  year={2018},
  url={https://openreview.net/forum?id=rJXMpikCZ},
  note={accepted as poster},
}
