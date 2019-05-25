#CUDA_VISIBLE_DEVICES=0,1 python execute.py cora
#CUDA_VISIBLE_DEVICES=0,1 python execute.py citeseer
#CUDA_VISIBLE_DEVICES=0,1,2,3 python execute.py pubmed
#CUDA_VISIBLE_DEVICES=0,1 python execute_sparse.py cora 0 
#CUDA_VISIBLE_DEVICES=0,1 python execute_sparse.py citeseer 0 
#CUDA_VISIBLE_DEVICES=0,1 python execute_sparse.py pubmed 0
#CUDA_VISIBLE_DEVICES=0,1 python execute_sparse.py cora 0.2
#CUDA_VISIBLE_DEVICES=0,1 python execute_sparse.py citeseer 0.2 
#CUDA_VISIBLE_DEVICES=0,1 python execute_sparse.py pubmed 0.2
#CUDA_VISIBLE_DEVICES=0,1 python execute_sparse.py cora 0.5
#CUDA_VISIBLE_DEVICES=0,1 python execute_sparse.py citeseer 0.5 
#CUDA_VISIBLE_DEVICES=0,1 python execute_sparse.py pubmed 0.5
#CUDA_VISIBLE_DEVICES=0,1 python execute_sparse.py cora 0.7
#CUDA_VISIBLE_DEVICES=0,1 python execute_sparse.py citeseer 0.7
#CUDA_VISIBLE_DEVICES=0,1 python execute_sparse.py pubmed 0.7
CUDA_VISIBLE_DEVICES=0,1 python execute_inductive.py ppi


##### note that, I have not implemented the code for reddit. It is a relative complex to modify the code for PPI.
#CUDA_VISIBLE_DEVICES=0,1 python execute_inductive.py reddit
