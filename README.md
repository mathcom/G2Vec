# G2Vec: Distributed gene representations for identification of cancer prognostic genes

G2Vec is a novel network-based deep learning method to identify prognostic gene signatures (biomarkers).

Please refer to included 'manual.pdf'.

For more detail, please refer to J.H. Choi, et al. "G2Vec: Distributed gene representations for identification of cancer prognostic genes" Scientific Reports 8.1 (2018): 1-10.

Latest update: 06 February 2020


--------------------------------------------------------------------------------------------
USAGE: 

	python G2Vec.py EXPRESSION_FILE CLINICAL_FILE NETWORK_FILE RESULT_NAME [optional parameters]

	
--------------------------------------------------------------------------------------------
example:

	$ python G2Vec.py ex_EXPRESSION.txt ex_CLINICAL.txt ex_NETWORK.txt ex_RESULT
	>>> 0. Arguments
	Namespace(CLINICAL_FILE='ex_CLINICAL.txt', EXPRESSION_FILE='ex_EXPRESSION.txt', NETWORK_FILE='ex_NETWORK.txt', RESULT_NAME='ex_RESULT', epoch=500, learningRate=0.005, lenPath=80, numBiomarker=50, numRepetition=10, sizeHiddenlayer=128)
	>>> 1. Load data
	>>> 2. Preprocess data
		n_samples: 135
		n_genes  : 7523     (common genes in both EXPRESSION and NETWORK)
		n_edges  : 216540   (edges with the common genes)
	>>> 3. Generate random paths from each group
		*** most time consuming step ***
		n_paths : 45402
		n_genes : 3773      (genes in good or poor random paths)
	>>> 4. Compute distributed representations using modified CBOW
	Start training the modified CBOW with early stopping
		- Epoch: 000        ACC[val]=0.6336 ACC[tr]=0.6310 (2.369 sec)
		- Epoch: 005        ACC[val]=0.8044 ACC[tr]=0.8232 (10.459 sec)
		- Epoch: 010        ACC[val]=0.8434 ACC[tr]=0.8633 (11.008 sec)
		- Epoch: 015        ACC[val]=0.8626 ACC[tr]=0.8860 (10.811 sec)
		- Epoch: 020        ACC[val]=0.8728 ACC[tr]=0.9006 (11.119 sec)
		- Epoch: 025        ACC[val]=0.8812 ACC[tr]=0.9106 (10.898 sec)
		- Epoch(stop): 027  ACC[val]=0.8837 ACC[tr]=0.9142 (6.811 sec)
		Optimization Finish
	>>> 5. Find L-groups
	>>> 6. Select biomarkers with gene scores
	>>> 7. Save results
		ex_RESULT_biomarkers.txt
		ex_RESULT_lgroups.txt
		ex_RESULT_vectors.txt
	$

	
--------------------------------------------------------------------------------------------
Note:

    The option parameter '-h' shows help message.
	
	$ python G2Vec.py -h
	
	
--------------------------------------------------------------------------------------------
