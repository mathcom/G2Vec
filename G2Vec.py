import argparse
import numpy as np
import tensorflow as tf
import time
from copy import deepcopy
from math import sqrt
from operator import itemgetter
from sklearn.cluster import KMeans


def main():
	'''
	args={'EXPRESSION_FILE', 'CLINICAL_FILE', 'NETWORK_FILE', 'RESULT_FILE',
	      'lenPath', 'numRepetition', 'sizeHiddenlayer', 'epoch', 'learningRate', 'numBiomarker'}
	'''
	args = parse_arguments()
	print('>>> 0. Arguments')
	print(args)
	
	
	######## 1. Load data
	print('>>> 1. Load data')
	'''
	data={'expr':exprArr,
	      'gene':geneList,
	      'sample':sampleList}
	clinicial: a dictionary whose key and value is 'sampleName' and its label, respectively.
	network={'edge':edgeList,
	         'gene':geneset}
	'''
	data     = load_data(args.EXPRESSION_FILE)
	clinical = load_clinical(args.CLINICAL_FILE)
	network  = load_network(args.NETWORK_FILE)
	
	
	######## 2. Restrict data with the intersection of gene sets
	print('>>> 2. Preprocess data')
	data['label'] = match_labels(clinical,data['sample'])
	'''
	NOTE: A given network and data can have different gene sets.
	      Find the common set, and restrict network and data.
	'''
	commonGeneList = find_commonGeneList(network['gene'],data['gene'])
	network        = restrict_network(network,commonGeneList)
	data           = restrict_data(data,commonGeneList)
	'''
	data information
	'''
	n_samples, n_genes = data['expr'].shape
	n_edges            = len(network['edge'])
	print('    n_samples: %d' % n_samples)
	print('    n_genes  : %d\t(common genes in both EXPRESSION and NETWORK)' % n_genes)
	print('    n_edges  : %d\t(edges with the common genes)' % n_edges)

	
	######## 3. Generate random paths from each group
	print('>>> 3. Generate random paths from each group')
	pathSetList = []
	for i, group in enumerate(['g', 'p']):
		adjMat = construct_adjMat(network['edge'], data, i)
		pathSet = generate_pathSet(adjMat, args.lenPath, args.numRepetition)    # pathSet = {(0,2,6), (1,3), (2,7,4), ... }
		pathSetList.append(pathSet)
	
	pathList = integrate_pathSet(pathSetList, n_genes)    # Note: pathList[i] = [g0, g1, g2, ... , gN-1, label]  <--  1 or 0 // (n_genes+1)-D array
	geneFreq = count_geneFreq(pathList, data['gene'])
	print("    n_paths : %d" % pathList.shape[0])
	print("    n_genes : %d\t(genes in good or poor random paths)" % len(geneFreq.keys()))

	
	######## 4. Compute distributed representations using modified CBOW
	print(">>> 4. Compute distributed representations using modified CBOW")
	genetovec = dict()
	genetovec['mat'] = compute_genetovec(pathList, n_genes, args.sizeHiddenlayer, args.epoch, args.learningRate)
	genetovec['gene'] = deepcopy(data['gene'])

	
	######## 5. Find L-groups
	print('>>> 5. Find L-groups')
	lgroupIdx = find_lgroups(genetovec, geneFreq)
	

	######## 6. Select biomarkers with gene scores
	print(">>> 6. Select biomarkers with gene scores")
	gene2idx = make_gene2idx(data['gene'])
	for i, y in enumerate(['g','p']):
		genetovec['%s_mat'%y]  = genetovec['mat'][lgroupIdx==i]
		genetovec['%s_gene'%y] = genetovec['gene'][lgroupIdx==i]
		idx               = list(map(lambda gene:gene2idx[gene], genetovec['%s_gene'%y]))
		data['%s_expr'%y] = data['expr'][:,idx]
		data['%s_gene'%y] = deepcopy(genetovec['%s_gene'%y])
	
	biomarkerList = list()
	for y in ['g','p']:
		## d-score
		dscores = np.linalg.norm(genetovec['%s_mat'%y], axis=1)
		nor_dscores = transform_minmax(dscores, 0., 1.)
		## t-score
		tscores = compute_tscores(data['%s_expr'%y], data['label'])
		nor_tscores = transform_minmax(tscores, 0., 1.)
		## gene score
		gscores = 0.5 * (nor_dscores + nor_tscores)
		## Select prognostic modules
		geneScoreList = list(zip(genetovec['%s_gene'%y], gscores))
		geneScoreList = sorted(geneScoreList, key=itemgetter(1), reverse=True)
		tmpList       = sorted(list(map(lambda elem:elem[0], geneScoreList[:args.numBiomarker])))
		biomarkerList += tmpList
	## biomarkers
	biomarkerList = sorted(biomarkerList)
	
	
	######## 7. Save results
	print(">>> 7. Save results")
	fwrite_biomarker(args.RESULT_FILE, biomarkerList)    # Save 1) *_biomarkers
	print('    %s_biomarkers.txt' % args.RESULT_FILE)
	fwrite_lgroupIdx(args.RESULT_FILE, lgroupIdx, data['gene'])    # Save 2) *_lgroups
	print('    %s_lgroups.txt' % args.RESULT_FILE)
	fwrite_genetovec(args.RESULT_FILE, genetovec)    # Save 3) *_vectors
	print('    %s_vectors.txt' % args.RESULT_FILE)

	
	
	
	
	
	
def fwrite_biomarker(RESULTFILE, biomarkerList):
	with open(RESULTFILE + "_biomarkers.txt",'w') as fout:
		fout.write("GeneSymbol\n")
		for gene in biomarkerList:
			fout.write('%s\n' % gene)
	
def transform_minmax(scores, new_min, new_max):
	old_min = scores.min()
	old_max = scores.max()
	return (new_max - new_min) / (old_max - old_min) * (scores - old_min) + new_min
	
def compute_tstatistics(X,Y):
	sampleMeans = [X.mean(), Y.mean()]
	sampleStds  = [X.std(ddof=1), Y.std(ddof=1)]
	n_poor = len(X)
	n_good = len(Y)
	firstDenominator  = sqrt(((float(n_poor)-1.)*sampleStds[0]*sampleStds[0] + (float(n_good)-1.)*sampleStds[1]*sampleStds[1]) / float(n_poor+n_good-2))
	secondDenominator = sqrt((1./float(n_poor)) + (1./float(n_good)))
	if firstDenominator > 0. and secondDenominator > 0.:
		TScore = (sampleMeans[0]-sampleMeans[1]) / firstDenominator / secondDenominator
	else:
		TScore = 0.
	return TScore
	
def compute_tscores(expr, label):
	result = np.zeros(expr.shape[1], dtype=np.float32)
	for i in range(expr.shape[1]):
		g_expr    = expr[label==0,i]
		p_expr    = expr[label==1,i]
		result[i] = abs(compute_tstatistics(g_expr,p_expr))
	return result
	
def fwrite_lgroupIdx(RESULTFILE, lgroupIdx, geneList):
	with open(RESULTFILE + "_lgroups.txt", 'w') as fout:
		## header
		fout.write('GeneSymbol\tLgroup(0:good,1:poor,2:other)\n')
		## body
		for gene, group in zip(geneList, lgroupIdx):
			fout.write('%s\t%d\n' % (gene,group))
	
def find_lgroups(genetovec, geneFreq):
	#### 1) K-Means
	km    = KMeans(n_clusters=3, random_state=0).fit(genetovec['mat'])
	kmIdx = km.labels_
	#### 2) gene frequencies
	freqIdx = list(map(lambda gene:geneFreq.get(gene,2), genetovec['gene']))
	#### 3) Identify the init cluster
	largestClusterIdx  = 0
	sizeLargestCluster = np.count_nonzero(kmIdx == 0)
	for i in [1,2]:
		sizeCluster = np.count_nonzero(kmIdx == i)
		if sizeCluster > sizeLargestCluster:
			largestClusterIdx  = i
			sizeLargestCluster = sizeCluster
	#### 4) Identify good / poor L-groups
	lgIdx= [0,1,2]
	lgIdx.remove(largestClusterIdx)
	gpDiff = np.zeros(3, dtype=np.float32)
	for i in lgIdx:
		n_moregood = np.count_nonzero(np.logical_and(kmIdx==i,freqIdx==0))
		n_morepoor = np.count_nonzero(np.logical_and(kmIdx==i,freqIdx==1))
		gpDiff[i] = n_moregood - n_morepoor
	if gpDiff[lgIdx[0]] > gpDiff[lgIdx[1]]:
		goodClusterIdx = lgIdx[0]
		poorClusterIdx = lgIdx[1]
	else:
		goodClusterIdx = lgIdx[1]
		poorClusterIdx = lgIdx[0]
	#### 5) Renumbering
	result = np.zeros(genetovec['mat'].shape[0], dtype=np.int32)
	result[kmIdx==goodClusterIdx] = 0  # 0: good
	result[kmIdx==poorClusterIdx] = 1  # 1: poor
	result[kmIdx==largestClusterIdx] = 2  # 2: init
	return result
	
	
def fwrite_genetovec(RESULTFILE, genetovec):
	with open(RESULTFILE + "_vectors.txt", 'w') as fout:
		## header
		fout.write('GeneSymbol')
		for i in range(genetovec['mat'].shape[1]):
			fout.write('\tV%d' % i)
		fout.write('\n')
		## body
		for gene, vector in zip(genetovec['gene'], genetovec['mat']):
			fout.write(gene)
			for val in vector:
				fout.write("\t%.6f" % val)
			fout.write("\n")

def compute_genetovec(pathList, n_genes, hidden_size, training_epochs, learning_rate):
	## Hold-out: training(80%) and validating(20%) data
	np.random.shuffle(pathList)
	pivot = int(len(pathList) * 0.8)
	tr_pathList = pathList[:pivot]
	vl_pathList = pathList[pivot:]
	x_training  = tr_pathList[:,:n_genes]
	y_training  = tr_pathList[:,-1].reshape([tr_pathList.shape[0],1])
	x_validation = vl_pathList[:,:n_genes]
	y_validation = vl_pathList[:,-1].reshape([vl_pathList.shape[0],1])
	'''
	Tensorflow
	'''
	## 1) Input & Actual data
	X = tf.placeholder(tf.float32, [None, n_genes], name="InputData")
	Y = tf.placeholder(tf.float32, [None, 1], name="ActualData")
	## 2) weight variables
	W_ih = tf.Variable(tf.truncated_normal([n_genes, hidden_size], stddev= 1./sqrt(hidden_size)), name="Weight_IH")
	W_ho = tf.Variable(tf.truncated_normal([hidden_size, 1], stddev= 1./sqrt(hidden_size)), name="Weight_HO")
	## 3) Construct Model and encapsulate all operations into scopes
	##    for the TensorBoard's graph visualization
	with tf.name_scope('CBOW'):
		H = tf.matmul(X, W_ih)
		O = tf.matmul(H, W_ho)
	
	with tf.name_scope('Cost'):
		cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=O, labels=Y))
	
	with tf.name_scope('Optimization'):
		optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

	with tf.name_scope('Accuracy'):
		pred_Y     = tf.cast(tf.sigmoid(O) > 0.5, tf.float32)
		correction = tf.equal(pred_Y, Y)  # bool
		acc        = tf.reduce_mean(tf.cast(correction, tf.float32))
	
	## 4) Training CBOW
	display_step = 5
	with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as sess:
		## 4-1) Initialize all variables
		tf.global_variables_initializer().run()
		## 4-2) Start learning the model
		print("     Start learning with %d epochs" % (training_epochs))
		start_time = time.time()
		before_acc_val = -1.
		for step in range(training_epochs):
			## Optimization
			_ = sess.run(optimizer, feed_dict={X:x_training, Y:y_training})
			## Compute Accuracy
			acc_val = acc.eval({X:x_validation, Y:y_validation})
			acc_tr  = acc.eval({X:x_training, Y:y_training})
			## Display logs per epoch step
			if step % display_step == 0:
				end_time = time.time()
				print("    - Epoch: %03d\tACC[val]=%.4f\tACC[tr]=%.4f (%.3f sec)" % (step, acc_val, acc_tr, end_time-start_time))
				start_time = time.time()
			## terminate condition
			current_acc_val = acc_val
			current_acc_tr  = acc_tr
			if current_acc_val < before_acc_val:
				end_time = time.time()
				print("    - Epoch(stop): %03d\tACC[val]=%.4f\tACC[tr]=%.4f (%.3f sec)" % (step-1, before_acc_val, before_acc_tr, end_time-start_time))
				break
			before_acc_val = current_acc_val
			before_acc_tr  = current_acc_tr
			## result
			result = sess.run(W_ih)
		print("    Optimization Finish")
		
	return result

def count_geneFreq(pathLabelList, geneList):
	pathList = pathLabelList[:,:-1]
	labelList = pathLabelList[:,-1]
	geneFreq = [dict(), dict()]  # 0:good, 1:poor
	targets = set()
	for path, label in zip(pathList, labelList):
		genes = set(geneList[path==1])
		for gene in genes:
			geneFreq[label][gene] = geneFreq[label].get(gene,0) + 1
			targets.add(gene)
	result = dict()
	for gene in targets:
		fg = geneFreq[0].get(gene,0)
		fp = geneFreq[1].get(gene,0)
		if fg > fp:
			result[gene] = 0
		elif fg < fp:
			result[gene] = 1
		else:
			result[gene] = 2
	return result
	
def integrate_pathSet(pathSetList, n_genes):
	pathList = list()
	## find common path
	commonPath = pathSetList[0].intersection(pathSetList[1])
	for label, pathSet in enumerate(pathSetList):
		for path in pathSet-commonPath:
			## Note: pathData = [g0, g1, g2, ... , gN, label]  <--  (n_genes+1)-D array
			pathData = np.zeros(n_genes+1, dtype=np.int32)
			pathData[list(path)] = 1
			pathData[-1]         = label
			pathList.append(pathData)
	pathList = np.array(pathList, dtype=np.int32)
	return pathList
		
def generate_pathSet(adjMat, maximumLength, iterations):
	pathSet = set()
	n_genes = adjMat.shape[0]
	
	def generate_randomPath(src, adjMat, n_genes, maximumLength):
		path        = list()
		currentNode = src
		for step in range(maximumLength):
			path.append(currentNode)
			## adjMat = src * dest
			prob = deepcopy(adjMat[currentNode])
			## A walker dosen't go back to where he once went.
			prob[path] = 0.
			## A walker selects next node randomly.
			normalization = prob.sum()
			if normalization > 0.:
				prob /= normalization
				currentNode = np.random.choice(n_genes, size=1, p=prob)[0]
			else:
				## Random walking stops if a walker reaches a dead end.
				break
		path = tuple(sorted(path))
		return path
	
	for step in range(iterations):
		for src in range(n_genes):
			path = generate_randomPath(src, adjMat, n_genes, maximumLength)
			pathSet.add(path)
	return pathSet

def compute_PCC(X,Y):
	## condition1: len(X) == len(Y)
	std_X = X.std()
	std_Y = Y.std()
	
	if std_X > 0. and std_Y > 0.:
		avg_X = X.mean()
		avg_Y = Y.mean()
		zscored_X = (X - avg_X) / std_X
		zscored_Y = (Y - avg_Y) / std_Y
		Z = zscored_X * zscored_Y
		pcc = Z.mean()
	else:
		pcc = 0.
	return pcc	
	
def construct_adjMat(edgeList, data, label):
	## make gene2idx
	gene2idx = dict()
	for i, gene in enumerate(data["gene"]):
		gene2idx[gene] = i
	## make adjacency matrix
	n_genes = len(data["gene"])
	adjMat  = np.zeros([n_genes, n_genes], dtype=np.float32)
	expr    = data["expr"][data["label"]==label]
	for edge in edgeList:
		src       = gene2idx[edge[0]]
		dest      = gene2idx[edge[1]]
		src_data  = expr[:,src]
		dest_data = expr[:,dest]
		## compute weight
		weight = abs(compute_PCC(src_data, dest_data))
		'''
		Note: We don't consider the interactions with low weight
		'''
		if weight > 0.5:
			adjMat[src][dest] = weight
	return adjMat

def restrict_network(network,commonGeneList):
	edgeList = list()
	commonGeneSet = set(commonGeneList)
	for edge in network['edge']:
		if edge[0] in commonGeneSet and edge[1] in commonGeneSet:
			edgeList.append(edge)
	
	result = {'edge':edgeList,
	          'gene':commonGeneSet}
	return result
	
def restrict_data(data,commonGeneList):	
	gene2idx = make_gene2idx(data['gene'])
	idx = list(map(lambda gene:gene2idx[gene], commonGeneList))
	
	result = {'sample':deepcopy(data['sample']),
	          'label':deepcopy(data['label']),
	          'expr':data['expr'][:,idx],
	          'gene':np.array(commonGeneList)}
	return result
	
def make_gene2idx(geneList):
	result=dict()
	for i, gene in enumerate(geneList):
		result[gene]=i
	return result
	
def find_commonGeneList(X,Y):
	X=set(X)
	Y=set(Y)
	result = X.intersection(Y)
	result = list(result)
	result = sorted(result)  # sorted by gene symbol A->Z
	return result
	
def match_labels(clinical,sampleList):
	try:
		result = list(map(lambda sample:clinical[sample], sampleList))
	except:
		print('ERROR: There is a mismatched sample between expression data and clinical data. Please check sample names')
		exit(1)
	return np.array(result)

def load_clinical(dataFile):
	'''
	PATIENT         LABEL
	TCGA-AR-A24H    0
	TCGA-AR-A24L    0
	TCGA-AR-A24M    0
	'''
	with open(dataFile) as fin:
		lines = fin.readlines()
	## preprocessing
	lines = list(map(lambda line:line.rstrip().split('\t'), lines))
	## make result
	result = dict()
	for line in lines[1:]:
		sample = line[0]
		label = int(line[1])
		result[sample]=label
	return result
	
def load_network(dataFile):
	'''
	GENE1   GENE2
	RPL37A  RPS27A
	MRPL1   MRPS36
	RFC3    SPRTN
	'''
	with open(dataFile) as fin:
		lines = fin.readlines()
	## preprocessing
	lines = list(map(lambda line:line.rstrip().split('\t'), lines))
	## list of edges
	edgeList = lines[1:]
	## gene set
	geneSet=set()
	for edge in edgeList:
		geneSet.add(edge[0])
		geneSet.add(edge[1])
	## make result
	result = {'edge':edgeList,
	          'gene':geneSet}
	return result
	
def load_data(dataFile):
	'''
	PATIENT   TCGA-AR-A24H  TCGA-AR-A24L  TCGA-AR-A24M  TCGA-AR-A24N
	A1CF      -0.436158     -0.276784     -0.309453     -0.305223
	A2M       1.90128       2.72735       4.03939       1.33212
	A4GALT    -0.408337     -0.247608     -0.260444     -0.234695
	'''
	with open(dataFile) as fin:
		lines = fin.readlines()
	## preprocessing
	lines = list(map(lambda line:line.rstrip().split('\t'), lines))
	## header
	sample = np.array(lines[0][1:])
	## body
	gene = list()
	expr = list()
	for line in lines[1:]:
		gene.append(line[0])
		expr.append(line[1:])
	gene = np.array(gene)
	expr = np.array(expr, dtype=np.float32).T  # genewise-->samplewise
	## maek result
	result = {'sample':sample,
	          'expr':expr,
	          'gene':gene}
	return result
	
def parse_arguments():
	parser=argparse.ArgumentParser(description="""G2Vec is a network-based deep learning method for identifying prognostic gene signatures(biomarkers).
	                                              Please refer to included 'manual.pdf'. For more detail, please refer to 'G2Vec: Distributed gene representations for identification of cancer prognostic genes'. (yet published)""")
	parser.add_argument('EXPRESSION_FILE', type=str, help="Tab-delimited file for gene expression profiles.")
	parser.add_argument('CLINICAL_FILE', type=str, help="Tab-delimited file for patient's clinical data. LABEL=0:good prognosis and 1:poor prognosis.")
	parser.add_argument('NETWORK_FILE', type=str, help="Tab-delimited file for gene interaction network.")
	parser.add_argument('RESULT_FILE', type=str, help="The results of G2Vec are saved with the following four names: 1) *_biomarkers.txt, 2) *_lgroups.txt, and 3) *_vectors.txt")
	parser.add_argument('-p', '--lenPath', type=int, default=80, help='')
	parser.add_argument('-r', '--numRepetition', type=int, default=10, help='')
	parser.add_argument('-s', '--sizeHiddenlayer', type=int, default=128, help='')
	parser.add_argument('-e', '--epoch', type=int, default=500, help='')
	parser.add_argument('-l', '--learningRate', type=float, default=0.005, help='')
	parser.add_argument('-n', '--numBiomarker', type=int, default=50, help='')
	return parser.parse_args()

	
if __name__=="__main__":
	main()