import argparse
import numpy as np
from copy import deepcopy
from math import sqrt, ceil
from operator import itemgetter
from scipy.stats import rankdata
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import auc, roc_curve, silhouette_score, accuracy_score
from sklearn.model_selection import StratifiedKFold

def main():
	'''
	args={'EXPRESSION_FILE', 'CLINICAL_FILE', 'NETWORK_FILE', 'RESULT_FILE',
	      'dampingFactor', 'numBiomarkers', 'numClusters', 'conditionHubgene', 'v'}
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
	labels = map_labels(clinical,data['sample'])
	
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
	
	
	######## 3. Gene selection with 'Clustering and Modified PageRank (CPR)'
	print('>>> 3. Conduct CPR')
	cpr = CPR(dampingFactor=args.dampingFactor,
	          n_biomarkers=args.numBiomarkers,
	          n_clusters=args.numClusters,
	          c_hubgene=args.conditionHubgene,
	          logshow=True)
	cpr.fit(expr=data['expr'],
	        labels=labels,
	        genes=data['gene'],
	        edges=network['edge'],
	        random_state=0)
	biomarkers = cpr.get_biomarkers()
	subnetwork = cpr.get_subnetwork()
	PRscores   = cpr.get_PRscores()
	
	######### 4. Summary for results
	print('>>> 4. Save results')
	print('    %s' % args.RESULT_FILE+'_biomarker.txt')
	with open(args.RESULT_FILE+'_biomarker.txt', 'w') as fout:
		## header
		fout.write('GeneSymbol\tPRscore\n')
		## body
		for elem in biomarkers:
			fout.write('%s\t%.6f\n' % elem)
			
	print('    %s' % args.RESULT_FILE+'_score.txt')		
	with open(args.RESULT_FILE+'_score.txt', 'w') as fout:
		## header
		fout.write('GeneSymbol')
		for i in range(len(PRscores[0])-1):
			fout.write('\tPRScore_%d' % i)
		fout.write('\n')
		## body
		for elem in PRscores:
			fout.write('%s' % elem[0])
			for i in range(1,len(elem)):
				fout.write('\t%.6f' % elem[i])
			fout.write('\n')
			
	print('    %s' % args.RESULT_FILE+'_subnetwork.txt')	
	with open(args.RESULT_FILE+'_subnetwork.txt', 'w') as fout:
		## header
		fout.write('source\ttarget\n')
		## body
		for edge in subnetwork:
			fout.write('%s\t%s\n' % edge)
			
	######## 5. 10 fold Cross Validation
	if args.crossvalidation:
		print('>>> 5. 10-fold Cross validation')	
		mean_auc, mean_acc = compute_accuracy_via_crossvaldiation(data=data, labels=labels, network=network,
		                                                          K=10, random_state=0,
		                                                          dampingFactor=args.dampingFactor,
		                                                          n_biomarkers=args.numBiomarkers,
		                                                          n_clusters=args.numClusters,
		                                                          c_hubgene=args.conditionHubgene)
		print('    AUC-ROC : %.3f' % mean_auc)
		print('    Accuracy: %.3f' % mean_acc)
		
		print('    %s' % args.RESULT_FILE+'_accuracy.txt')	
		with open(args.RESULT_FILE+'_accuracy.txt', 'w') as fout:
			fout.write('AUC-ROC\t%.3f\n' % mean_auc)
			fout.write('Accuracy\t%.3f\n' % mean_acc)	

class CPR:
	def __init__(self,
		         dampingFactor=0.7,
		         n_biomarkers=70,
		         n_clusters=0,
		         c_hubgene=0.02,
		         logshow=False):
		self.dampingFactor = dampingFactor
		self.n_biomarkers  = n_biomarkers
		self.n_clusters    = n_clusters
		self.c_hubgene     = c_hubgene
		self.logshow       = logshow
		
	def get_biomarkers(self):
		return self.biomarkers
		
	def get_PRscores(self):
		return self.PRscores
		
	def get_subnetwork(self):
		return self.subnetwork
		
	def fit(self, expr, labels, genes, edges, random_state=None):
		#### Set parameters
		n_samples, n_genes = expr.shape
		
		#### K-means clustering
		if self.logshow:
			print('    K-means clustering')
		if self.n_clusters==1:
			idx = np.zeros(n_samples, dtype=np.int32)
		else:
			idx = self._conduct_sampleClustering(expr, random_state=random_state)
		'''
		cluster information
		'''
		if self.logshow:
			print('    -> n_clusters: %d' % self.n_clusters)
			for i in range(self.n_clusters):
				n_samples = np.count_nonzero(idx==i)
				n_goods   = np.count_nonzero(labels[idx==i]==0)
				n_poors   = np.count_nonzero(labels[idx==i]==1)
				print('        In cluster[%d], n_samples:%d, n_goods:%d, n_poors:%d' % (i, n_samples, n_goods, n_poors))
			
		#### Adjacency Matrix
		adjMat = self._make_adjacencyMatrix(edges,genes)
		
		#### Modified PageRank
		if self.logshow:
			print('    Modified PageRank')
		PRScores = np.zeros([n_genes,self.n_clusters], dtype=np.float32)
		for i in range(self.n_clusters):
			tmp_expr   = expr[idx==i,:]
			tmp_labels = labels[idx==i]
			PRScores[:,i] += self._conduct_modifiedPageRank(tmp_expr, tmp_labels, adjMat)
		geneScores    = PRScores.mean(axis=1)
		
		self.PRscores = list()
		for gene, scores in zip(genes, PRScores.tolist()):
			self.PRscores.append(tuple([gene]+scores))
		
		#### Degree in network
		geneDegrees  = self._compute_geneDegree(adjMat)
		if self.c_hubgene == 1.:
			hubCriterion = 0
		elif self.c_hubgene > 0. and self.c_hubgene < 1.:
			hubCriterion = self._find_criterionHubgene(geneDegrees)
		else:
			print('ERROR: The condition of hub-gene must be between 0 and 1.')
			exit(1)
		
		#### Sorting
		geneList        = list(zip(genes,geneScores,geneDegrees))
		geneList_sorted = sorted(geneList, key=itemgetter(1), reverse=True)
		
		#### Identify biomarkers
		biomarkerList = list()
		for gene, score, degree in geneList_sorted:
			if degree > hubCriterion:
				biomarkerList.append((gene,score))
				if len(biomarkerList) == self.n_biomarkers:
					break
		self.biomarkers = biomarkerList
		
		#### Subnetwork
		subnetwork = list()
		biomarkerSet = set(list(map(lambda elem:elem[0], biomarkerList)))
		for edge in edges:
			if edge[0] in biomarkerSet or edge[1] in biomarkerSet:
				subnetwork.append((edge[0],edge[1]))
		self.subnetwork = subnetwork
		
	def _find_criterionHubgene(self, geneDegrees):
		n_genes  = len(geneDegrees)
		c_hubgene = self.c_hubgene
		n_hubgene = ceil(n_genes * c_hubgene)
		geneDegrees_sorted = sorted(geneDegrees, reverse=True)
		return geneDegrees_sorted[n_hubgene]
		
	def _compute_geneDegree(self, adjMat):
		n_genes = adjMat.shape[0]
		result = np.zeros(n_genes, dtype=np.int32)
		for i in range(n_genes):
			result[i] = adjMat[i].sum() + adjMat[:,i].sum()
		return result
		
	def _conduct_modifiedPageRank(self, expr, labels, adjMat):
		## Set parameters
		n_samples, n_genes = expr.shape
		n_poors = np.count_nonzero(labels==1)
		n_goods = n_samples - n_poors
		threshold = 1E-5
		maxIteration = 100
		
		## Important conditions
		if n_poors < 2 or n_goods < 2:
			return np.zeros(n_genes, dtype=np.float32)
		
		## t-score
		expr_good = expr[labels==0]
		expr_poor = expr[labels==1]
		tscores   = np.zeros(n_genes, dtype=np.float32)
		for i in range(n_genes):
			tscores[i] = self._compute_tscore(expr_good[:,i], expr_poor[:,i])
		if tscores.sum() == 0:
			return np.zeros(n_genes, dtype=np.float32)
		
		## weighted adjacency matrix
		adjMat_weighted = self._make_weighted_adjacencyMatrix(adjMat, tscores)
		
		## Normalize adjacency matrix to make a transition matrix
		adjMat_normalized          = self._normalize_adjMat(adjMat)
		adjMat_weighted_normalized = self._normalize_adjMat(adjMat_weighted)
		
		## PageRank
		scores          = self._pageRank(adjMat_normalized, threshold, maxIteration)
		scores_weighted = self._pageRank(adjMat_weighted_normalized, threshold, maxIteration)
		return scores_weighted / scores
		
	def _pageRank(self, mat, threshold, maxIteration):
		n_genes = mat.shape[0]
		d = self.dampingFactor
		
		init_score    = np.zeros(n_genes, dtype=np.float32) + 1./float(n_genes)
		current_score = deepcopy(init_score)
		before_score  = deepcopy(current_score)
		
		for i in range(maxIteration):
			current_score = (1.-d)*init_score + d*(mat.dot(current_score))
			## termination
			if np.abs(current_score-before_score).sum() < threshold:
				break
			else:
				before_score = current_score
		return current_score
		
	def _normalize_adjMat(self, adjMat):
		## preprocess
		n_genes = adjMat.shape[0]
		correction = np.zeros(n_genes, dtype=np.float32) + 1./n_genes
		## normalize
		result = np.zeros(adjMat.shape, dtype=np.float32)
		for i in range(n_genes):
			normalizer = adjMat[:,i].sum()
			if normalizer > 0.:
				result[:,i] = adjMat[:,i] / normalizer
			else:
				result[:,i] = correction
		return result
		
	def _make_weighted_adjacencyMatrix(self, adjMat, tscores):
		return adjMat * tscores.reshape([len(tscores),1])
		
	def _compute_tscore(self, X, Y):
		n_X = float(len(X))
		n_Y = float(len(Y))
		mean_X = X.mean()
		mean_Y = Y.mean()
		var_X = X.var(ddof=1)
		var_Y = Y.var(ddof=1)
		
		if var_X > 0. and var_Y > 0.:
			## t-statistics
			p = (n_X-1.) / (n_X+n_Y-2.)
			denominator = p*var_X + (1.-p)*var_Y
			denominator *= ((1./n_X) + (1./n_Y))
			tscore = (mean_X - mean_Y) / sqrt(denominator)
			tscore = abs(tscore)
		else:
			tscore = 0.
		return tscore
		
	def _make_adjacencyMatrix(self, edges, genes):
		n_genes=len(genes)
		gene2idx=make_gene2idx(genes)
		## adjanceny matrix
		adjMat=np.zeros([n_genes,n_genes], dtype=np.float32)
		for edge in edges:
			source=gene2idx[edge[0]]
			target=gene2idx[edge[1]]
			adjMat[source][target]=1.
		return adjMat
		
	def _conduct_sampleClustering(self, expr, random_state):
		## 1) zscoring genewise
		expr_zscored = np.zeros(expr.shape)
		for i in range(expr.shape[1]):
			m = expr[:,i].mean()
			s = expr[:,i].std()
			if s > 0:
				expr_zscored[:,i] = (expr[:,i] - m) / s
	
		## 2) PCA
		pca = PCA(n_components=2, random_state=random_state)
		expr_projected = pca.fit_transform(expr_zscored)
		
		## 3) K-Means
		if self.n_clusters==0:
			silhouetteScores = list()
			for i in [2,3,4]:
				kmeans = KMeans(n_clusters=i, random_state=random_state).fit(expr_projected)
				score  = silhouette_score(expr_projected, kmeans.labels_, random_state=random_state)
				silhouetteScores.append((i,score))
			silhouetteScores = sorted(silhouetteScores, key=itemgetter(1), reverse=True)
			self.n_clusters  = silhouetteScores[0][0]
			
		kmeans = KMeans(n_clusters=self.n_clusters, random_state=random_state).fit(expr_projected)
		return kmeans.labels_
	
	
	
def compute_accuracy_via_crossvaldiation(data, labels, network, K, random_state, dampingFactor, n_biomarkers, n_clusters, c_hubgene):
		## set parameters
		n_samples, n_genes = data['expr'].shape
		
		## Cross validation
		cv = StratifiedKFold(n_splits=K, random_state=random_state, shuffle=False)
		clf = RandomForestClassifier(n_estimators=100, random_state=random_state)
		
		k = 1
		mean_tpr = np.zeros(100, dtype=np.float32)
		mean_fpr = np.linspace(start=0, stop=1, num=100)
		mean_acc = 0.
		for train, test in cv.split(data['expr'], labels):
			## 1) gene selection
			cpr = CPR(dampingFactor=dampingFactor,
			          n_biomarkers=n_biomarkers,
			          n_clusters=n_clusters,
			          c_hubgene=c_hubgene)
			cpr.fit(expr=data['expr'][train],
			        labels=labels[train],
			        genes=data['gene'],
			        edges=network['edge'],
			        random_state=1)
			biomarkers = cpr.get_biomarkers()
			## 2) Restrict data
			biomarkerList   = list(map(lambda elem:elem[0], biomarkers))
			data_restricted = restrict_data(data,biomarkerList)
			## 3) rankdata
			expr_ranked = np.zeros(data_restricted['expr'].shape, dtype=np.float32)
			for i, arr in enumerate(data_restricted['expr']):
				expr_ranked[i] = rankdata(arr)
			## 4) fit
			clf.fit(expr_ranked[train], labels[train])
			## 5) compute accuracy
			pred     = clf.predict(expr_ranked[test])
			mean_acc += accuracy_score(labels[test], pred, normalize=False)
			## 6) compute AUC-ROC
			probas_              = clf.predict_proba(expr_ranked[test])
			fpr, tpr, thresholds = roc_curve(labels[test], probas_[:,1])
			mean_tpr             += np.interp(mean_fpr, fpr, tpr)
			## log
			print('    %d%% complete!' % int(k/K*100))
			k += 1
		## 7) Average of AUC-ROC
		mean_tpr /= K
		mean_tpr[0] = 0.
		mean_tpr[-1] = 1.
		mean_auc = auc(mean_fpr, mean_tpr)
		mean_acc /= n_samples
		return mean_auc, mean_acc
		
		
def parse_arguments():
	parser=argparse.ArgumentParser(description="""CPR is a program to identify prognostic genes (biomarkers) and use them to predict prognosis of cancer patients.
	                                              Please refer to included 'manual.pdf'. For more detail, please refer to 'Improved prediction for breast cancer outcome by identifying heterogeneous biomarkers'.""")
	parser.add_argument('EXPRESSION_FILE', type=str, help="Tab-delimited file for gene expression profiles.")
	parser.add_argument('CLINICAL_FILE', type=str, help="Tab-delimited file for patient's clinical data. LABEL=0:good prognosis and 1:poor prognosis.")
	parser.add_argument('NETWORK_FILE', type=str, help="Tab-delimited file for gene interaction network.")
	parser.add_argument('RESULT_FILE', type=str, help="The results of CPR are saved with the following three names: 1) *_biomarker.txt, 2) *_subnetwork.txt, 3) *_score.txt")
	parser.add_argument('-m', '--numClusters', type=int, default=0, help="A parameter of K-means clustering algorithm. This parameter decides number of sample clusters to handle the heterogeneity of patients. If the default value is given, the number of clusters is determined by the silhouette score. If a specific integer is given, the K-means clustering is conducted with the number. (default=0)")
	parser.add_argument('-d', '--dampingFactor', type=float, default=0.7, help="A parameter of PageRank algorithm.This parameter decides an influence of network information on prediction. The value must be between 0.0 and 1.0. (default=0.7)")
	parser.add_argument('-n', '--numBiomarkers', type=int, default=70, help="This parameter decides number of biomarkers to use in prediction. (default=70)")
	parser.add_argument('-c', '--conditionHubgene', type=float, default=0.02, help="This parameter is used to identify a hub-gene. When c is a given parameter and x is the total of genes, we define top cx genes with high degree as hub-genes. (default=0.02)")
	parser.add_argument('-v', '--crossvalidation', action='store_true', help="When this option is given, CPR.py will conduct 10-fold cross validation with the given data. The result of cross validation is provided in 4) *_accuracy.txt")
	return parser.parse_args()

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
	
def find_commonGeneList(X,Y):
	X=set(X)
	Y=set(Y)
	result = X.intersection(Y)
	result = list(result)
	result = sorted(result)  # sorted by gene symbol A->Z
	return result
	
def make_gene2idx(geneList):
	result=dict()
	for i, gene in enumerate(geneList):
		result[gene]=i
	return result
	
def restrict_data(data,commonGeneList):	
	## find indices corresponding to common genes
	geneToIdx = make_gene2idx(data['gene'])
	idx = list(map(lambda gene:geneToIdx[gene], commonGeneList))
	## make result
	result = {'sample':deepcopy(data['sample']),
	          'expr':data['expr'][:,idx],
	          'gene':np.array(commonGeneList)}
	return result

def restrict_network(network,commonGeneList):
	## find edges with target genes
	edgeList = list()
	commonGeneSet = set(commonGeneList)
	for edge in network['edge']:
		if edge[0] in commonGeneSet and edge[1] in commonGeneSet:
			edgeList.append(edge)
	## make result
	result = {'edge':edgeList,
	          'gene':commonGeneSet}
	return result
	
def map_labels(clinical,sampleList):
	try:
		result = list(map(lambda sample:clinical[sample], sampleList))
	except:
		print('ERROR: There is a mismatched sample between expression data and clinical data. Please check sample names')
		exit(1)
	return np.array(result)


if __name__=="__main__":
	main()