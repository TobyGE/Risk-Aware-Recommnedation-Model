import numpy as np
from heapq import heappush, heappop
from sklearn import metrics
import matplotlib.pyplot as plt
# auxiliary heap structure

DEBUG = False

def ranking_performance(score_dict,k):
	"""
	Input:
	- score_dict: dictionary of (user: (scores, #ground_truth)); 
			- where scores is: a heap array of (pred_score, (itemId, is_ground_truth))
	"""
	assert k > 0
	avgPrec = np.zeros(k)
	avgRecall = np.zeros(k)
	avgNDCG = np.zeros(k)
	avgAUC = 0
	# avgFPR = np.zeros(k)
	# avgTPR = np.zeros(k)
	for u, (itemScores, n_gt) in score_dict.items():
		if n_gt <= 0:
			continue
		userPrecision = np.zeros(k)
		userRecall = np.zeros(k)
		userDCG = np.zeros(k)
		userIDCG = np.zeros(k)
		hit = 0
		N = 0
		scores = []
		hits = []
		for score, (itemId, rel) in itemScores:
			scores.append(score)
			hits.append(rel)
			if rel == 1:
				hit = hit + 1
			N = N + 1
			if N <= k:
				# recall
				userRecall[N-1] = float(hit) / n_gt
				# precision
				userPrecision[N-1] = float(hit) / N
				# ndcg
				if N == 1:
					userDCG[0] = float(rel)
					userIDCG[0] = 1.0
				else:
					userDCG[N-1] = userDCG[N-2] + float(rel) / np.log2(N)
					if N <= n_gt:
						userIDCG[N-1] = userIDCG[N-2] + 1.0 / np.log2(N)
					else:
						userIDCG[N-1] = userIDCG[N-2]
				if DEBUG:
					print("{score: " + str(score) + "; rel: " + str(rel) + "}")
		#auc
		# userFPR, userTPR, _ = metrics.roc_curve(hits, scores, pos_label=2)
		#input()
		avgPrec += userPrecision
		avgRecall += userRecall
		avgNDCG += (userDCG / userIDCG)
		avgAUC += metrics.roc_auc_score(hits,scores)
		# avgFPR += userFPR
		# avgTPR += userTPR
	# print('avg precision', avgPrec[:10])
	# print('avg recall', avgRecall[:10])
	# print('avg NDCG', avgNDCG[:10])
	avgPrec /= len(score_dict)
	print('length of score_dict', len(score_dict))
	avgRecall /= len(score_dict)
	avgNDCG /= len(score_dict)
	avgAUC /= len(score_dict)
	# avgFPR /= len(score_dict) 
	# avgTPR /= len(score_dict)
	F1 = 2/(1/avgPrec+1/avgRecall)

	# plt.plot(avgFPR,avgTPR)
	# plt.xlabel('False positive rate')
	# plt.ylabel('True positive rate')
	# plt.title('ROC curve')
	# plt.show()

	print("\tPrecision@: {1:" + str(avgPrec[0]) + "; 5: " + str(avgPrec[4]) + "; 10: " + str(avgPrec[9]) + "; 20: " + str(avgPrec[19]) + "}")
	print("\tRecall@: {1:" + str(avgRecall[0]) + "; 5: " + str(avgRecall[4]) + "; 10: " + str(avgRecall[9]) + "; 20: " + str(avgRecall[19]) + "}")
	print("\tF1@: {1:" + str(F1[0]) + "; 5: " + str(F1[4]) + "; 10: " + str(F1[9]) + "; 20: " + str(F1[19]) + "}")
	print("\tNDCG@: {1:" + str(avgNDCG[0]) + "; 5: " + str(avgNDCG[4]) + "; 10: " + str(avgNDCG[9]) + "; 20: " + str(avgNDCG[19]) +"}")
# 	print("\tAUC:" + str(avgAUC))

	

	avgPrec = list(avgPrec)
	avgRecall = list(avgRecall)
	avgNDCG = list(avgNDCG)
	F1 = list(F1)
	return {"avg_precision": avgPrec, "avg_recall": avgRecall, "avg_ndcg": avgNDCG, "F1":F1, "AUC": avgAUC}
	# , "avg_fpr":avgFPR, "avg_tpr":avgTPR}




