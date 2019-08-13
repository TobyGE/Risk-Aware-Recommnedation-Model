import numpy as np
from heapq import heappush, heappop

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
        for score, (itemId, rel) in itemScores:
            scores.append(score)
            if rel == 1:
                hit = hit + 1
            N = N + 1
            if N > k:
                break
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
        #input()
        avgPrec += userPrecision
        avgRecall += userRecall
        avgNDCG += (userDCG / userIDCG)
    avgPrec /= len(score_dict)
    avgRecall /= len(score_dict)
    avgNDCG /= len(score_dict)
    print("\tPrecision@: {1:" + str(avgPrec[0]) + "; 5: " + str(avgPrec[4]) + "; 10: " + str(avgPrec[9]) + "}")
    print("\tRecall@: {1:" + str(avgRecall[0]) + "; 5: " + str(avgRecall[4]) + "; 10: " + str(avgRecall[9]) + "}")
    print("\tNDCG@: {1:" + str(avgNDCG[0]) + "; 5: " + str(avgNDCG[4]) + "; 10: " + str(avgNDCG[9]) + "}")
    return {"avg_precision": avgPrec, "avg_recall": avgRecall, "avg_ndcg": avgNDCG}




