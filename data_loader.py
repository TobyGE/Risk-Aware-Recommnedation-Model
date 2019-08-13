import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import math
from scipy.stats import norm

def read_data(category):
	address = "./data/" + category + "/" + category + "_"
	with open(address + "TrainSamples.txt","r") as f:
		data = f.readlines()
	TrainSamples = []
	for line in data:
		row = line[:-1].split(",")
		sample = [int(float(i)) for i in row]
		TrainSamples.append(sample)

	with open(address + "ValidationSamples.txt","r") as f:
		data = f.readlines()
	ValSamples = []
	for line in data:
		row = line[:-1].split(",")
		sample = [int(float(i)) for i in row]
		ValSamples.append(sample)

	with open(address + "TestSamples.txt","r") as f:
		data = f.readlines()
	TestSamples = []
	for line in data:
		row = line[:-1].split(",")
		sample = [int(float(i)) for i in row]
		TestSamples.append(sample)

	return TrainSamples, ValSamples, TestSamples

def approx_Gaussian(frequency):
    distribution = []
    for i in range(len(frequency)):
        mu = 0
        for j in range(5):
            mu += (j+1) * frequency[i][j]
        sigma = 0
        for j in range(5):
            sigma += math.pow(j+1-mu,2) * frequency[i][j]
        if sigma == 0:
            sigma = 0.1
        prob_ij = []
        cdf_ij = []
        for r in range(1,5):
            cdf_ij.append(norm.cdf(r+0.5,mu,sigma))
        prob_ij.append(cdf_ij[0])
        prob_ij.append(cdf_ij[1]-cdf_ij[0])
        prob_ij.append(cdf_ij[2]-cdf_ij[1])
        prob_ij.append(cdf_ij[3]-cdf_ij[2])
        prob_ij.append(1 - cdf_ij[3])
        distribution.append(prob_ij)
    return distribution

def get_datasize(category):
	address = "./data/" + category + "/" + category + "_" + "AllSamples.txt"
	AllSamples = list()
	with open(address,'r') as f:
		data = f.readlines()
	for line in data:
		row = line.split(',')
		samples = [int(float(i)) for i in row]
		AllSamples.append(samples)
	all_data = np.array(AllSamples)
	userNum = len(np.unique(all_data[:,0]))
	itemNum = len(np.unique(all_data[:,1]))
	return userNum, itemNum

def get_price(category):
    address = "./data/" + category + "/" + category + "_" + "item_price.npy"
    price = np.load(address)
    return price

def get_distribution(category):
    address = "./data/" + category + "/" + category + "_" + "ItemResult.npy"
    distribution = np.load(address)
    return distribution

class TransactionData(Dataset):
	"""docstring for TransactionData"""
	def __init__(self, transctions, userNum, itemNum, rating_distribution):
		super(TransactionData, self).__init__()
		self.transctions = transctions
		self.L = len(transctions)
		self.users = np.unique(np.array(transctions)[:,0])
		self.userNum = userNum
		self.itemNum = itemNum
		self.negNum = 2
		self.rating_distribution = rating_distribution
		self.userHist = [[] for i in range(self.userNum)]
		for row in transctions:
			self.userHist[row[0]].append(row[1])


	def __len__(self):
		return self.L

	def __getitem__(self,idx):
		row = self.transctions[idx]
		user = row[0]
		item = row[1]
		rating = row[2]
		negItem = self.get_neg(user, item)
		distribution = self.rating_distribution[item]
		return {"user": torch.tensor(user).to(torch.long), \
		        "item": torch.tensor(item).to(torch.long), \
		        "r_distribution": torch.tensor(distribution).to(torch.float), \
		        "rating": torch.tensor(rating).to(torch.float), \
		        "negItem": torch.tensor(negItem).to(torch.long)}

	def get_neg(self, userid, itemid):
		neg = list()
		hist = self.userHist[userid]
		for i in range(self.negNum):
			while True:
				negId = np.random.randint(self.itemNum)
				if negId not in hist and negId not in neg:
					neg.append(negId)
					break
		return neg

	def set_negN(self, n):
		if n < 1:
		    return
		self.negNum = n

class UserTransactionData(Dataset):
	"""docstring for UserTransactionData"""
	def __init__(self, transactions, userNum, itemNum, item_price, trainHist):
		super(UserTransactionData, self).__init__()
		self.transactions = transactions
		self.user = np.unique(np.array(transactions)[:,0])
		self.L = userNum
		self.userNum = self.L
		self.itemNum = itemNum
		self.negNum = 1000
		self.userHist = [[] for i in range(self.userNum)]
		self.trainHist = trainHist
		self.item_price = item_price
		for row in transactions:
		    self.userHist[row[0]].append(row[1])

	def __len__(self):
		return self.L

	def __getitem__(self, idx):
		user = self.user[idx]
		posItem = self.userHist[idx]
		posPrice = []
		for i in posItem:
		    posPrice.append(self.item_price[i])
		negPrice = []
		negItem = self.get_neg(idx)
		for i in negItem:
		    negPrice.append(self.item_price[i])
		budget = self.get_budget(idx)
		return {"user": torch.tensor(user).to(torch.long), \
		        "budget": torch.tensor(budget).to(torch.float), \
		        "posItem": torch.tensor(posItem).to(torch.long), \
		        "posPrice": torch.tensor(posPrice).to(torch.float), \
		        "negPrice": torch.tensor(negPrice).to(torch.float), \
		        "negItem": torch.tensor(negItem).to(torch.long)}

	def get_neg(self, userId):
		hist = self.userHist[userId] + self.trainHist[userId]
		neg = []
		for i in range(self.negNum):
		    while True:
		        negId = np.random.randint(self.itemNum)
		        if negId not in hist and negId not in neg:
		            neg.append(negId)
		            break
		return neg

	def set_negN(self, n):
		if n < 1: 
		    return
		self.negNum = n

	def get_budget(self, userId):
		price = []
		for i in self.trainHist[userId]:
		    price = self.item_price[i]
		budget = np.max(np.array(price))
		return budget



# if __name__ == "__main__":
# 	train, val, test = read_data("Movies")
# 	print(len(train))
# 	print(len(val))
# 	print(len(test))
# 	distribution = get_distribution("Movies")
# 	price = get_price("Movies")
# 	userNum, itemNum = get_datasize("Movies")
# 	trainData = TransactionData(train, userNum, itemNum, distribution)
# 	valData = UserTransactionData(val, userNum, itemNum, price, trainData.userHist)
# 	print(valData[0])
