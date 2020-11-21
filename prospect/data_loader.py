import numpy as np
import math

from scipy.stats import norm
import copy
from torch.utils.data import Dataset, DataLoader
import torch

import os

prefix = '../data/'

def read_data(category):
	address = prefix + category + '/' + category + '_'
	with open(address + 'TrainSamples.txt', 'r') as f:
		data = f.readlines()
	TrainSamples = []
	for line in data:
		row = line[:-1].split(',')
		sample=[int(float(i)) for i in row]
		TrainSamples.append(sample)

	# with open(address+'ValidationSamples.txt', 'r') as f:
	# 	data = f.readlines()
	# ValSamples = []
	# for line in data:
	# 	row = line[:-1].split(',')
	# 	sample = [int(float(i)) for i in row]
	# 	ValSamples.append(sample)


	with open(address+'TestSamples.txt', 'r') as f:
		data = f.readlines()
	TestSamples = []
	for line in data:
		row = line[:-1].split(',')
		sample = [int(float(i)) for i in row]
		TestSamples.append(sample)

	return TrainSamples, TestSamples


# def convert(samples):
# 	# input one of [TrainSamples, ValSamples, TestSamples]
# 	length = len(samples)
# 	user_ids = []
# 	item_ids = []
# 	ratings = []
# 	for transaction in samples:
# 		user_ids.append(transaction[0])
# 		item_ids.append(transaction[1])
# 		ratings.append(transaction[2])
# 	user_ids = np.array(user_ids)
# 	user_ids.astype(np.int32)
# 	item_ids = np.array(item_ids)
# 	item_ids.astype(np.int32)
# 	ratings = np.array(ratings)
# 	ratings.astype(np.int32)
# 	# return type doesn't support index
# 	return interactions.Interactions(user_ids, item_ids, ratings)


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
		prob_ij.append(filter(cdf_ij[0]))
		prob_ij.append(filter(cdf_ij[1]-cdf_ij[0]))
		prob_ij.append(filter(cdf_ij[2]-cdf_ij[1]))
		prob_ij.append(filter(cdf_ij[3]-cdf_ij[2]))
		prob_ij.append(filter(1 - cdf_ij[3]))
		distribution.append(prob_ij)
	return np.array(distribution)

# def approx_Gaussian(frequency):
# 	distribution = []
# 	for i in range(len(frequency)):
# 		if list(frequency[i]).count(0) != 0:
# 			mu = 0
# 			for j in range(5):
# 				mu += (j+1) * frequency[i][j]
# 			sigma = 0
# 			for j in range(5):
# 				sigma += math.pow(j+1-mu,2) * frequency[i][j]
# 			if sigma == 0:
# 				sigma = 0.1
# 			prob_ij = []
# 			cdf_ij = []
# 			for r in range(1,5):
# 				cdf_ij.append(norm.cdf(r+0.5,mu,sigma))
# 			prob_ij.append(filter(cdf_ij[0]))
# 			prob_ij.append(filter(cdf_ij[1]-cdf_ij[0]))
# 			prob_ij.append(filter(cdf_ij[2]-cdf_ij[1]))
# 			prob_ij.append(filter(cdf_ij[3]-cdf_ij[2]))
# 			prob_ij.append(filter(1 - cdf_ij[3]))
# 			distribution.append(prob_ij)
# 		else:
# 			distribution.append(list(frequency[i]))
# 	return np.array(distribution)


def filter(prob):
	if prob <= 1e-4:
		return 1e-4
	elif prob >= 1-1e-4:
		return 1-1e-4
	else:
		return prob


def get_decumulative(distribution):
	decumulative = [[1.0] for i in range(distribution.shape[0])]
	# decumulative = copy.deepcopy(cumulative)
	for i in range(distribution.shape[0]):
		distribution_i = distribution[i]
		# print('distribution', distribution_i)
		# decumulative[i].append(1.0)
		for j in range(1, 6):
			summation = sum(distribution_i[:j])
			if summation >= 1.:
				decumulative[i].append(1e-10)
			elif summation <= 1e-10:
				decumulative[i].append(1.0)
			else:
				decumulative[i].append(1.-summation)
	return np.array(decumulative)


def get_datasize(category):
	address = prefix + category + "/" + category + "_" + "AllSamples.txt"
	AllSamples = list()
	with open(address,'r') as f:
		data = f.readlines()
	for line in data:
		row = line.rstrip().split(',')
		samples = [int(float(i)) for i in row]
		AllSamples.append(samples)
	all_data = np.array(AllSamples)
	userNum = len(np.unique(all_data[:,0]))
	itemNum = len(np.unique(all_data[:,1]))
	return userNum, itemNum

def get_price(category):
	address = prefix + category + "/" + category + "_" + "item_price.npy"
	price = np.load(address)
	return price

def get_distribution(category):
	address = prefix + category + "/" + category + "_" + "ItemResult.npy"
	distribution = np.load(address)
	return distribution


class TransactionData(Dataset):
	def __init__(self, transactions, userNum, itemNum, rating_distribution):
		super(TransactionData, self).__init__()
		self.transactions = transactions
		self.L = len(transactions)
		self.users = np.unique(np.array(transactions)[:, 0])
		self.userNum = userNum
		self.itemNum = itemNum
		self.negNum = 2
		self.rating_distribution = rating_distribution
		self.userHist = [[] for i in range(self.userNum)]
		for row in transactions:
			self.userHist[row[0]].append(row[1])

	def __len__(self):
		return self.L

	def __getitem__(self, idx):
		row = self.transactions[idx]
		user = row[0]
		item = row[1]
		rating = row[2]
		negItem = self.get_neg(user, item)
		distribution = self.rating_distribution[item]
		return {'user': np.array(user).astype(np.int64),
				'item': np.array(item).astype(np.int64),
				'r_distribution': np.array(distribution).astype(float),
				'rating': np.array(rating).astype(float),
				'negItem': np.array(negItem).astype(np.int64)
				}

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
	def __init__(self, transactions, userNum, itemNum, trainHist):
		super(UserTransactionData, self).__init__()
		self.transactions = transactions
		self.L = userNum
		self.user = np.unique(np.array(transactions)[:, 0])
		self.userNum = userNum
		self.itemNum = itemNum
		self.negNum = 10
		self.userHist = [[] for i in range(self.userNum)]
		self.trainHist = trainHist
		for row in transactions:
			self.userHist[row[0]].append(row[1])

	def __len__(self):
		return self.L

	def __getitem__(self, idx):
		user = self.user[idx]
		posItem = self.userHist[idx]

		negPrice = []
		negItem = self.get_neg(idx)

		return {'user': np.array(user).astype(np.int64),
				'posItem': np.array(posItem).astype(np.int64),
				'negItem': np.array(negItem).astype(np.int64)
				}

	def get_neg(self, userId):
		hist = self.userHist[userId]+self.trainHist[userId]
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


if __name__ == '__main__':
	train = read_data('Movielens')[0]
	# print(train[1])
	# inter = convert(train)
	# print(type(inter))


	category = 'Movielens'
	params = dict()
	params['batch_size'] = 32
	params['epoch_limit'] = 1
	params['w_decay'] = 1
	params['negNum_test'] = 1000
	params['negNum_train'] = 2
	params['l_size'] = 16


	train, test = read_data(category)
	userNum, itemNum = get_datasize(category)
	frequency = get_distribution(category)
	distribution = approx_Gaussian(frequency)

	trainset = TransactionData(train, userNum, itemNum, distribution)
	testset = UserTransactionData(test, userNum, itemNum, trainset.userHist)

	trainset.set_negN(params['negNum_train'])
	trainLoader = DataLoader(trainset, batch_size = params['batch_size'], shuffle=True, num_workers=0)

	testset.set_negN(params['negNum_test'])
	testLoader = DataLoader(testset, batch_size=1, shuffle=True, num_workers=0)

	for counter, batchData in enumerate(trainLoader):
		if counter == 1:
			break

		users = batchData['user'].numpy().astype(np.int32)
		print('users', type(users))
		print('keys: ', batchData.keys())
		print('r distribution', batchData['r_distribution'])
		print('summation', batchData['r_distribution'].sum(1))
		print('negItem', batchData['negItem'])
