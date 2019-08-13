import numpy as np
import torch
import torch.nn as nn
import torch.optim as opt
from tqdm import tqdm
# from tqdm import tqdm_notebook as tqdm
from heapq import heappush, heappop
from torch.utils.data import Dataset, DataLoader

import evaluation
import data_loader
from model import PT
import pdb

def main(category):
    params = dict()
    params['lr'] = 0.0001
    params['batch_size'] = 128
    params['epoch_limit'] = 3
    params['w_decay'] = 1
    params['negNum_test'] = 1000
    params['epsilon'] = 1e-4
    params['negNum_train'] = 4
    params['l_size'] = 16
    params['gpu']=True

    train, val, test = data_loader.read_data(category)
    item_price = data_loader.get_price(category)
    userNum, itemNum = data_loader.get_datasize(category)
    frequency = data_loader.get_distribution(category)
    distribution = data_loader.approx_Gaussian(frequency)
    trainset = data_loader.TransactionData(train, userNum, itemNum, distribution)
    valset = data_loader.UserTransactionData(val, userNum, itemNum, item_price, trainset.userHist)
    testset = data_loader.UserTransactionData(test,userNum, itemNum, item_price, trainset.userHist)


    trainset.set_negN(params['negNum_train'])
    trainLoader = DataLoader(trainset, batch_size = params['batch_size'], \
                            shuffle = True, num_workers = 0)
    valset.set_negN(params['negNum_test'])
    valLoader = DataLoader(valset, batch_size = 1, \
                            shuffle = True, num_workers = 0)
    testset.set_negN(params['negNum_test'])
    testLoader = DataLoader(testset, batch_size = 1, \
                            shuffle = True, num_workers = 0)

    model = PT(userLen = userNum, itemLen = itemNum,\
             distribution = distribution, item_price = item_price, params = params)
    model.to(model.device)
    optimizer = opt.SGD(model.parameters(), lr = params['lr'], \
            weight_decay = params['w_decay'])
    criterion = nn.Sigmoid()

    epsilon = params['epsilon']
    epoch = 0
    error = np.float('inf')

    trainErrorList = []
    valErrorList = []
    valHistory = []
    explodeTempreture = 3
    convergenceTempreture = 3
    print("starting training")
    while epoch < params['epoch_limit']:
        epoch += 1
        print("Epoch " + str(epoch) + " training...")
        L = len(trainLoader.dataset)
        pbar = tqdm(total = L)
        for i, batchData in enumerate(trainLoader):
#             if i >= 2:
#                 break
            optimizer.zero_grad()
            # get input
            users = torch.LongTensor(batchData['user']).to(model.device)
            items = torch.LongTensor(batchData['item']).to(model.device)
            negItems = torch.LongTensor(batchData['negItem']).reshape(-1).to(model.device)

            nusers = users.view(-1,1).to(model.device)
            nusers = nusers.expand(nusers.shape[0], params['negNum_train']).reshape(-1).to(model.device)

    #         print('users: %d' % users.shape)
    #         print('items: %d' % items.shape)
    #         print('neg_users: %d' % nusers.shape)
    #         print('neg_items: %d' % negItems.shape)
            
            pOut = model.forward(users,items).view(-1,1).expand(users.shape[0],params['negNum_train']).reshape(-1,1)
    #         print(pOut.shape)

            nOut = model.forward(nusers, negItems).reshape(-1,1)
    #         print(nOut.shape)
            diff = torch.mean(pOut - nOut)
    #             print (totalOut.shape)
            loss = - criterion(diff)
    #             print(loss.shape)
            loss.backward()
            model.filter_grad()
#             pdb.set_trace()
            optimizer.step()
            pbar.update(users.shape[0])
        pbar.close()
        

        #validation
        print("Epoch " + str(epoch) + " validating...")
        L = len(valLoader.dataset)
        pbar = tqdm(total = L)
        model.eval()
        with torch.no_grad():
            scoreDict = dict()
            for i, batchData in enumerate(valLoader):
                if i >= 1000:
                    break
                user = torch.LongTensor(batchData['user']).to(model.device)
                posItems = torch.LongTensor(batchData['posItem']).to(model.device)
                negItems = torch.LongTensor(batchData['negItem']).to(model.device)
                budget = torch.FloatTensor(batchData['budget']).to(model.device)
                posPrices = torch.FloatTensor(batchData['posPrice']).to(model.device)
                negPrices = torch.FloatTensor(batchData['negPrice']).to(model.device)

                items = torch.cat((posItems, negItems),1).view(-1)
                prices = torch.cat((posPrices, negPrices),1).view(-1)
                users = user.expand(items.shape[0])

                out = model.forward(users,items)
                scoreHeap = list()
                for j in range(out.shape[0]):
                    gt = False
                    if j < posItems.shape[1]:
                        gt = True
#                     if prices[j] > budget:
#                         heappush(scoreHeap, (1000, (0 + items[j].cpu().numpy(), gt)))
#                     else:
#                         heappush(scoreHeap, (1 - out[j].cpu().numpy(), (0 + items[j].cpu().numpy(), gt)))
                    heappush(scoreHeap, (1 - out[j].cpu().numpy(), (0 + items[j].cpu().numpy(), gt)))
                scores = list()
                candidate = len(scoreHeap)
                for k in range(candidate):
                    scores.append(heappop(scoreHeap))
                pbar.update(1)
#                 pdb.set_trace()
                scoreDict[user[0]] = (scores, posItems.shape[1])
        pbar.close()

        valHistory.append(evaluation.ranking_performance(scoreDict,100))
        valError = 1 - valHistory[-1]["avg_ndcg"][0]
        valErrorList.append(valError)
        improvement = np.abs(error - valError)
        error = valError
        if improvement < epsilon:
            break

    # test
    print("starting test...")
    L = len(testLoader.dataset)
    pbar = tqdm(total = L)
    with torch.no_grad():
        scoreDict = dict()
        for i, batchData in enumerate(testLoader):
            user = torch.LongTensor(batchData['user']).to(model.device)
            posItems = torch.LongTensor(batchData['posItem']).to(model.device)
            negItems = torch.LongTensor(batchData['negItem']).to(model.device)
            budget = torch.FloatTensor(batchData['budget']).to(model.device)
            posPrices = torch.FloatTensor(batchData['posPrice']).to(model.device)
            negPrices = torch.FloatTensor(batchData['negPrice']).to(model.device)

            items = torch.cat((posItems, negItems),1).view(-1)
            prices = torch.cat((posPrices, negPrices),1).view(-1)
            users = user.expand(items.shape[0])

            out = model.forward(users,items)
            scoreHeap = list()
            for j in range(out.shape[0]):
                gt = False
                if j < posItems.shape[1]:
                    gt = True
                if prices[j] > budget:
                    heappush(scoreHeap, (1000, (0 + items[j].cpu().numpy(), gt)))
                else:
                    heappush(scoreHeap, (1 - out[j].cpu().numpy(), (0 + items[j].cpu().numpy(), gt)))
            scores = list()
            candidate = len(scoreHeap)
            for k in range(candidate):
                scores.append(heappop(scoreHeap))
            pbar.update(1)
            scoreDict[user[0]] = (scores, posItems.shape[1])
    pbar.close()
    testResult = evaluation.ranking_performance(scoreDict,100)


if __name__ == '__main__':
    main("Movies")
    # print(torch.tensor(1))
#     category = "Movies"
#     frequency = data_loader.get_distribution(category)
#     distribution = data_loader.approx_Gaussian(frequency)
#     userNum, itemNum = data_loader.get_datasize(category)
#     for i in range(itemNum):
#         for j in range(5):
#             if distribution[i][j] < 0.001:
#                 print("=========")
#                 print("("+str(i)+","+str(j)+")")
#                 print(distribution[i][j])
                
