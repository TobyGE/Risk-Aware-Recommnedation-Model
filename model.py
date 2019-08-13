
import numpy as np
import torch
import torch.nn as nn
import data_loader
import torch.optim as opt
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm_notebook as tqdm
import pdb



class MF(nn.Module):
    """
        - userLen: the number of users
        - itemLen: the number of items
        - params: the parameters dict used for constructing model
            - l_size: latent dimension size
            - gpu: True/False, whether using GPU
            
    """
    def __init__(self, userLen, itemLen, params):
        super(MF, self).__init__()
        self.userNum = userLen
        self.itemNum = itemLen
        self.params = params
        if 'gpu' in params and params['gpu'] == True:
            self.device = 'cuda'
        else:
            self.device = 'cpu'

        l_size = params['l_size']
        
        """
            Initialize  global bias,
                        user bias,
                        item bias,
                        user embedding,
                        item embedding
        """
        self.globalBias = nn.Embedding(1,1).to(self.device)
        self.uBias = nn.Embedding(userLen,1).to(self.device)
        self.itemBias = nn.Embedding(itemLen,1).to(self.device)
        self.uEmbed = nn.Embedding(userLen, l_size).to(self.device)
        self.itemEmbed = nn.Embedding(itemLen, l_size).to(self.device)
    
    def forward(self, users, items):
        uE = self.uEmbed(users)
        uB = self.uBias(users)
        iE = self.itemEmbed(items)
        iB = self.itemBias(items)
        try: 
            gB = self.globalBias.weight.data.expand(users.shape[0],users.shape[1])
            score = gB + uB + iB + torch.mul(uE, iE).sum(2)
        except:
            gB = self.globalBias.weight.data.expand(users.shape[0],1)
            score = gB + uB + iB + torch.mul(uE, iE).sum(1).view(-1,1)
        return score

class MUD(nn.Module):
    """docstring for MUD"""
    def __init__(self, userLen, itemLen, distribution, item_price, RMF, params):
        super(MUD, self).__init__()
        self.userNum = userLen
        self.itemNum = itemLen
        self.distribution = torch.tensor(distribution).to(torch.float)
        self.price = torch.tensor(item_price).to(torch.float)
        self.rating = RMF
        self.params = params
        if 'gpu' in params and params['gpu'] == True:
            self.device = 'cuda'
        else:
            self.device = 'cpu'
        l_size = params['l_size']

        self.gBias = nn.Embedding(1,1).to(self.device)
        self.uBias = nn.Embedding(userLen,1).to(self.device)
        self.itemBias = nn.Embedding(itemLen,1).to(self.device)
        self.uEmbed = nn.Embedding(userLen, l_size).to(self.device)
        self.itemEmbed = nn.Embedding(itemLen, l_size).to(self.device)

    def forward(self, users, items):
        uE = self.uEmbed(users)
        uB = self.uBias(users)
        iE = self.itemEmbed(items)
        iB = self.itemBias(items)

        try:
            gB = self.gBias.weight.data.expand(users.shape[0],users.shape[1])
            alpha = gB + uB + iB + torch.mul(uE, iE).sum(2)
        except:
            gB = self.gBias.weight.data.expand(users.shape[0],1)
            alpha = gB + uB + iB + torch.mul(uE, iE).sum(1).view(-1,1)
        with torch.no_grad():
            r = self.rating.forward(users, items)
            tanh_r = torch.tanh(r).view(-1,1)
        price = self.price[items]
        u = torch.mul(alpha, tanh_r)
        out = 0.5 * torch.div(u.view(-1), torch.sigmoid(price))
        return out

    def EU(self, users, items):
        gB = self.gBias.weight.data.expand(users.shape[0],1)
        uE = self.uEmbed(users)
        uB = self.uBias(users)
        iE = self.itemEmbed(items)
        iB = self.itemBias(items)
        
        alpha = gB + uB + iB + torch.mul(uE, iE).sum(1).view(-1,1).expand(users.shape[0],5)
        distribution = self.distribution[items]
        rating = torch.tensor([1,2,3,4,5]).expand(users.shape[0],5).to(torch.float)
        tanh_r = torch.tanh(rating)
        U = torch.log(torch.tensor(2).to(torch.float)) * torch.mul(alpha, tanh_r)
        EU = torch.mul(distribution, U).sum(1)
        return EU

    def UE(self, users, items):
        gB = self.gBias.weight.data.expand(users.shape[0],1)
        uE = self.uEmbed(users)
        uB = self.uBias(users)
        iE = self.itemEmbed(items)
        iB = self.itemBias(items)
        
        alpha = gB + uB + iB + torch.mul(uE, iE).sum(1).view(-1,1)
        distribution = self.distribution[items]
        rating = torch.tensor([1,2,3,4,5]).expand(users.shape[0],5).to(torch.float)
        r_bar = torch.mul(distribution, rating).sum(1)
        tanh_r_bar = torch.tanh(r_bar)
        UE = torch.log(torch.tensor(2).to(torch.float)) * torch.mul(alpha, tanh_r_bar)
        return UE

      
class PT(nn.Module):
    def __init__(self, userLen, itemLen, distribution, item_price, params):
        super(PT, self).__init__()
        self.userNum = userLen
        self.itemNum = itemLen
        self.params = params
        if 'gpu' in params and params['gpu'] == True:
            self.device = 'cuda'
        else:
            self.device = 'cpu'

        l_size = params['l_size']
        self.distribution = torch.FloatTensor(distribution).to(self.device)
        self.item_price = torch.FloatTensor(item_price).to(self.device)

        """
            Initialize  global bias,
                        user bias,
                        item bias,
                        user embedding,
                        item embedding,
            For variable 
                        alpha, 
                        beta, 
                        gamma, 
                        delta,
                        reference point
        """
        self.globalBias_a = nn.Embedding(1,1).to(self.device)
        self.globalBias_a.weight.data = torch.tensor(1).to(torch.float)
        self.uBias_a = nn.Embedding(userLen,1).to(self.device)
        self.uBias_a.weight.data = torch.zeros_like(self.uBias_a.weight.data)
        self.itemBias_a = nn.Embedding(itemLen,1).to(self.device)
        self.itemBias_a.weight.data = torch.zeros_like(self.itemBias_a.weight.data)
        self.uEmbed_a = nn.Embedding(userLen, l_size).to(self.device)
        self.uEmbed_a.weight.data = torch.tensor(np.random.normal(scale=1./l_size, 
                        size=(userLen, l_size))).to(torch.float)
        self.itemEmbed_a = nn.Embedding(itemLen, l_size).to(self.device)
        self.itemEmbed_a.weight.data = torch.tensor(np.random.normal(scale=1./l_size, 
                        size=(itemLen, l_size))).to(torch.float)
        
        self.globalBias_b = nn.Embedding(1,1).to(self.device)
        self.globalBias_b.weight.data = torch.tensor(1).to(torch.float)
        self.uBias_b = nn.Embedding(userLen,1).to(self.device)
        self.uBias_b.weight.data = torch.zeros_like(self.uBias_b.weight.data)
        self.itemBias_b = nn.Embedding(itemLen,1).to(self.device)
        self.itemBias_b.weight.data = torch.zeros_like(self.itemBias_b.weight.data)
        self.uEmbed_b = nn.Embedding(userLen, l_size).to(self.device)
        self.uEmbed_b.weight.data = torch.tensor(np.random.normal(scale=1./l_size, 
                        size=(userLen, l_size))).to(torch.float)
        self.itemEmbed_b = nn.Embedding(itemLen, l_size).to(self.device)
        self.itemEmbed_b.weight.data = torch.tensor(np.random.normal(scale=1./l_size, 
                        size=(itemLen, l_size))).to(torch.float)
        
        self.globalBias_g = nn.Embedding(1,1).to(self.device)
        self.globalBias_g.weight.data = torch.tensor(1).to(torch.float)
        self.uBias_g = nn.Embedding(userLen,1).to(self.device)
        self.uBias_g.weight.data = torch.zeros_like(self.uBias_g.weight.data)
        self.itemBias_g = nn.Embedding(itemLen,1).to(self.device)
        self.itemBias_g.weight.data = torch.zeros_like(self.itemBias_g.weight.data)
        self.uEmbed_g = nn.Embedding(userLen, l_size).to(self.device)
        self.uEmbed_g.weight.data = torch.tensor(np.random.normal(scale=1./l_size, 
                        size=(userLen, l_size))).to(torch.float)
        self.itemEmbed_g = nn.Embedding(itemLen, l_size).to(self.device)
        self.itemEmbed_g.weight.data = torch.tensor(np.random.normal(scale=1./l_size, 
                        size=(itemLen, l_size))).to(torch.float)
        
        self.globalBias_d = nn.Embedding(1,1).to(self.device)
        self.globalBias_d.weight.data = torch.tensor(1).to(torch.float)
        self.uBias_d = nn.Embedding(userLen,1).to(self.device)
        self.uBias_d.weight.data = torch.zeros_like(self.uBias_d.weight.data)
        self.itemBias_d = nn.Embedding(itemLen,1).to(self.device)
        self.itemBias_d.weight.data = torch.zeros_like(self.itemBias_d.weight.data)
        self.uEmbed_d = nn.Embedding(userLen, l_size).to(self.device)
        self.uEmbed_d.weight.data = torch.tensor(np.random.normal(scale=1./l_size, 
                        size=(userLen, l_size))).to(torch.float)
        self.itemEmbed_d = nn.Embedding(itemLen, l_size).to(self.device)
        self.itemEmbed_d.weight.data = torch.tensor(np.random.normal(scale=1./l_size, 
                        size=(itemLen, l_size))).to(torch.float)

        self.reference_point = nn.Embedding(userLen,1).to(self.device)
        self.reference_point.weight.data = torch.ones_like(self.reference_point.weight.data) * 1.5
        self.to(self.device)
    
    def filter_grad(self):
        checkList = [self.uEmbed_g.weight.grad, \
                     self.itemEmbed_g.weight.grad, \
                     self.uBias_g.weight.grad, \
                     self.itemBias_g.weight.grad, \
                     self.uEmbed_d.weight.grad, \
                     self.itemEmbed_d.weight.grad, \
                     self.uBias_d.weight.grad, \
                     self.itemBias_d.weight.grad]
        for A in checkList:
            A[A != A] = 0
            
        
    def forward(self, users, items):
        gB_a = self.globalBias_a.weight.data
        uE_a = self.uEmbed_a(users)
        uB_a = self.uBias_a(users)
        iE_a = self.itemEmbed_a(items)
        iB_a = self.itemBias_a(items)
        
        gB_b = self.globalBias_b.weight.data
        uE_b = self.uEmbed_b(users)
        uB_b = self.uBias_b(users)
        iE_b = self.itemEmbed_b(items)
        iB_b = self.itemBias_b(items)
        
        gB_g = self.globalBias_g.weight.data
        uE_g = self.uEmbed_g(users)
        uB_g = self.uBias_g(users)
        iE_g = self.itemEmbed_g(items)
        iB_g = self.itemBias_g(items)
        
        gB_d = self.globalBias_d.weight.data
        uE_d = self.uEmbed_d(users)
        uB_d = self.uBias_d(users)
        iE_d = self.itemEmbed_d(items)
        iB_d = self.itemBias_d(items)

        price = self.item_price[items].view(-1,1).expand(users.shape[0], 5).to(self.device)


        alpha = gB_a + uB_a + iB_a + torch.mul(uE_a, iE_a).sum(1).view(-1,1)
        alpha = torch.max(torch.ones_like(alpha).to(self.device) * 0.1, alpha)
        beta = gB_b + uB_b + iB_b + torch.mul(uE_b, iE_b).sum(1).view(-1,1)
        beta = torch.max(torch.ones_like(beta).to(self.device) * 0.1, beta)
        gamma = gB_g + uB_g + iB_g + torch.mul(uE_g, iE_g).sum(1).view(-1,1)
        gamma = torch.max(torch.ones_like(gamma).to(self.device) * 0.1, gamma)
        delta = gB_d + uB_d + iB_d + torch.mul(uE_d, iE_d).sum(1).view(-1,1)
        delta = torch.max(torch.ones_like(delta).to(self.device) * 0.1, delta)
        reference_point = self.reference_point(users)
        
        distribution = self.distribution[items].to(self.device)
        rating = torch.tensor([1,2,3,4,5]).expand(users.shape[0],5).to(torch.float).to(self.device)
        
        tanh_r = torch.tanh(rating - reference_point)
        tanh_r_pos = torch.gt(tanh_r, torch.FloatTensor([0]).to(self.device)).to(torch.float)
        tanh_r_neg = torch.ones_like(tanh_r).to(self.device) - tanh_r_pos
        
        """
            calculate weights
        """
        temp_pos = torch.mul(tanh_r_pos,distribution)
        weights_pos_numerator = temp_pos.pow(gamma)
        weights_pos_denominator =  (temp_pos.pow(gamma) + (torch.ones_like(temp_pos).to(self.device)-temp_pos).pow(gamma)).pow(1/gamma)
        weight_pos = torch.div(weights_pos_numerator,weights_pos_denominator)
        
        temp_neg = torch.mul(tanh_r_neg,distribution)
        weights_neg_numerator = temp_neg.pow(delta)
        weights_neg_denominator =  (temp_neg.pow(delta) + (torch.ones_like(temp_neg).to(self.device)-temp_neg).pow(delta)).pow(1/delta)
        weight_neg = torch.div(weights_neg_numerator,weights_neg_denominator)
        
        weight = weight_pos + weight_neg
#         pdb.set_trace()
#         weight = distribution
        
        """
        calculate values
        """
        
        r_pos = torch.mul(tanh_r, tanh_r_pos)
        x_pos = torch.mul(r_pos, price)
        temp_pos_val = torch.mul(alpha, torch.log(x_pos + 1) )
        r_neg = torch.mul(tanh_r, tanh_r_neg)
        x_neg = torch.mul(r_neg, price)
        temp_neg_val = torch.mul(-beta, torch.log(1 - x_neg) )
        value = temp_pos_val + temp_neg_val
        """
        calculate prospect value using weights and values
        """
        prospect_value = torch.mul(weight, value).sum(1)
        if np.isnan(prospect_value.detach().cpu().numpy()).any():
            print("gB_g: \n" + str(gB_g))
            print("uB_g: \n" + str(uB_g))
            print("iB_g: \n" + str(iB_g))
            print("uE_g: \n" + str(uE_g))
            print("iE_g: \n" + str(iE_g))
            print("gamma: \n" + str(gamma))
            print("weight_pos: \n" + str(weight_pos))
            print("weights_pos_denominator: \n" + str(weights_pos_denominator))
            print("delta: \n" + str(delta))
            print("weight_neg: \n" + str(weight_neg))
            print("weights_neg_denominator: \n" + str(weights_neg_denominator))
        # print(prospect_value.device)
        return prospect_value
        
        
