import torch
from os.path import exists
import os
import numpy as np
import sys
import shutil

import Params
import Utils

class Trainer():

    def __init__(self, model,dataload_train,dataload_test,load=True,root_path=''):
        self.model = model
        self.root_path = root_path
        self.dataload_train = dataload_train
        self.dataload_test = dataload_test
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=Params.train_params.LR, weight_decay=Params.train_params.WGT_DECAY)
        self.criterion = torch.nn.CTCLoss().cuda()
        self.epochs=0
        self.loss_logger = []
        self.test_logger = []
        self.per_logger = []
        self.min_test_loss = 1000
        self.per = 1.
        self.model.get_n_params()
        self.folding = [[13,40],[46,27,42],[36,52],[49,21],[10,25],[45,44],[39,50],[59,23,48],[2,24],[8,34],[57,58],[33,41,7,29,19,26,61,47,53]]
        data_exist = exists(root_path+'Checkpoints/checkpoint.pt')
        if data_exist and load:
            self.load()


    def __len__(self):
        return self.epochs

    def train(self, epochs,test_interval=10,per_interval=50):
        """Train the model and output the train, test, and phoneme error rate (PER) to a txt file."""
        
        self.model.train()

        nSeqs = len(self.dataload_train)
        loss_avg = 0
        test_result=self.min_test_loss
        
        if Params.train_params.USE_TENSOR_CORES:
            scaler = torch.cuda.amp.GradScaler()
            torch.backends.cudnn.benchmark = True

        for i in range(epochs+1):
            loss_avg_aux=0
            for batch_idx, sample in enumerate(self.dataload_train):

                self.optimizer.zero_grad(set_to_none=True)
                
                batch_sz=len(sample['audio'])
                x = sample['audio'].cuda()
                x = x.view(batch_sz,1,Params.default.NMELCHANNELS,-1)

                if Params.train_params.USE_TENSOR_CORES:
                    with torch.cuda.amp.autocast():
                        y = self.model(x)
                        y = y.transpose(0,1) # (T,N,nClasses)
                        target = sample['phones'].cuda()

                        y_length = sample['audio_len'].cuda()
                        target_length = sample['phones_len'].cuda()

                        loss = self.criterion(y, target, y_length, target_length)
                else:
                    y = self.model(x)
                    y = y.transpose(0,1) # (T,N,nClasses)
                    target = sample['phones'].cuda()

                    y_length = sample['audio_len'].cuda()
                    target_length = sample['phones_len'].cuda()

                    loss = self.criterion(y, target, y_length, target_length)

                loss_avg_aux += loss.item()

                if Params.train_params.USE_TENSOR_CORES:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                if Params.train_params.GRADIENT_CLIP:
                    norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
                
                if Params.train_params.USE_TENSOR_CORES:
                    scaler.step(self.optimizer)
                    scaler.update()
                else:
                    self.optimizer.step()
            
            loss_avg=loss_avg_aux/nSeqs
            test_result=self.handle_logs_and_test(i,loss_avg,test_interval,test_result,per_interval)

    def handle_logs_and_test(self,i,loss_avg,test_interval,test,per_interval):
        self.loss_logger.append(loss_avg)
        if i%test_interval==0:
            test=self.test(dataload=self.dataload_test)
        self.test_logger.append(test)

        if i%per_interval==0:
            self.per=self.PER(dataload=self.dataload_test)
        self.per_logger.append(self.per)

        if i%test_interval==0:
            if i > 0:
                self.checkpoint()
            self.save_loss()
            if self.min_test_loss>test:
                self.save_best(prev_best=self.min_test_loss,curr_best=test)
                self.min_test_loss=test

        self.epochs=self.epochs+1
        print(self.epochs,loss_avg,test,self.per)
        return test

    def checkpoint(self):
        print("saving checkpoint..")
        torch.save({'model_state_dict': self.model.state_dict(),'optimizer_state_dict': self.optimizer.state_dict(), 'epochs' : self.epochs, 'loss_logger' : self.loss_logger,  'test_logger' : self.test_logger,  'per_logger' : self.per_logger}, self.root_path+'checkpoint.pt')

    def save_best(self,prev_best,curr_best):
        if not os.path.exists(self.root_path+"bests"):
            os.makedirs(self.root_path+"bests")
        print("saving best checkpoint..")
        name_cbest = str(curr_best)
        name_cbest = name_cbest[:7]
        name_pbest = str(prev_best)
        name_pbest = name_pbest[:7]

        checkpoint_name="checkpoint_"+name_pbest+".pt"
        if os.path.exists(self.root_path+"bests/"+checkpoint_name):
            os.remove(self.root_path+"bests/"+checkpoint_name)

        params_name="Params_"+name_pbest+".py"
        if os.path.exists(self.root_path+"bests/"+params_name):
            os.remove(self.root_path+"bests/"+params_name)

        checkpoint_name="checkpoint_"+name_cbest+".pt"
        params_name="Params_"+name_cbest+".py"
        torch.save({'model_state_dict': self.model.state_dict(),'optimizer_state_dict': self.optimizer.state_dict(), 'epochs' : self.epochs, 'loss_logger' : self.loss_logger,  'test_logger' : self.test_logger,  'per_logger' : self.per_logger}, self.root_path+"bests/"+checkpoint_name)
        shutil.copyfile(self.root_path+"Params.py", self.root_path+"bests/"+params_name)

    def load(self):
        print("loading checkpoint..")
        checkpoint = torch.load(self.root_path+'Checkpoints/checkpoint.pt')
        self.epochs = checkpoint['epochs']
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.loss_logger = checkpoint['loss_logger']
        self.test_logger = checkpoint['test_logger']
        self.per_logger = checkpoint['per_logger']
        self.min_test_loss = min(self.test_logger)

    def save_loss(self):
        f = open(self.root_path+'loss.txt', 'w')
        for idx in range(len(self.loss_logger)):
            f.write(str(idx)+" "+str(self.loss_logger[idx])+" "+str(self.test_logger[idx])+" "+str(self.per_logger[idx])+'\n')
        f.close()

    def test(self, dataload):
        self.model.eval()
        nSeqs = len(dataload)
        loss_avg_aux=0
        with torch.no_grad():
            for batch_idx, sample in enumerate(dataload):

                batch_sz=len(sample['audio'])
                x = sample['audio'].cuda()
                x = x.view(batch_sz,1,Params.default.NMELCHANNELS,-1)

                y = self.model(x)
                y = y.transpose(0,1) # (T,N,nClasses)

                target_length = sample['phones_len'].cuda()
                target = sample['phones'].cuda()

                y_length = sample['audio_len'].cuda()
                loss = self.criterion(y, target, y_length, target_length)
                loss_avg_aux += loss.item()

        loss_avg=loss_avg_aux/nSeqs
        self.model.train()
        return loss_avg

    def per_sample(self,sample): #phoneme error rate per sample
        self.model.eval()

        PER = 0
        with torch.no_grad():
            for i in range(len(sample['audio'])):
                x = sample['audio'][i].cuda()
                x = x.view(1,1,Params.default.NMELCHANNELS,-1)

                y = self.model(x)
                y = y.transpose(0,1) # (T,N,nClasses)
                y.squeeze()
                target = sample['phones'][i]

                torch.set_printoptions(threshold=sys.maxsize)
                z = y.cpu().detach()
                z = torch.argmax(z,2)
                z = z.transpose(0,1)
                z = z.squeeze()
                z = Utils.Delete_spaces_and_fold_phsec(z,self.folding)
                target = Utils.Delete_spaces_and_fold_phsec(target,self.folding)

                PER = PER + Utils.compute_per(target,z)
        self.model.train()
        return PER/len(sample['audio'])

    def PER(self,dataload):
        per = 0
        for batch_idx, sample in enumerate(dataload):
            per = per + self.per_sample(sample)
        return per/len(dataload)



