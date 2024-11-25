import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import *
from torch_geometric.nn import GCNConv, GATConv  # Graph convolutional layers
from torch_geometric.utils import add_self_loops

class LSTMBase(nn.Module):
    def __init__(self,device,cond_vocab_size,proc_vocab_size,med_vocab_size,out_vocab_size,chart_vocab_size,lab_vocab_size,eth_vocab_size,gender_vocab_size,age_vocab_size,ins_vocab_size,modalities,embed_size,rnn_size,latent_size, rnnLayers, batch_size):
        super(LSTMBase, self).__init__()
        self.embed_size=embed_size
        self.latent_size=latent_size
        self.rnn_size=rnn_size
        self.cond_vocab_size=cond_vocab_size
        self.proc_vocab_size=proc_vocab_size
        self.med_vocab_size=med_vocab_size
        self.out_vocab_size=out_vocab_size
        self.chart_vocab_size=chart_vocab_size
        self.lab_vocab_size=lab_vocab_size
        
        self.eth_vocab_size=eth_vocab_size
        self.gender_vocab_size=gender_vocab_size
        self.age_vocab_size=age_vocab_size
        self.ins_vocab_size=ins_vocab_size

        self.rnnLayers=rnnLayers
        self.batch_size=batch_size
        self.padding_idx = 0
        self.device=device
        self.modalities=modalities
        self.build()
        
    def build(self):
        
        if self.med_vocab_size:
            self.med=ValEmbed(self.device,self.med_vocab_size,self.embed_size,self.latent_size)                
        if self.proc_vocab_size:
            self.proc=CodeEmbed(self.device,self.proc_vocab_size,self.embed_size,self.latent_size)
        if self.out_vocab_size:
            self.out=CodeEmbed(self.device,self.out_vocab_size,self.embed_size,self.latent_size)
        if self.chart_vocab_size:
            self.chart=ValEmbed(self.device,self.chart_vocab_size,self.embed_size,self.latent_size)
        if self.lab_vocab_size:
            self.lab=ValEmbed(self.device,self.lab_vocab_size,self.embed_size,self.latent_size)
        
        if self.cond_vocab_size:
            self.cond=StatEmbed(self.device,self.cond_vocab_size,self.embed_size,self.latent_size)
        
        self.ethEmbed=nn.Embedding(self.eth_vocab_size,self.latent_size,self.padding_idx) 
        self.genderEmbed=nn.Embedding(self.gender_vocab_size,self.latent_size,self.padding_idx) 
        self.ageEmbed=nn.Embedding(self.age_vocab_size,self.latent_size,self.padding_idx) 
        self.insEmbed=nn.Embedding(self.ins_vocab_size,self.latent_size,self.padding_idx) 
       
        
        self.embedfc=nn.Linear((self.latent_size*(self.modalities+4)), self.latent_size, True)
        self.rnn=nn.LSTM(input_size=self.latent_size,hidden_size=self.rnn_size,num_layers = self.rnnLayers,batch_first=True)
        self.fc1=nn.Linear(self.rnn_size, int((self.rnn_size)/2), True)
        self.fc2=nn.Linear(int((self.rnn_size)/2), 1, True)
        
        #self.sig = nn.Sigmoid()
        
    def forward(self,meds,chart,out,proc,lab,conds,demo):   
#         if interpret:
#             meds,chart,out,proc,lab,conds,demo=X[0],X[1],X[2],X[3],X[4],X[5],X[6]
        #print(meds[0])
        
        out1=torch.zeros(size=(0,0))
#         print("out",out1.shape)
#         print(meds.shape)
#         print(chart.shape)
#         print(out.shape)
#         print(proc.shape)
#         print("Within the model:" , lab.shape)
#         print(conds.shape)
#         print(demo.shape)
#         print(conds[:,0:10])
        #print(demo)
#         if demo.shape[0]>self.batch_size:
#             print(demo[0],demo[200],demo[400],demo[600],demo[800])
        if meds.shape[0]:
            if meds.shape[0]>self.batch_size:
                meds=meds[-self.batch_size:]
            medEmbedded=self.med(meds)
            
            if out1.nelement():
                out1=torch.cat((out1,medEmbedded),2)
            else:
                out1=medEmbedded
            #print(out1.shape)
            #print(out1.nelement())
        if proc.shape[0]:
            if proc.shape[0]>self.batch_size:
                proc=proc[-self.batch_size:]
            procEmbedded=self.proc(proc)
            
            if out1.nelement():
                out1=torch.cat((out1,procEmbedded),2)
            else:
                out1=procEmbedded
        if lab.shape[1]:
            if lab.shape[0]>self.batch_size:
                lab=lab[-self.batch_size:]
            labEmbedded=self.lab(lab)
            #print("lab",labEmbedded.shape)
            if out1.nelement():
                out1=torch.cat((out1,labEmbedded),2)
            else:
                out1=labEmbedded
#         print("out",out1.shape)
        if out.shape[0]:
            if out.shape[0]>self.batch_size:
                out=out[-self.batch_size:]
            outEmbedded=self.out(out)
            
            if out1.nelement():
                out1=torch.cat((out1,outEmbedded),2)
            else:
                out1=outEmbedded

            
        if chart.shape[0]:
            if chart.shape[0]>self.batch_size:
                chart=chart[-self.batch_size:]
            chartEmbed=self.chart(chart)
#             print("chartEmbed",chartEmbed.shape)
#             print(chartEmbed[5,:,0:10])
            if out1.nelement():
                out1=torch.cat((out1,chartEmbed),2)
            else:
                out1=chartEmbed
        
#         print("out",out1.shape)
        if conds.shape[0]>self.batch_size:
                conds=conds[-self.batch_size:]
        conds=conds.to(self.device)
        #print("[IN] Conds shape: ", conds.shape)
        condEmbed=self.cond(conds)
        condEmbed=condEmbed.unsqueeze(1)
        condEmbed=condEmbed.repeat(1,out1.shape[1],1)
        condEmbed=condEmbed.type(torch.FloatTensor)
        condEmbed=condEmbed.to(self.device)
#         print("cond",condEmbed.shape)
        out1=torch.cat((out1,condEmbed),2)
#         print("cond",condEmbed.shape)
        
#         print("out",out1.shape)
        if demo.shape[0]>self.batch_size:
                demo=demo[-self.batch_size:]
        gender=demo[:,0].to(self.device)
        gender=gender.type(torch.LongTensor)
        gender=gender.to(self.device)
        gender=self.genderEmbed(gender)
        gender=gender.unsqueeze(1)
        gender=gender.repeat(1,out1.shape[1],1)
        gender=gender.type(torch.FloatTensor)
        gender=gender.to(self.device)
        out1=torch.cat((out1,gender),2)
#         print(gender.shape)
        
        eth=demo[:,1].to(self.device)
        eth=eth.type(torch.LongTensor)
        eth=eth.to(self.device)
        eth=self.ethEmbed(eth)
        eth=eth.unsqueeze(1)
        eth=eth.repeat(1,out1.shape[1],1)
        eth=eth.type(torch.FloatTensor)
        eth=eth.to(self.device)
        out1=torch.cat((out1,eth),2)
#         print(eth.shape)
        
        ins=demo[:,2].to(self.device)
        ins=ins.type(torch.LongTensor)
        ins=ins.to(self.device)
        ins=self.insEmbed(ins)
        ins=ins.unsqueeze(1)
        ins=ins.repeat(1,out1.shape[1],1)
        ins=ins.type(torch.FloatTensor)
        ins=ins.to(self.device)
        out1=torch.cat((out1,ins),2)
#         print(ins.shape)
        
        age=demo[:,3].to(self.device)
        age=age.type(torch.LongTensor)
        age=age.to(self.device)
        age=self.ageEmbed(age)
        age=age.unsqueeze(1)
        age=age.repeat(1,out1.shape[1],1)
        age=age.type(torch.FloatTensor)
        age=age.to(self.device)
        out1=torch.cat((out1,age),2)
#         print(age.shape)
        
#         print("out",out1.shape)
        
        out1=out1.type(torch.FloatTensor)
        out1=out1.to(self.device)
        out1=self.embedfc(out1)
        #print("fcout",out1.shape)
        
        h_0, c_0 = self.init_hidden()
        h_0, c_0 = h_0.to(self.device), c_0.to(self.device)
        
        _, (out1, code_c_n)=self.rnn(out1, (h_0, c_0))
        out1=out1[-1,:,:]
        out1=out1.squeeze()
        #print("rnnout",out1.shape)
        
        
        out1 = self.fc1(out1)
        out1 = self.fc2(out1)
        #print("out1",out1.shape)
        
        
        #print("sig out",sigout1[16])
        
        
        sig = nn.Sigmoid()
        sigout1=sig(out1)
        #return sigout1
        return sigout1,out1
    
    def init_hidden(self):
        # initialize the hidden state and the cell state to zeros
        h=torch.zeros(self.rnnLayers,self.batch_size, self.rnn_size)
        c=torch.zeros(self.rnnLayers,self.batch_size, self.rnn_size)

#         if self.hparams.on_gpu:
#             hidden_a = hidden_a.cuda()
#             hidden_b = hidden_b.cuda()

        h = Variable(h)
        c = Variable(c)

        return (h, c)    
    
    
    
class LSTMBaseH(nn.Module):
    def __init__(self,device,cond_vocab_size,proc_vocab_size,med_vocab_size,out_vocab_size,chart_vocab_size,lab_vocab_size,eth_vocab_size,gender_vocab_size,age_vocab_size,ins_vocab_size,modalities,embed_size,rnn_size,latent_size, rnnLayers, batch_size):
        super(LSTMBaseH, self).__init__()
        self.embed_size=embed_size
        self.latent_size=latent_size
        self.rnn_size=rnn_size
        self.cond_vocab_size=cond_vocab_size
        self.proc_vocab_size=proc_vocab_size
        self.med_vocab_size=med_vocab_size
        self.out_vocab_size=out_vocab_size
        self.chart_vocab_size=chart_vocab_size
        self.lab_vocab_size=lab_vocab_size
        
        self.eth_vocab_size=eth_vocab_size
        self.gender_vocab_size=gender_vocab_size
        self.age_vocab_size=age_vocab_size
        self.ins_vocab_size=ins_vocab_size

        self.rnnLayers = rnnLayers
        self.batch_size=batch_size
        self.padding_idx = 0
        self.device=device
        self.modalities=modalities
        self.build()
        
    def build(self):
            
        if self.med_vocab_size:
            self.med=ValEmbed(self.device,self.med_vocab_size,self.embed_size,self.latent_size)                
        if self.proc_vocab_size:
            self.proc=CodeEmbed(self.device,self.proc_vocab_size,self.embed_size,self.latent_size)
        if self.out_vocab_size:
            self.out=CodeEmbed(self.device,self.out_vocab_size,self.embed_size,self.latent_size)
        if self.chart_vocab_size:
            self.chart=ValEmbed(self.device,self.chart_vocab_size,self.embed_size,self.latent_size)
        if self.lab_vocab_size:
            self.lab=ValEmbed(self.device,self.lab_vocab_size,self.embed_size,self.latent_size)
        
        if self.cond_vocab_size:
            self.cond=StatEmbed(self.device,self.cond_vocab_size,self.embed_size,self.latent_size)
        
        self.ethEmbed=nn.Embedding(self.eth_vocab_size,self.latent_size,self.padding_idx) 
        self.genderEmbed=nn.Embedding(self.gender_vocab_size,self.latent_size,self.padding_idx) 
        self.ageEmbed=nn.Embedding(self.age_vocab_size,self.latent_size,self.padding_idx) 
        self.insEmbed=nn.Embedding(self.ins_vocab_size,self.latent_size,self.padding_idx) 
       
        
        self.embedfc=nn.Linear((self.latent_size*(self.modalities-1)), self.latent_size, True)
        self.statfc=nn.Linear(int(self.latent_size*5), self.latent_size, True)
        self.statfc2=nn.Linear(self.latent_size, self.rnn_size, True)
        self.rnn=nn.LSTM(input_size=self.latent_size,hidden_size=self.rnn_size,num_layers = self.rnnLayers,batch_first=True)
        self.fc1=nn.Linear(self.rnn_size*2, self.rnn_size, True)
        self.fc2=nn.Linear(self.rnn_size, 1, False)
        
        #self.sig = nn.Sigmoid()
    
#     def model_interpret(self,net,X):
        
#         print("======= INTERPRETING ========")
#         deep_lift=IntegratedGradients(net)
#         attr=deep_lift.attribute(torch.tensor(X).float(),target=0.)
#         print(attr)
#         print(attr.shape)
        
        
    def forward(self,meds,chart,out,proc,lab,conds,demo):   
        #print(len(X))
        #print(X[4].shape)
        #meds,chart,out,proc,lab,conds,demo=X[0],X[1],X[2],X[3],X[4],X[5],X[6] 
        
        out1=torch.zeros(size=(0,0))
        
        if meds.shape[0]:
            medEmbedded=self.med(meds)
            
            if out1.nelement():
                out1=torch.cat((out1,medEmbedded),2)
            else:
                out1=medEmbedded
            #print(out1.shape)
            #print(out1.nelement())
        if proc.shape[0]:
            procEmbedded=self.proc(proc)
            
            if out1.nelement():
                out1=torch.cat((out1,procEmbedded),2)
            else:
                out1=procEmbedded
        if lab.shape[1]:
            labEmbedded=self.lab(lab)
            #self.model_interpret(self.lab,lab)
            if out1.nelement():
                out1=torch.cat((out1,labEmbedded),2)
            else:
                out1=labEmbedded
        if out.shape[0]:
            outEmbedded=self.out(out)
            
            if out1.nelement():
                out1=torch.cat((out1,outEmbedded),2)
            else:
                out1=outEmbedded

            
        if chart.shape[0]:
            chartEmbed=self.chart(chart)
#             print("chartEmbed",chartEmbed.shape)
#             print(chartEmbed[5,:,0:10])
            if out1.nelement():
                out1=torch.cat((out1,chartEmbed),2)
            else:
                out1=chartEmbed

        
        out1=out1.type(torch.FloatTensor)
        out1=out1.to(self.device)
        out1=self.embedfc(out1)
        
        out2=torch.zeros(size=(0,0))
        conds=conds.to(self.device)
        condEmbed=self.cond(conds)
        
        
        condEmbed=condEmbed.type(torch.FloatTensor)
        condEmbed=condEmbed.to(self.device)
        out2=condEmbed
        #print("cond",condEmbed.shape)
        
        gender=demo[:,0].to(self.device)
        gender=self.genderEmbed(gender)
        gender=gender.type(torch.FloatTensor)
        gender=gender.to(self.device)
        out2=torch.cat((out2,gender),1)
#         print(gender.shape)
        
        eth=demo[:,1].to(self.device)
        eth=self.ethEmbed(eth)
        eth=eth.type(torch.FloatTensor)
        eth=eth.to(self.device)
        out2=torch.cat((out2,eth),1)
#         print(eth.shape)
        
        ins=demo[:,2].to(self.device)
        ins=self.insEmbed(ins)
        ins=ins.type(torch.FloatTensor)
        ins=ins.to(self.device)
        out2=torch.cat((out2,ins),1)
#         print(ins.shape)
        
        age=demo[:,3].to(self.device)
        age=self.ageEmbed(age)
        age=age.type(torch.FloatTensor)
        age=age.to(self.device)
        out2=torch.cat((out2,age),1)
#         print(age.shape)
        
#         print("out",out1.shape)
        
        out2=out2.type(torch.FloatTensor)
        out2=out2.to(self.device)
        out2=self.statfc(out2)
        out2=self.statfc2(out2)
#         print("fcout",out1.shape)
        
        h_0, c_0 = self.init_hidden()
        h_0, c_0 = h_0.to(self.device), c_0.to(self.device)
        
        _, (code_h_n, code_c_n)=self.rnn(out1, (h_0, c_0))
        code_h_n=code_h_n[-1,:,:]
        code_h_n=code_h_n.squeeze()
#         print("rnnout",code_h_n.shape)
        
        out1=torch.cat((code_h_n,out2),1)
        out1 = self.fc1(out1)
        out1 = self.fc2(out1)
        #print("out1",out1.shape)
        
        sig = nn.Sigmoid()
        sigout1=sig(out1)
        #print("sig out",sigout1[16])
        #print("sig out",sigout1)
        #print(out1[0])
        #print("hi")
        
        return sigout1,out1
    
    def init_hidden(self):
        # initialize the hidden state and the cell state to zeros
        h=torch.zeros(self.rnnLayers,self.batch_size, self.rnn_size)
        c=torch.zeros(self.rnnLayers,self.batch_size, self.rnn_size)

#         if self.hparams.on_gpu:
#             hidden_a = hidden_a.cuda()
#             hidden_b = hidden_b.cuda()

        h = Variable(h)
        c = Variable(c)

        return (h, c)    
            
class StatEmbed(nn.Module):
    def __init__(self,device,code_vocab_size,embed_size,latent_size):             
        super(StatEmbed, self).__init__()
        self.embed_size=embed_size
        self.latent_size=latent_size
        self.code_vocab_size=code_vocab_size
        
        self.device=device
        
        self.build()
    
    def build(self):
        self.codeEmbed=nn.Embedding(self.code_vocab_size,self.embed_size)
        self.fc=nn.Linear(self.embed_size*self.code_vocab_size, self.latent_size, True)
        
    def forward(self, code):
        ids = torch.arange(0, code.shape[1])
        ids=ids.type(torch.LongTensor)

#         print(ids.shape)
        codeEmbedded=self.codeEmbed(ids.to(self.device))
        #print(codeEmbedded.shape)
#         print(codeEmbedded)

        codeEmbedded=codeEmbedded.unsqueeze(0)
        codeEmbedded=codeEmbedded.repeat(code.shape[0],1,1)

        codeEmbedded=codeEmbedded.type(torch.FloatTensor)
        #print(codeEmbedded.shape)

        #print(code.shape)
        code=code.unsqueeze(2)
        code=code.type(torch.FloatTensor)
#         print(code[5,0:5,0:10])
        
        
        #print("[!] Shapes: ", code.shape ,codeEmbedded.shape)
        codeEmbedded=torch.mul(code,codeEmbedded)
        codeEmbedded=torch.reshape(codeEmbedded,(codeEmbedded.shape[0],-1))
        codeEmbedded=codeEmbedded.to(self.device)
#         print(codeEmbedded.shape)

        codeEmbedded=self.fc(codeEmbedded)
        #print("[HERE]: ", codeEmbedded.shape)
        
        return codeEmbedded
    
class CodeEmbed(nn.Module):
    def __init__(self,device,code_vocab_size,embed_size,latent_size):             
        super(CodeEmbed, self).__init__()
        self.embed_size=embed_size
        self.latent_size=latent_size
        self.code_vocab_size=code_vocab_size
        
        self.device=device
        
        self.build()
    
    def build(self):
        self.codeEmbed=nn.Embedding(self.code_vocab_size,self.embed_size)
        self.fc=nn.Linear(self.embed_size*self.code_vocab_size, self.latent_size, True)
        
    def forward(self, code):
        ids = torch.arange(0, code.shape[2])
        ids=ids.type(torch.LongTensor)

#         print(ids.shape)
        codeEmbedded=self.codeEmbed(ids.to(self.device))
#         print(codeEmbedded.shape)
#         print(codeEmbedded)

        codeEmbedded=codeEmbedded.unsqueeze(0)
        codeEmbedded=codeEmbedded.unsqueeze(0)
        codeEmbedded=codeEmbedded.repeat(code.shape[0],code.shape[1],1,1)

        codeEmbedded=codeEmbedded.type(torch.FloatTensor)
#         print(codeEmbedded.shape)

        code=code.unsqueeze(3)
        code=code.repeat(1,1,1,codeEmbedded.shape[3])
        code=code.type(torch.FloatTensor)
#         print(code[5,0:5,0:10])
        
        

        codeEmbedded=torch.mul(code,codeEmbedded)
        codeEmbedded=torch.reshape(codeEmbedded,(codeEmbedded.shape[0],codeEmbedded.shape[1],-1))
        codeEmbedded=codeEmbedded.to(self.device)
#         print(codeEmbedded.shape)

        codeEmbedded=self.fc(codeEmbedded)
#         print(codeEmbedded.shape)
        
        return codeEmbedded
        

class ValEmbed(nn.Module):
    def __init__(self,device,code_vocab_size,embed_size,latent_size):             
        super(ValEmbed, self).__init__()
        self.embed_size=embed_size
        self.latent_size=latent_size
        self.code_vocab_size=code_vocab_size
        
        self.device=device
        
        self.build()
    
    def build(self):
#         self.codeEmbed=nn.Embedding(self.code_vocab_size,self.embed_size)
#         self.fc=nn.Linear((self.embed_size+1)*self.code_vocab_size, self.latent_size, True)
        self.codeEmbed=nn.BatchNorm1d(self.code_vocab_size)
        self.fc=nn.Linear(self.code_vocab_size, self.latent_size, True)
        
    def forward(self, code):
        #print("code",code.shape)
        code=code.permute(0,2,1)
        
#         ids=torch.range(0,code.shape[2]-1)
#         ids=ids.type(torch.LongTensor)

#         print(ids.shape)
#         codeEmbedded=self.codeEmbed(ids.to(self.device))
        code=code.type(torch.FloatTensor)
        code=code.to(self.device)
        codeEmbedded=self.codeEmbed(code)
#         print(codeEmbedded.shape)
#         print(codeEmbedded)

#         codeEmbedded=codeEmbedded.unsqueeze(0)
#         codeEmbedded=codeEmbedded.unsqueeze(0)
#         codeEmbedded=codeEmbedded.repeat(code.shape[0],code.shape[1],1,1)

        codeEmbedded=codeEmbedded.type(torch.FloatTensor)
#         print(codeEmbedded.shape)

#         code=code.unsqueeze(3)
#         code=code.type(torch.FloatTensor)
#         print(code[5,0:5,0:10])

#         print(codeEmbedded.shape)
#         codeEmbedded=torch.cat((code,codeEmbedded),3)
#         codeEmbedded=torch.reshape(codeEmbedded,(codeEmbedded.shape[0],codeEmbedded.shape[1],-1))
#         codeEmbedded=codeEmbedded.to(self.device)
#         print(codeEmbedded.shape)
        codeEmbedded=codeEmbedded.permute(0,2,1)
#         print(codeEmbedded.shape)
        codeEmbedded=codeEmbedded.to(self.device)
        codeEmbedded=self.fc(codeEmbedded)
#         print(codeEmbedded.shape)
        
        return codeEmbedded    

class LSTMAttn(nn.Module):
    def __init__(self,device,cond_vocab_size,cond_seq_len,proc_vocab_size,proc_seq_len,med_vocab_size,med_seq_len,out_vocab_size,out_seq_len,chart_vocab_size,chart_seq_len,lab_vocab_size,lab_seq_len,eth_vocab_size,gender_vocab_size,age_vocab_size,med_signal,lab_signal,embed_size,rnn_size,batch_size):
        super(LSTMAttn, self).__init__()
        self.embed_size=embed_size
        self.rnn_size=rnn_size
        self.eth_vocab_size=eth_vocab_size
        self.gender_vocab_size=gender_vocab_size
        self.age_vocab_size=age_vocab_size
        self.cond_vocab_size=cond_vocab_size
        self.cond_seq_len=cond_seq_len
        self.proc_vocab_size=proc_vocab_size
        self.proc_seq_len=proc_seq_len
        self.med_vocab_size=med_vocab_size
        self.med_seq_len=med_seq_len
        self.out_vocab_size=out_vocab_size
        self.out_seq_len=out_seq_len
        self.chart_vocab_size=chart_vocab_size
        self.chart_seq_len=chart_seq_len
        self.lab_vocab_size=lab_vocab_size
        self.lab_seq_len=lab_seq_len
        if self.chart_seq_len>500:
            self.chart_seq_len=500
        self.batch_size=batch_size
        self.padding_idx = 0
        self.modalities=0
        self.device=device
        self.med_signal,self.lab_signal=med_signal,lab_signal
        self.build()
        
    def build(self):
        
        if self.med_vocab_size:
            self.med=CodeAttn(self.device,self.embed_size,self.rnn_size,self.med_vocab_size,self.med_seq_len,self.batch_size,self.med_signal,False)
            self.modalities=self.modalities+1
                
        if self.proc_vocab_size:
            self.proc=CodeAttn(self.device,self.embed_size,self.rnn_size,self.proc_vocab_size,self.proc_seq_len,self.batch_size,True,False)
            self.modalities=self.modalities+1
        if self.out_vocab_size:
            self.out=CodeAttn(self.device,self.embed_size,self.rnn_size,self.out_vocab_size,self.out_seq_len,self.batch_size,True,False)
            self.modalities=self.modalities+1
        if self.chart_vocab_size:
            self.chart=CodeAttn(self.device,self.embed_size,self.rnn_size,self.chart_vocab_size,self.chart_seq_len,self.batch_size,self.lab_signal,True)
            self.modalities=self.modalities+1
        if self.lab_vocab_size:
            self.lab=CodeAttn(self.device,self.embed_size,self.rnn_size,self.lab_vocab_size,self.lab_seq_len,self.batch_size,self.lab_signal,False)
            self.modalities=self.modalities+1

        
        self.condEmbed=nn.Embedding(self.cond_vocab_size,self.embed_size,self.padding_idx) 
        self.condfc=nn.Linear((self.embed_size*self.cond_seq_len),self.rnn_size, False)
        
        self.ethEmbed=nn.Embedding(self.eth_vocab_size,self.embed_size,self.padding_idx) 
        self.genderEmbed=nn.Embedding(self.gender_vocab_size,self.embed_size,self.padding_idx) 
        self.ageEmbed=nn.Embedding(self.age_vocab_size,self.embed_size,self.padding_idx) 
        self.demo_fc=nn.Linear(self.embed_size*3, self.rnn_size, False)
        
        #self.fc=nn.Linear((self.embed_size*self.cond_seq_len)+3*self.rnn_size, 1, False)
        self.fc1=nn.Linear(int(self.rnn_size*(self.modalities+2)), int((self.rnn_size*(self.modalities+2))/2), False)
        self.fc2=nn.Linear(int((self.rnn_size*(self.modalities+2))/2), int((self.rnn_size*(self.modalities+2))/4), False)
        self.fc3=nn.Linear(int((self.rnn_size*(self.modalities+2))/4), 1, False)
        
        #self.sig = nn.Sigmoid()
        
    def forward(self,X):        
        meds,chart,out,proc,lab,conds,demo=X[0],X[1],X[2],X[3],X[4],X[5],X[6]    
        
        out1 = torch.zeros(size=(1,0))
        
        if len(meds[0]):
            med_h_n = self.med(meds)  
            med_h_n=med_h_n.view(med_h_n.shape[0],-1)
            #print("med_h_n",med_h_n.shape)
            out1=med_h_n
            #print(out1.shape)
            #print(out1.nelement())
        if len(procs):
            proc_h_n = self.proc(procs)  
            proc_h_n=proc_h_n.view(proc_h_n.shape[0],-1)
            #print("proc_h_n",proc_h_n.shape)
            if out1.nelement():
                out1=torch.cat((out1,proc_h_n),1)
            else:
                out1=proc_h_n
        if len(labs[0]):
            lab_h_n = self.lab(labs)  
            lab_h_n=lab_h_n.view(lab_h_n.shape[0],-1)
            #print("lab_h_n",lab_h_n.shape)
            if out1.nelement():
                out1=torch.cat((out1,lab_h_n),1)
            else:
                out1=lab_h_n
        if len(outs):
            out_h_n = self.out(outs)  
            out_h_n=out_h_n.view(out_h_n.shape[0],-1)
            if out1.nelement():
                out1=torch.cat((out1,out_h_n),1)
            else:
                out1=out_h_n
        if len(charts[0]):
            chart_h_n = self.chart(charts)  
            chart_h_n=out_h_n.view(chart_h_n.shape[0],-1)
            if out1.nelement:
                out1=torch.cat((out1,chart_h_n),1)
            else:
                out1=chart_h_n
        
        conds=conds.to(self.device)
        conds=self.condEmbed(conds)
        #print(conds.shape)
        conds=conds.view(conds.shape[0],-1)
        conds=self.condfc(conds)
        #print(conds.shape)
        #print("cond_pool_ob",cond_pool_ob.shape)
        #out1=torch.cat((cond_pool,cond_pool_ob),1)
        #out1=cond_pool
        eth=demo[0].to(self.device)
        eth=self.ethEmbed(eth)
        
        gender=demo[1].to(self.device)
        gender=self.genderEmbed(gender)
        
        age=demo[2].to(self.device)
        age=self.ageEmbed(age)
        
        demog=torch.cat((eth,gender),1)
        demog=torch.cat((demog,age),1)
        #print("demog",demog.shape)
        demog=self.demo_fc(demog)
        
        out1=torch.cat((out1,conds),1)
        out1=torch.cat((out1,demog),1)
        #print("out1",out1.shape)
        out1 = self.fc1(out1)
        out1 = self.fc2(out1)
        out1 = self.fc3(out1)
        #print("out1",out1.shape)
        
        sig = nn.Sigmoid()
        sigout1=sig(out1)
        #print("sig out",sigout1[16])
        #print("sig out",sigout1)
        #print(out1[0])
        #print("hi")
        
        return sigout1,out1
        
            


# In[ ]:


class CodeAttn(nn.Module):
    def __init__(self,device,embed_size,rnn_size,code_vocab_size,code_seq_len,batch_size,signal,lab):           
        super(CodeAttn, self).__init__()
        self.embed_size=embed_size
        self.rnn_size=rnn_size
        self.code_vocab_size=code_vocab_size
        self.code_seq_len=code_seq_len
        self.batch_size=batch_size
        self.padding_idx = 0
        self.device=device
        self.signal=signal
        self.build()
        self.lab_sig=lab
    
    def build(self):
        
        self.codeEmbed=nn.Embedding(self.code_vocab_size,self.embed_size,self.padding_idx)
        if self.signal: 
            self.codeRnn = nn.LSTM(input_size=int(self.embed_size*self.code_seq_len),hidden_size=self.rnn_size,num_layers = 2,dropout=0.2,batch_first=True)
            #self.codeRnn = nn.LSTM(input_size=self.embed_size,hidden_size=self.rnn_size,num_layers = 2,dropout=0.2,batch_first=True)
        else:
            self.codeRnn = nn.LSTM(input_size=int((self.embed_size+1)*self.code_seq_len),hidden_size=self.rnn_size,num_layers = 2,dropout=0.2,batch_first=True)
            #self.codeRnn = nn.LSTM(input_size=self.embed_size+1,hidden_size=self.rnn_size,num_layers = 2,dropout=0.2,batch_first=True)

        self.code_fc=nn.Linear(self.rnn_size, 1, False)
        #self.dropout1 = nn.Dropout(0.2)
        
    def forward(self, code):
        #print(conds.shape)

        h_0, c_0 = self.init_hidden()
        h_0, c_0, code = h_0.to(self.device), c_0.to(self.device),code.to(self.device)

        #Embedd all sequences
        #print(code.shape)
        #print(code[0,:,:])

        if code.shape[0]==2:
            dat=code[1]
            code=code[0]
            if self.lab_sig:
                if code.shape[1]>500:
                    code=code[:,0:500,:]
                    dat=dat[:,0:500,:]
            codeEmbedded=self.codeEmbed(code)
            #code=torch.transpose(code,1,2)
            #code=torch.reshape(code,(code.shape[0],code.shape[1],-1))
            #code=torch.sum(code,1)
            #print(code.shape)
            #print(self.signal)
            if not self.signal:
                if self.lab_sig:
                    test=torch.max(code,2)
                    test=test.values
                    test=test.unsqueeze(2)
                    code=torch.zeros(code.shape[0],code.shape[1],code.shape[2])
                    code=code.type(torch.FloatTensor)
                    code=code.to(self.device)
                    test=test.type(torch.FloatTensor)
                    test=test.to(self.device)
                    code=torch.add(code,test)
                    code=code.type(torch.LongTensor)
                    code=code.to(self.device)
                    codeEmbedded=self.codeEmbed(code)
                dat=dat.unsqueeze(3)
                #print(dat.shape)
                dat=dat.type(torch.FloatTensor)
                dat=dat.to(self.device)
                codeEmbedded=torch.cat((codeEmbedded,dat),3)
            code=torch.transpose(codeEmbedded,1,2)
            code=torch.reshape(code,(code.shape[0],code.shape[1],-1))
            #code=torch.sum(codeEmbedded,1)
            
                #print(code.shape)
        else:
            code=self.codeEmbed(code)
            code=torch.transpose(code,1,2)
            code=torch.reshape(code,(code.shape[0],code.shape[1],-1))
            #code=torch.sum(code,1)
        #print(code.shape)
        #code=torch.transpose(code,1,2)
        #print(code[0])
        #print(dat[0])
        #print(code[0,:,:])

        h_0, c_0, code = h_0.to(self.device), c_0.to(self.device),code.to(self.device)
        #print(code.shape)
        #code=code.type(torch.FloatTensor)
#        code_time=code_time.type(torch.FloatTensor)
        #h_0, c_0, code = h_0.to(self.device), c_0.to(self.device),code.to(self.device)

#        code=torch.cat((code,code_time),dim=2)
            
        #Run through LSTM
        code_output, (code_h_n, code_c_n)=self.codeRnn(code, (h_0, c_0))
        #print("code_output",code_output.shape)
        
        code_softmax=self.code_fc(code_output)
        #code_output=self.dropout1(code_output) 
        #print("softmax",code_softmax.shape)
        code_softmax=F.softmax(code_softmax)
        #print("softmax",code_softmax.shape)
        code_softmax=torch.sum(torch.mul(code_output,code_softmax),dim=1)
        #print("softmax",code_softmax.shape)
        #print("========================")
        
        return code_softmax
    
    
    def init_hidden(self):
        # initialize the hidden state and the cell state to zeros
        h=torch.zeros(2,self.batch_size, self.rnn_size)
        c=torch.zeros(2,self.batch_size, self.rnn_size)

#         if self.hparams.on_gpu:
#             hidden_a = hidden_a.cuda()
#             hidden_b = hidden_b.cuda()

        h = Variable(h)
        c = Variable(c)

        return (h, c)    
    

            
class CNNBase(nn.Module):
    def __init__(self,device,cond_vocab_size,proc_vocab_size,med_vocab_size,out_vocab_size,chart_vocab_size,lab_vocab_size,eth_vocab_size,gender_vocab_size,age_vocab_size,ins_vocab_size,modalities,embed_size,rnn_size,latent_size, batch_size):
        super(CNNBase, self).__init__()
        self.embed_size=embed_size
        self.latent_size=latent_size
        self.rnn_size=rnn_size
        self.cond_vocab_size=cond_vocab_size
        self.proc_vocab_size=proc_vocab_size
        self.med_vocab_size=med_vocab_size
        self.out_vocab_size=out_vocab_size
        self.chart_vocab_size=chart_vocab_size
        self.lab_vocab_size=lab_vocab_size
        
        self.eth_vocab_size=eth_vocab_size
        self.gender_vocab_size=gender_vocab_size
        self.age_vocab_size=age_vocab_size
        self.ins_vocab_size=ins_vocab_size

        self.batch_size=batch_size
        self.padding_idx = 0
        self.device=device
        self.modalities=modalities
        self.build()

        print("lab_vocab_size",  lab_vocab_size)
        
    def build(self):
            
        if self.med_vocab_size:
            self.med=ValEmbed(self.device,self.med_vocab_size,self.embed_size,self.latent_size)                
        if self.proc_vocab_size:
            self.proc=CodeEmbed(self.device,self.proc_vocab_size,self.embed_size,self.latent_size)
        if self.out_vocab_size:
            self.out=CodeEmbed(self.device,self.out_vocab_size,self.embed_size,self.latent_size)
        if self.chart_vocab_size:
            self.chart=ValEmbed(self.device,self.chart_vocab_size,self.embed_size,self.latent_size)
        if self.lab_vocab_size:
            self.lab=ValEmbed(self.device,self.lab_vocab_size,self.embed_size,self.latent_size)
        
        if self.cond_vocab_size:
            self.cond=StatEmbed(self.device,self.cond_vocab_size,self.embed_size,self.latent_size)
        
        self.ethEmbed=nn.Embedding(self.eth_vocab_size,self.latent_size,self.padding_idx) 
        self.genderEmbed=nn.Embedding(self.gender_vocab_size,self.latent_size,self.padding_idx) 
        self.ageEmbed=nn.Embedding(self.age_vocab_size,self.latent_size,self.padding_idx) 
        self.insEmbed=nn.Embedding(self.ins_vocab_size,self.latent_size,self.padding_idx) 
       
        
        self.embedfc=nn.Linear((self.latent_size*(self.modalities+4)), self.latent_size, True)
        
        self.conv1 = nn.Conv1d(self.latent_size,self.rnn_size, kernel_size = 10, stride = 1, padding =0)   
        self.bn1 = nn.BatchNorm1d(self.rnn_size)
        self.maxpool1 = nn.AdaptiveMaxPool1d(1, True)
        
        self.fc1=nn.Linear(self.rnn_size, int((self.rnn_size)/2), True)
        self.fc2=nn.Linear(int((self.rnn_size)/2), 1, True)
        
        
        
        
        
        #self.sig = nn.Sigmoid()
        
    def forward(self,meds,chart,out,proc,lab,conds,demo):         
        #meds,chart,out,proc,lab,conds,demo=X[0],X[1],X[2],X[3],X[4],X[5],X[6]   
        
        out1=torch.zeros(size=(0,0))
        
        if meds.shape[0]:
            medEmbedded=self.med(meds)
            
            if out1.nelement():
                out1=torch.cat((out1,medEmbedded),2)
            else:
                out1=medEmbedded
            #print(out1.shape)
            #print(out1.nelement())
        if proc.shape[0]:
            procEmbedded=self.proc(proc)
            
            if out1.nelement():
                out1=torch.cat((out1,procEmbedded),2)
            else:
                out1=procEmbedded
        if lab.shape[1]:
            labEmbedded=self.lab(lab)
            
            if out1.nelement():
                out1=torch.cat((out1,labEmbedded),2)
            else:
                out1=labEmbedded
        if out.shape[0]:
            outEmbedded=self.out(out)
            
            if out1.nelement():
                out1=torch.cat((out1,outEmbedded),2)
            else:
                out1=outEmbedded

            
        if chart.shape[0]:
            chartEmbed=self.chart(chart)
#             print("chartEmbed",chartEmbed.shape)
#             print(chartEmbed[5,:,0:10])
            if out1.nelement():
                out1=torch.cat((out1,chartEmbed),2)
            else:
                out1=chartEmbed
        
#         print("out1",out1.shape)
        conds=conds.to(self.device)
        condEmbed=self.cond(conds)
        condEmbed=condEmbed.unsqueeze(1)
        condEmbed=condEmbed.repeat(1,out1.shape[1],1)
        condEmbed=condEmbed.type(torch.FloatTensor)
        condEmbed=condEmbed.to(self.device)
#         print("cond",condEmbed.shape)
        out1=torch.cat((out1,condEmbed),2)
#         print("cond",condEmbed.shape)
        
        gender=demo[:,0].to(self.device)
        gender=self.genderEmbed(gender)
        gender=gender.unsqueeze(1)
        gender=gender.repeat(1,out1.shape[1],1)
        gender=gender.type(torch.FloatTensor)
        gender=gender.to(self.device)
        out1=torch.cat((out1,gender),2)
#         print(gender.shape)
        
        eth=demo[:,1].to(self.device)
        eth=self.ethEmbed(eth)
        eth=eth.unsqueeze(1)
        eth=eth.repeat(1,out1.shape[1],1)
        eth=eth.type(torch.FloatTensor)
        eth=eth.to(self.device)
        out1=torch.cat((out1,eth),2)
#         print(eth.shape)
        
        ins=demo[:,2].to(self.device)
        ins=self.insEmbed(ins)
        ins=ins.unsqueeze(1)
        ins=ins.repeat(1,out1.shape[1],1)
        ins=ins.type(torch.FloatTensor)
        ins=ins.to(self.device)
        out1=torch.cat((out1,ins),2)
#         print(ins.shape)
        
        age=demo[:,3].to(self.device)
        age=self.ageEmbed(age)
        age=age.unsqueeze(1)
        age=age.repeat(1,out1.shape[1],1)
        age=age.type(torch.FloatTensor)
        age=age.to(self.device)
        out1=torch.cat((out1,age),2)
#         print(age.shape)
        
#         print("out",out1.shape)
        
        out1=out1.type(torch.FloatTensor)
        out1=out1.to(self.device)
        out1=self.embedfc(out1)
#         print("fcout",out1.shape)
           
        #Run through CNN
        out1=out1.permute(0,2,1)
        code_output = self.conv1(out1)
#         print("output",code_output.shape)
        code_output = self.bn1(code_output)
#         print("output",code_output.shape)
        
        code_pool, code_indices = self.maxpool1(code_output)
#         print("output",code_pool.shape)
        
        
        code_pool = torch.squeeze(code_pool)
        code_pool=code_pool.view(code_pool.shape[0],-1)
#         print("output",code_pool.shape)
        
        out1 = self.fc1(code_pool)
        out1 = self.fc2(out1)
        #print("out1",out1.shape)
        
        sig = nn.Sigmoid()
        sigout1=sig(out1)
        #print("sig out",sigout1[16])
        #print("sig out",sigout1)
        #print(out1[0])
        #print("hi")
        
        return sigout1,out1
    
    
    
    
    
class CNNBaseH(nn.Module):
    def __init__(self,device,cond_vocab_size,proc_vocab_size,med_vocab_size,out_vocab_size,chart_vocab_size,lab_vocab_size,eth_vocab_size,gender_vocab_size,age_vocab_size,ins_vocab_size,modalities,embed_size,rnn_size,latent_size,batch_size):
        super(CNNBaseH, self).__init__()
        self.embed_size=embed_size
        self.latent_size=latent_size
        self.rnn_size=rnn_size
        self.cond_vocab_size=cond_vocab_size
        self.proc_vocab_size=proc_vocab_size
        self.med_vocab_size=med_vocab_size
        self.out_vocab_size=out_vocab_size
        self.chart_vocab_size=chart_vocab_size
        self.lab_vocab_size=lab_vocab_size
        
        self.eth_vocab_size=eth_vocab_size
        self.gender_vocab_size=gender_vocab_size
        self.age_vocab_size=age_vocab_size
        self.ins_vocab_size=ins_vocab_size

        self.batch_size=batch_size
        self.padding_idx = 0
        self.device=device
        self.modalities=modalities
        self.build()
        
    def build(self):
            
        if self.med_vocab_size:
            self.med=ValEmbed(self.device,self.med_vocab_size,self.embed_size,self.latent_size)                
        if self.proc_vocab_size:
            self.proc=CodeEmbed(self.device,self.proc_vocab_size,self.embed_size,self.latent_size)
        if self.out_vocab_size:
            self.out=CodeEmbed(self.device,self.out_vocab_size,self.embed_size,self.latent_size)
        if self.chart_vocab_size:
            self.chart=ValEmbed(self.device,self.chart_vocab_size,self.embed_size,self.latent_size)
        if self.lab_vocab_size:
            self.lab=ValEmbed(self.device,self.lab_vocab_size,self.embed_size,self.latent_size)
        
        if self.cond_vocab_size:
            self.cond=StatEmbed(self.device,self.cond_vocab_size,self.embed_size,self.latent_size)
        
        self.ethEmbed=nn.Embedding(self.eth_vocab_size,self.latent_size,self.padding_idx) 
        self.genderEmbed=nn.Embedding(self.gender_vocab_size,self.latent_size,self.padding_idx) 
        self.ageEmbed=nn.Embedding(self.age_vocab_size,self.latent_size,self.padding_idx) 
        self.insEmbed=nn.Embedding(self.ins_vocab_size,self.latent_size,self.padding_idx) 
       
        
        self.embedfc=nn.Linear((self.latent_size*(self.modalities-1)), self.latent_size, True)
        self.statfc=nn.Linear(int(self.latent_size*5), self.latent_size, True)
        self.statfc2=nn.Linear(self.latent_size, self.rnn_size, True)
        
        self.conv1 = nn.Conv1d(self.latent_size,self.rnn_size, kernel_size = 10, stride = 1, padding =0)   
        self.bn1 = nn.BatchNorm1d(self.rnn_size)
        self.maxpool1 = nn.AdaptiveMaxPool1d(1, True)
        
        self.fc1=nn.Linear(self.rnn_size*2, self.rnn_size, True)
        self.fc2=nn.Linear(self.rnn_size, 1, False)
        
        #self.sig = nn.Sigmoid()
        
    def forward(self,meds,chart,out,proc,lab,conds,demo):   
        #meds,chart,out,proc,lab,conds,demo=X[0],X[1],X[2],X[3],X[4],X[5],X[6]
        
        out1=torch.zeros(size=(0,0))
        
        if meds.shape[0]:
            medEmbedded=self.med(meds)
            
            if out1.nelement():
                out1=torch.cat((out1,medEmbedded),2)
            else:
                out1=medEmbedded
            #print(out1.shape)
            #print(out1.nelement())
        if proc.shape[0]:
            procEmbedded=self.proc(proc)
            
            if out1.nelement():
                out1=torch.cat((out1,procEmbedded),2)
            else:
                out1=procEmbedded
        if lab.shape[1]:
            labEmbedded=self.lab(lab)
            
            if out1.nelement():
                out1=torch.cat((out1,labEmbedded),2)
            else:
                out1=labEmbedded
        if out.shape[0]:
            outEmbedded=self.out(out)
            
            if out1.nelement():
                out1=torch.cat((out1,outEmbedded),2)
            else:
                out1=outEmbedded

            
        if chart.shape[0]:
            chartEmbed=self.chart(chart)
#             print("chartEmbed",chartEmbed.shape)
#             print(chartEmbed[5,:,0:10])
            if out1.nelement():
                out1=torch.cat((out1,chartEmbed),2)
            else:
                out1=chartEmbed

        
        out1=out1.type(torch.FloatTensor)
        out1=out1.to(self.device)
        out1=self.embedfc(out1)
        
        out2=torch.zeros(size=(0,0))
        conds=conds.to(self.device)
        condEmbed=self.cond(conds)
        
        
        condEmbed=condEmbed.type(torch.FloatTensor)
        condEmbed=condEmbed.to(self.device)
        out2=condEmbed
        #print("cond",condEmbed.shape)
        
        gender=demo[:,0].to(self.device)
        gender=self.genderEmbed(gender)
        gender=gender.type(torch.FloatTensor)
        gender=gender.to(self.device)
        out2=torch.cat((out2,gender),1)
#         print(gender.shape)
        
        eth=demo[:,1].to(self.device)
        eth=self.ethEmbed(eth)
        eth=eth.type(torch.FloatTensor)
        eth=eth.to(self.device)
        out2=torch.cat((out2,eth),1)
#         print(eth.shape)
        
        ins=demo[:,2].to(self.device)
        ins=self.insEmbed(ins)
        ins=ins.type(torch.FloatTensor)
        ins=ins.to(self.device)
        out2=torch.cat((out2,ins),1)
#         print(ins.shape)
        
        age=demo[:,3].to(self.device)
        age=self.ageEmbed(age)
        age=age.type(torch.FloatTensor)
        age=age.to(self.device)
        out2=torch.cat((out2,age),1)
#         print(age.shape)
        
#         print("out",out1.shape)
        
        out2=out2.type(torch.FloatTensor)
        out2=out2.to(self.device)
        out2=self.statfc(out2)
        out2=self.statfc2(out2)
#         print("fcout",out1.shape)
        
        out1=out1.permute(0,2,1)
        code_output = self.conv1(out1)
        #print("output",cond_output.shape)
        code_output = self.bn1(code_output)
        #print("output",code_output.shape)
        
        code_pool, code_indices = self.maxpool1(code_output)
        #print("output",code_pool.shape)
        
        
        code_pool = torch.squeeze(code_pool)
        code_pool=code_pool.view(code_pool.shape[0],-1)
#         print("rnnout",code_h_n.shape)
        
        out1=torch.cat((code_pool,out2),1)
        out1 = self.fc1(out1)
        out1 = self.fc2(out1)
        #print("out1",out1.shape)
        
        sig = nn.Sigmoid()
        sigout1=sig(out1)
        #print("sig out",sigout1[16])
        #print("sig out",sigout1)
        #print(out1[0])
        #print("hi")
        
        print("sigout1 shape: ", sigout1.shape)
        print("out shape: ", out1.shape)
        return sigout1,out1

class GNNBase(nn.Module):
    def __init__(self, device, cond_vocab_size, proc_vocab_size, med_vocab_size, out_vocab_size,
                 chart_vocab_size, lab_vocab_size, eth_vocab_size, gender_vocab_size, age_vocab_size,
                 ins_vocab_size, modalities, embed_size, gnn_size, batch_size):
        super(GNNBase, self).__init__()

        self.embed_size=embed_size
        self.cond_vocab_size=cond_vocab_size
        self.proc_vocab_size=proc_vocab_size
        self.med_vocab_size=med_vocab_size
        self.out_vocab_size=out_vocab_size
        self.chart_vocab_size=chart_vocab_size
        self.lab_vocab_size=lab_vocab_size
        
        self.eth_vocab_size=eth_vocab_size
        self.gender_vocab_size=gender_vocab_size
        self.age_vocab_size=age_vocab_size
        self.ins_vocab_size=ins_vocab_size
        
        self.latent_size = embed_size  # Matching latent size to LSTM model for compatibility
        self.gnn_size = gnn_size
        self.batch_size = batch_size
        self.device = device
        self.modalities = modalities

        # Embedding layers
        self.build_embeddings(cond_vocab_size, proc_vocab_size, med_vocab_size, out_vocab_size,
                              chart_vocab_size, lab_vocab_size, eth_vocab_size, gender_vocab_size,
                              age_vocab_size, ins_vocab_size)

        # Graph convolution and attention layers
        self.gcn1 = GCNConv(self.latent_size, self.gnn_size)
        self.gcn2 = GCNConv(self.gnn_size, self.gnn_size)
        self.gat = GATConv(self.gnn_size, self.gnn_size, heads=4, concat=False)

        # Fully connected layers
        self.fc1 = nn.Linear(self.gnn_size, int(self.gnn_size / 2))
        self.fc2 = nn.Linear(int(self.gnn_size / 2), 1)
        self.sigmoid = nn.Sigmoid()

    def build_embeddings(self, cond_vocab_size, proc_vocab_size, med_vocab_size, out_vocab_size,
                     chart_vocab_size, lab_vocab_size, eth_vocab_size, gender_vocab_size,
                     age_vocab_size, ins_vocab_size):
        # Check for zero vocab sizes and handle accordingly
        self.ethEmbed = nn.Embedding(eth_vocab_size, self.latent_size) if eth_vocab_size > 0 else None
        self.genderEmbed = nn.Embedding(gender_vocab_size, self.latent_size) if gender_vocab_size > 0 else None
        self.ageEmbed = nn.Embedding(age_vocab_size, self.latent_size)
        self.insEmbed = nn.Embedding(ins_vocab_size, self.latent_size)
        
        # Create embeddings for other fields
        self.cond = nn.Embedding(cond_vocab_size, self.latent_size)
        self.proc = nn.Embedding(proc_vocab_size, self.latent_size)
        self.med = nn.Embedding(med_vocab_size, self.latent_size)
        self.out = nn.Embedding(out_vocab_size, self.latent_size)
        self.chart = nn.Embedding(chart_vocab_size, self.latent_size)
        
        # Handle potential issue with lab field
        if lab_vocab_size > 0:
            self.lab = nn.Embedding(lab_vocab_size, self.latent_size)
        else:
            self.lab = None  # Handle missing lab data appropriately

    def entropy_attention(self, x, edge_index):
        """
        Entropy-based attention over graph features.
        Here `x` is the node feature matrix, and `edge_index` defines connections.
        """
        # Entropy-based attention scores (simplified)
        attention_weights = torch.exp(-torch.sum(x * torch.log(x + 1e-9), dim=1))
        attention_weights = attention_weights / torch.sum(attention_weights)  # Normalize
        
        # Apply attention scores to node features
        x = x * attention_weights.unsqueeze(1)
        return x
    
    def create_edge_index(self):
        # Number of nodes (seq_len * batch_size)
        num_nodes = self.seq_len * self.batch_size
        
        # Initialize an empty list to store the edges
        edges = []

        # Create edges between all nodes (pair every node with every other node)
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:  # No self-connections
                    edges.append([i, j])

        # Convert the list of edges to a tensor
        edge_index = torch.tensor(edges, dtype=torch.long).t()  # Shape: [2, num_edges]
        
        # Return the edge index
        return edge_index

    def create_edge_index_v2(self):
        # Number of nodes (seq_len * batch_size)
        num_nodes = self.seq_len * self.batch_size
        
        # Initialize an empty list to store the edges
        edges = []

        # Create edges between all nodes (pair every node with every other node)
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:  # No self-connections
                    edges.append([i, j])

        # Convert the list of edges to a tensor
        edge_index = torch.tensor(edges, dtype=torch.long).t()  # Shape: [2, num_edges]
        
        # Return the edge index
        return edge_index


    def forward(self, meds, chart, out, proc, lab, conds, demo, stat=None, edge_index=None):
        device = self.device  # Define the device once for easier reference

        self.seq_len = meds.size(1)

        edge_index = self.create_edge_index_v2()  # Generate edge_index based on the data

        # Ensure valid indices before embedding
        meds = torch.clamp(meds, 0, self.med_vocab_size - 1)
        chart = torch.clamp(chart, 0, self.chart_vocab_size - 1)
        out = torch.clamp(out, 0, self.out_vocab_size - 1)
        proc = torch.clamp(proc, 0, self.proc_vocab_size - 1)
        lab = torch.clamp(lab, 0, self.lab_vocab_size - 1) if self.lab is not None else lab
        conds = torch.clamp(conds, 0, self.cond_vocab_size - 1)

        # Embedding each input (handle empty tensors with conditional checks)
        med_embed = self.med(meds) if meds.numel() > 0 else torch.zeros(self.batch_size, self.seq_len, self.embed_size, device=self.device)
        chart_embed = self.chart(chart) if chart.numel() > 0 else torch.zeros(self.batch_size, self.seq_len, self.embed_size, device=self.device)
        out_embed = self.out(out) if out.numel() > 0 else torch.zeros(self.batch_size, self.seq_len, self.embed_size, device=self.device)
        proc_embed = self.proc(proc) if proc.numel() > 0 else torch.zeros(self.batch_size, self.seq_len, self.embed_size, device=self.device)
        lab_embed = self.lab(lab) if self.lab is not None and lab.numel() > 0 else torch.zeros(self.batch_size, self.seq_len, self.embed_size, device=self.device)
        cond_embed = self.cond(conds) if conds.numel() > 0 else torch.zeros(self.batch_size, self.seq_len, self.embed_size, device=self.device)

        # Demo embeddings (ensure demo tensor is on the correct device)
        demo = demo.to(device)
        gender_embed = self.genderEmbed(demo[:, 0].long()) if self.genderEmbed is not None else torch.zeros(self.batch_size, self.seq_len, self.embed_size, device=self.device)
        eth_embed = self.ethEmbed(demo[:, 1].long()) if self.ethEmbed is not None else torch.zeros(self.batch_size, self.seq_len, self.embed_size, device=self.device)
        ins_embed = self.insEmbed(demo[:, 2].long())
        age_embed = self.ageEmbed(demo[:, 3].long())

        # Reshaping the embeddings
        med_embed_reshaped = med_embed.permute(0, 1, 3, 2)  # Shape: [batch_size, 24, 52, 19]
        chart_embed_reshaped = chart_embed.permute(0, 1, 3, 2)  # Shape: [batch_size, 24, 52, 45]
        out_embed_reshaped = out_embed.permute(0, 1, 3, 2)  # Shape: [batch_size, 24, 52, 7]
        proc_embed_reshaped = proc_embed.permute(0, 1, 3, 2)  # Shape: [batch_size, 24, 52, 15]
        lab_embed_reshaped = lab_embed.unsqueeze(-1)  # Shape: [batch_size, 24, 52, 1]
        cond_embed_reshaped = cond_embed.unsqueeze(1).expand(-1, 24, -1, -1).permute(0, 1, 3, 2)  # Shape: [batch_size, 24, 52, 152]

        # Unsqueezing demo embeddings
        gender_embed = gender_embed.unsqueeze(1).repeat(1, 24, 1)
        eth_embed = eth_embed.unsqueeze(1).repeat(1, 24, 1)
        ins_embed = ins_embed.unsqueeze(1).repeat(1, 24, 1)
        age_embed = age_embed.unsqueeze(1).repeat(1, 24, 1)

        # Reshaping demo embeddings
        gender_embed_reshaped = gender_embed.unsqueeze(-1)
        eth_embed_reshaped = eth_embed.unsqueeze(-1)
        ins_embed_reshaped = ins_embed.unsqueeze(-1)
        age_embed_reshaped = age_embed.unsqueeze(-1)

        # Concatenate embeddings into one tensor
        x = torch.cat([med_embed_reshaped, chart_embed_reshaped, out_embed_reshaped, proc_embed_reshaped,
                    lab_embed_reshaped, cond_embed_reshaped, gender_embed_reshaped, eth_embed_reshaped, 
                    ins_embed_reshaped, age_embed_reshaped], dim=-1)
        
        # Debug: Check shape after concatenation
        print(f"Shape after concatenation: {x.shape}")

        # Mean across the second-to-last dimension
        x = torch.mean(x, dim=-2)

        # Debug: Check shape after mean operation
        print(f"Shape after mean operation: {x.shape}")

        # Reshape x for input to the linear layer
        x = x.view(self.batch_size * self.seq_len, -1).to(self.device)  # [3072, 243]

        # Debug: Check shape after mean operation
        #print(f"Shape after view operation: {x.shape}")

        # Linear transformation to latent space
        x = nn.Linear(x.size(-1), self.latent_size).to(device)(x)
        exit()

        # Debug: Check shape after mean operation
        #print(f"Shape after linear operation: {x.shape}")
        #print(f"Shape of edge index: {edge_index.shape}")
        
        # Apply GCN layers with entropy-based attention
        x = self.gcn1(x, edge_index.to(self.device))
        x = F.relu(x)
        x = self.entropy_attention(x, edge_index.to(self.device))
        x = self.gcn2(x, edge_index.to(self.device))
        #print(f"Shape after gcns: {x.shape}")     

        # Reshape it to [batch_size, seq_len, 64] for pooling
        x = x.view(self.batch_size, self.seq_len, -1)  # [batch_size, seq_len, 64]
        #print(f"Shape after view: {x.shape}")
        # Apply mean pooling across the sequence dimension (dim=1)
        x_pooled = x.mean(dim=1)  # Shape: [batch_size, 64] (mean across sequence length)
        #print(f"Shape after pooled: {x.shape}")
        # Now apply the fully connected layers
        x_fc = F.relu(self.fc1(x_pooled))  # Shape: [batch_size, fc1_out_features]
        out = self.fc2(x_fc)  # Shape: [batch_size, 1] (final output)
        sigout1 = self.sigmoid(out)  # Sigmoid for final output probability

        return sigout1, out


    def old_forward(self, meds, chart, out, proc, lab, conds, demo, stat=None, edge_index=None):
        device = self.device  # Define the device once for easier reference

        self.seq_len = meds.size(1)
        #print("Sequence length equals ", self.seq_len)

        edge_index = self.create_optimized_edge_index() # self.seq_len

        # Ensure valid indices before embedding
        meds = torch.clamp(meds, 0, self.med_vocab_size - 1)
        chart = torch.clamp(chart, 0, self.chart_vocab_size - 1)
        out = torch.clamp(out, 0, self.out_vocab_size - 1)
        proc = torch.clamp(proc, 0, self.proc_vocab_size - 1)
        lab = torch.clamp(lab, 0, self.lab_vocab_size - 1) if self.lab is not None else lab
        conds = torch.clamp(conds, 0, self.cond_vocab_size - 1)

        # Embedding each input (handle empty tensors with conditional checks)
        med_embed = self.med(meds) if meds.numel() > 0 else torch.zeros(self.batch_size, self.seq_len, self.embed_size, device=self.device)
        chart_embed = self.chart(chart) if chart.numel() > 0 else torch.zeros(self.batch_size, self.seq_len, self.embed_size, device=self.device)
        out_embed = self.out(out) if out.numel() > 0 else torch.zeros(self.batch_size, self.seq_len, self.embed_size, device=self.device)
        proc_embed = self.proc(proc) if proc.numel() > 0 else torch.zeros(self.batch_size, self.seq_len, self.embed_size, device=self.device)
        lab_embed = self.lab(lab) if self.lab is not None and lab.numel() > 0 else torch.zeros(self.batch_size, self.seq_len, self.embed_size, device=self.device)
        cond_embed = self.cond(conds) if conds.numel() > 0 else torch.zeros(self.batch_size, self.seq_len, self.embed_size, device=self.device)
        """
        print("Meds embed: ", med_embed.shape)
        print("Chart embed: ", chart_embed.shape)
        print("Out embed: ", out_embed.shape)
        print("Proc embed: ", proc_embed.shape)
        print("Lab embed: ", lab_embed.shape)
        print("Cond embed: ", cond_embed.shape)
        #print("[INITIAL] Cond embed: ", cond_embed.shape)
        """

        # Demo embeddings (ensure demo tensor is on the correct device)
        demo = demo.to(device)
        gender_embed = self.genderEmbed(demo[:, 0].long()) if self.genderEmbed is not None else torch.zeros(self.batch_size, self.seq_len, self.embed_size, device=self.device)
        eth_embed = self.ethEmbed(demo[:, 1].long()) if self.ethEmbed is not None else torch.zeros(self.batch_size, self.seq_len, self.embed_size, device=self.device)
        ins_embed = self.insEmbed(demo[:, 2].long())
        age_embed = self.ageEmbed(demo[:, 3].long())
        """
        print("Gender: ", gender_embed.shape)
        print("eth_embed: ", eth_embed.shape)
        print("ins_embed: ", ins_embed.shape)
        print("age_embed: ", age_embed.shape)
        """
        # Reshaping to match dimensions
        med_embed_reshaped = med_embed.permute(0, 1, 3, 2)  # Shape: [batch_size, 24, 52, 19]
        chart_embed_reshaped = chart_embed.permute(0, 1, 3, 2)  # Shape: [batch_size, 24, 52, 45]
        out_embed_reshaped = out_embed.permute(0, 1, 3, 2)  # Shape: [batch_size, 24, 52, 7]
        proc_embed_reshaped = proc_embed.permute(0, 1, 3, 2)  # Shape: [batch_size, 24, 52, 15]
        lab_embed_reshaped = lab_embed.unsqueeze(-1)  # Shape: [batch_size, 24, 52, 1]
        cond_embed_reshaped = cond_embed.unsqueeze(1).expand(-1, 24, -1, -1).permute(0, 1, 3, 2)  # Shape: [batch_size, 24, 52, 152]
        #print(f"[RESHAPED] Cond embed: {cond_embed_reshaped.shape}")

        # First, unsqueeze the embeddings to add a dimension for time (sequence length).
        gender_embed = gender_embed.unsqueeze(1).repeat(1, 24, 1)  # Shape: [batch_size, 1, 52]
        eth_embed = eth_embed.unsqueeze(1).repeat(1, 24, 1)        # Shape: [batch_size, 1, 52]
        ins_embed = ins_embed.unsqueeze(1).repeat(1, 24, 1)        # Shape: [batch_size, 1, 52]
        age_embed = age_embed.unsqueeze(1).repeat(1, 24, 1)        # Shape: [batch_size, 1, 52]

        # Finally, unsqueeze the last dimension to add the singleton dimension (size 1).
        gender_embed_reshaped = gender_embed.unsqueeze(-1)  # Shape: [batch_size, 24, 52, 1]
        eth_embed_reshaped = eth_embed.unsqueeze(-1)        # Shape: [batch_size, 24, 52, 1]
        ins_embed_reshaped = ins_embed.unsqueeze(-1)        # Shape: [batch_size, 24, 52, 1]
        age_embed_reshaped = age_embed.unsqueeze(-1)        # Shape: [batch_size, 24, 52, 1]

        """
        # Print the new shapes of the embeddings after reshaping
        print(f"Meds embed: {med_embed_reshaped.shape}")
        print(f"Chart embed: {chart_embed_reshaped .shape}")
        print(f"Out embed: {out_embed_reshaped .shape}")
        print(f"Proc embed: {proc_embed_reshaped .shape}")
        print(f"Lab embed: {lab_embed_reshaped .shape}")
        print(f"Cond embed: {cond_embed_reshaped.shape}")
        print(f"Gender embed: {gender_embed_reshaped .shape}")
        print(f"Eth embed: {eth_embed_reshaped .shape}")
        print(f"Ins embed: {ins_embed_reshaped .shape}")
        print(f"Age embed: {age_embed_reshaped .shape}")
        """

        # Concatenate embeddings into one tensor
        x = torch.cat([med_embed_reshaped, chart_embed_reshaped, out_embed_reshaped, proc_embed_reshaped,
               lab_embed_reshaped, cond_embed_reshaped, gender_embed_reshaped, eth_embed_reshaped, 
               ins_embed_reshaped, age_embed_reshaped], dim=-1)

        #print("Cat: ", x.shape)
        #print("Passed torch.cat embeddings..")
        x = torch.mean(x, dim=-2)
        print("Mean shape: ", x.shape)
        # Ensure x matches the number of nodes defined by edge_index
        #x = x.view(self.batch_size * self.seq_len, -1).to(self.device) # TODO: REMOVE
        # Linear transformation to latent space
        x = nn.Linear(x.size(-1), self.latent_size).to(device)(x)
        print("[INFO] edges SHAPE: ", edge_index.shape)
        print("linear: ", x.shape)
        exit()
        # Apply GNN layers with entropy-based attention
        x = self.gcn1(x, edge_index)
        x = F.relu(x)
        x = self.entropy_attention(x, edge_index)
        x = self.gcn2(x, edge_index)

        #print("gcn2: ", x.shape)
        """
        # Reshape the tensor to be [batch_size * seq_len, feature_size]
        batch_size, seq_len, num_nodes, feature_size = x.shape  # (batch_size, 24, 52, 256)
        x_reshaped = x.view(batch_size * seq_len, num_nodes * feature_size)
        print("--------------------------")
        print("x_reshaped: ", x_reshaped.shape)
        print("edge_index: ", edge_index.shape)
        print("--------------------------")

       # Step 1: Calculate the number of edges in the graph and expand edge_index
        num_edges = edge_index.shape[1]  # Number of edges in the original edge index

        # Calculate the total number of nodes across all batches and time steps
        num_nodes = batch_size * seq_len  # Total nodes = batch_size * seq_len

        # Repeat edge_index for the number of time steps and batches
        expanded_edge_index = edge_index.repeat(1, batch_size * seq_len)  # Shape: [2, num_edges * batch_size * seq_len]

        # Step 2: Ensure that each edge connects valid nodes in x_reshaped
        # Create a batch_indices tensor to adjust the node indices across batches and time steps
        batch_indices = torch.arange(batch_size * seq_len).repeat(num_edges, 1).view(2, -1).repeat_interleave(2, dim=1)  # Shape: [2, num_edges * batch_size * seq_len]
        print("batch_indices: ", batch_indices.shape)
        # Add batch indices to the expanded edge_index
        expanded_edge_index = expanded_edge_index + batch_indices.to(device)

        # Print expanded_edge_index shape to check
        print("expanded_edge_index shape: ", expanded_edge_index.shape)  # Expected: [2, num_edges * batch_size * seq_len]
        


        # Step 4: Pass the reshaped tensor to GAT
        x = self.gat(x_reshaped, expanded_edge_index)
        print("gat: ", x.shape)
        """
        # If needed, flatten batch and sequence dimensions for a fully connected layer
        x_flat = x.view(-1, 256)  # Shape: [4800, 256] (flattened but sequence preserved per batch item)
        # Apply mean pooling along the sequence dimension (dim=1) to aggregate temporal information
        x_pooled = x.mean(dim=1)  # Shape: [200, 256]
        #print("Pooled x:", x_pooled.shape)  # Should be [200, 256]
        # Final prediction layers
        x_fc = F.relu(self.fc1(x_pooled))  # Shape: [200, hidden_dim]
        out = self.fc2(x_fc)  # Shape: [200, 1] for final output
        #print("out shape:", out.shape)

        # Apply sigmoid for probability output
        sigout1 = self.sigmoid(out)  # Shape: [200, 1]
        #print("sigout1 shape:", sigout1.shape)

        return sigout1,out