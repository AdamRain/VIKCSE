## 提供基础工具，如模型layer（embedding、encoder），loss计算

## embedding 就用 bertembedding啊，反正也没改什么
## encoder 就用 bertencoder啊，反正也没改什么
import sys
import copy
import math
import torch
import torch.nn as nn
from prettytable import PrettyTable

PATH_TO_SENTEVAL = 'F:/Models/temp/SentEval/'
PATH_TO_DATA = 'F:/Models/temp/SentEval/data'


# Import SentEval
sys.path.insert(0, PATH_TO_SENTEVAL)
import senteval

# from icecream import ic

class MLPLayer(nn.Module):
    """
    Head for getting sentence representations over RoBERTa/BERT's CLS representation.
    """

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size) # ic| config.hidden_size: 768
        self.activation = nn.Tanh()

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = self.activation(x)
        return x

class ProjectionMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        in_dim = config.hidden_size
        hidden_dim = config.hidden_size * 2
        out_dim = config.hidden_size
        affine=False
        list_layers = [nn.Linear(in_dim, hidden_dim, bias=False),
                       nn.BatchNorm1d(hidden_dim),
                       nn.ReLU(inplace=True)] #nn.ReLU(inplace=True)
        list_layers += [nn.Linear(hidden_dim, out_dim, bias=False),
                        nn.BatchNorm1d(out_dim, affine=affine)]
        self.net = nn.Sequential(*list_layers)

    def forward(self, x):
        return self.net(x)

class Similarity(nn.Module):
    """
    Dot product or cosine similarity
    """

    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp

class InfoNCE(nn.Module):
    def __init__(self,temp=0.05):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)
        self.loss_fct = nn.CrossEntropyLoss()
        
    def forward(self,online_output,target_output):
        target = torch.LongTensor([i for i in range(online_output.shape[0])]).cuda()
        sim_matrix = self.cos(online_output.unsqueeze(1), target_output.unsqueeze(0)) / self.temp
        loss = self.loss_fct(sim_matrix,target)
        return loss


class InfoNCEWithQueue(nn.Module):
    def __init__(self,temp=0.05):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)
        self.loss_fct = nn.CrossEntropyLoss()
        
    def forward(self,query,keys,queue):
        # ic(query.shape)  torch.Size([bs, dim])   torch.Size([122, 768])
        # ic(keys.shape)   torch.Size([bs, dim])   torch.Size([122, 768])
        # ic(queue.shape)  torch.Size([bs, neg_len, dim])   torch.Size([122, 8, 768])

        target = torch.LongTensor([i for i in range(query.shape[0])]).cuda()  # torch.Size([bs])
        # ic| target.shape: torch.Size([bs])
        
        # ic| query.unsqueeze(1).shape: torch.Size([bs, 1, dim])
        # ic| keys.unsqueeze(0).shape: torch.Size([1, bs, dim])
        sim_matrix_pos = self.cos(query.unsqueeze(1),keys.unsqueeze(0))  # torch.Size([bs, bs])

    
        b,l,d = queue.shape  # torch.Size([122, 8, 768])
        # ic| query: torch.Size([bs, 1, dim])             torch.Size([122, 1, 768])
        # ic| queue: torch.Size([1, neg_len*bs, dim])     torch.Size([1, 976, 768])
        sim_matrix_neg = self.cos(query.unsqueeze(1),queue.reshape(b*l,d).unsqueeze(0))   # torch.Size([122, 976])
        sim_matrix = torch.cat((sim_matrix_pos,sim_matrix_neg),dim=1).cuda() / self.temp  # torch.Size([122, 1098])
        loss = self.loss_fct(sim_matrix,target)
        return loss


# evaluate model in all STS tasks
def print_table(task_names, scores):
    tb = PrettyTable()
    tb.field_names = task_names
    tb.add_row(scores)
    print(tb)

def print_full_table(task_names, scores, aligns, uniforms):
    tb = PrettyTable()
    tb.field_names = task_names
    tb.add_row(scores)
    tb.add_row(aligns)
    tb.add_row(uniforms)
    print(tb)

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

def evalModel(model,tokenizer, pooler = "cls_before_pooler"): 
    tasks = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'STSBenchmark', 'SICKRelatedness']
    
    params = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 5}
    params['classifier'] = {'nhid': 0, 'optim': 'rmsprop', 'batch_size': 128,
                                'tenacity': 3, 'epoch_size': 2}
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def prepare(params, samples):
        return

    def batcher(params, batch, max_length=None):
            # Handle rare token encoding issues in the dataset
            if len(batch) >= 1 and len(batch[0]) >= 1 and isinstance(batch[0][0], bytes):
                batch = [[word.decode('utf-8') for word in s] for s in batch]

            sentences = [' '.join(s) for s in batch]
            
            batch = tokenizer.batch_encode_plus(
                sentences,
                return_tensors='pt',
                padding=True,
                max_length=max_length,
                truncation=True
            )
            # Move to the correct device
            for k in batch:
                batch[k] = batch[k].to(device)
            
            # Get raw embeddings
            with torch.no_grad():
                pooler_output = model(**batch, output_hidden_states=True, return_dict=True,sent_emb = True)
                if pooler == "cls_before_pooler":
                    pooler_output = pooler_output.last_hidden_state[:, 0]
                elif pooler == "cls_after_pooler":
                    pooler_output = pooler_output.pooler_output

            return pooler_output.cpu()
    results = {}

    for task in tasks:
        se = senteval.engine.SE(params, batcher, prepare)
        result = se.eval(task)
        results[task] = result
    task_names = []
    scores = []
    for task in ['STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'STSBenchmark', 'SICKRelatedness']:
        task_names.append(task)
        if task in results:
            if task in ['STS12', 'STS13', 'STS14', 'STS15', 'STS16']:
                scores.append("%.2f" % (results[task]['all']['spearman']['all'] * 100))
            else:
                scores.append("%.2f" % (results[task]['test']['spearman'].correlation * 100))
        else:
            scores.append("0.00")
    task_names.append("Avg.")
    scores.append("%.2f" % (sum([float(score) for score in scores]) / len(scores)))
    print_table(task_names, scores)

    return sum([float(score) for score in scores])/len(scores)

class EMA(torch.nn.Module):
    """
    [https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage]
    """

    def __init__(self, model, decay,total_step=15000):
        super().__init__()
        self.decay = decay
        self.total_step = total_step
        self.step = 0
        self.model = copy.deepcopy(model).eval()

    def update(self, model):
        self.step = self.step+1
        decay_new = 1-(1-self.decay)*(math.cos(math.pi*self.step/self.total_step)+1)/2
        # 慢慢把self.model往model移动
        with torch.no_grad():
            e_std = self.model.state_dict().values()
            m_std = model.state_dict().values()
            for e, m in zip(e_std, m_std):
                e.copy_(decay_new * e + (1. - decay_new) * m)

class BYOLMSE(nn.Module):
    def __init__(self,temp=0.05):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)
        #self.loss_fct = nn.CrossEntropyLoss()
    def forward(self,online_output,target_output):
        # online_output batch-size * hidden-size 
        # target_output batch-size * hidden-size 
        
        # 实际上就是2-2*cosine_similarity
        out = torch.diag(self.cos(online_output.unsqueeze(1), target_output.unsqueeze(0)))

        return (2-2*out).mean() / self.temp

def evalTransferModel(model,tokenizer, pooler): 
    tasks = [ 'MR','CR','SUBJ','MPQA','SST2','TREC','MRPC']
    
    params = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 10}
    params['classifier'] = {'nhid': 0, 'optim': 'adam', 'batch_size': 64,
                                         'tenacity': 5, 'epoch_size': 4}
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def prepare(params, samples):
        return

    def batcher(params, batch, max_length=None):
            # Handle rare token encoding issues in the dataset
            if len(batch) >= 1 and len(batch[0]) >= 1 and isinstance(batch[0][0], bytes):
                batch = [[word.decode('utf-8') for word in s] for s in batch]

            sentences = [' '.join(s) for s in batch]
            
            batch = tokenizer.batch_encode_plus(
                sentences,
                return_tensors='pt',
                padding=True,
                max_length=max_length,
                truncation=True
            )
            # Move to the correct device
            for k in batch:
                batch[k] = batch[k].to(device)
            
            # Get raw embeddings
            with torch.no_grad():
                pooler_output = model(**batch, output_hidden_states=True, return_dict=True,sent_emb = True)[0]
                if pooler == "cls_before_pooler":
                    pooler_output = pooler_output.last_hidden_state[:, 0]
                elif pooler == "cls_after_pooler":
                    pooler_output = pooler_output.pooler_output

            return pooler_output.cpu()
    results = {}

    for task in tasks:
        se = senteval.engine.SE(params, batcher, prepare)
        result = se.eval(task)
        results[task] = result
    
    scores = []
    for task in tasks:
        result = results[task]
        scores.append(result['devacc'])
    tasks.append("Avg.")
    scores.append("%.2f" % (sum([float(score) for score in scores]) / len(scores)))
    print_table(tasks, scores)
    return scores