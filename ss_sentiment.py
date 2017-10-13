import argparse
import collections
import numpy as np
import pickle
from tqdm import tqdm

import chainer
from chainer import cuda, optimizers, initializers
import chainer.links as L
import chainer.functions as F
from chainer.utils import walker_alias

# parser setting
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', '-g', default=-1, type=int,
                    help='GPU ID(negative value indicates CPU)')

parser.add_argument('--embed', '-em', default=100, type=int,
                    help='number of units')

parser.add_argument('--window', '-w', default=5, type=int,
                    help='window size')

parser.add_argument('--batchsize', '-b', type=int, default=1000,
                    help='learning minibatch size')

parser.add_argument('--epoch', '-e', default=20, type=int,
                    help='number of epochs to learn')

parser.add_argument('--negative-size', '-ns', default=5, type=int,
                    help='number of negative samples')

parser.set_defaults(test=False)
args = parser.parse_args()

# GPU setting
print('====================')
if args.gpu >= 0:
    chainer.cuda.get_device_from_id(args.gpu).use()
    cuda.check_cuda_available()
    print(' Use GPU  : {}'.format(args.gpu))
else:
    print(' Use CPU')

# print parameter
print('====================')
print(' Embedding     : {}'.format(args.embed))
print(' Window        : {}'.format(args.window))
print(' Minibatch     : {}'.format(args.batchsize))
print(' Epoch         : {}'.format(args.epoch))
print(' Train model   : Skip-gram')
print(' Output type   : Negative Sampling')
print(' Sampling size : {}'.format(args.negative_size))

#=================
# Skip-gram Model
#=================
class SkipGram(chainer.Chain):
    
    def __init__(self, n_vocab, n_units):
        super(SkipGram, self).__init__()

        with self.init_scope():
            self.embed = L.Linear(n_vocab, n_units, initialW=initializers.Uniform(1. / n_units))
            

    def __call__(self, target, context, label):
        ec = self.embed(context)
        et = self.embed(target)
        dot = F.sum(et * ec, axis=1)        

        return F.sigmoid_cross_entropy(dot,label)        
        
        
    
def convert(batch):
    if args.gpu >= 0:
        batch = cuda.to_gpu(batch)
    return batch


#===============
# 損失計算関数
# 本家のw2vと同じくwindow幅をランダム(最大値は指定可)に設定
#==============



class Loss_Calculate:
    def __init__(self,dataset, cs):
        self.window  = args.window
        #self.bs  = args.batchsize
        self.ns = args.negative_size
        
        self.dataset = dataset
        self.sampler = walker_alias.WalkerAlias(np.power(cs,0.75))
        

    def __call__(self,position):
        w = np.random.randint(self.window - 1) + 1
        bs = len(position)
        
        context = None # 文脈語
        target  = None # 対象語　
        label   = []   # Negative 0, Positive : 1 

        for offset in range(-w, w + 1):
            if offset == 0:
                continue

            # positive sampling
            c = np.asarray(self.dataset[position + offset])
            if context is None:
                context = np.eye(n_vocab)[c]
            else:
                context = np.concatenate((context,np.eye(n_vocab)[c]), axis=0)
            
            x = np.asarray(self.dataset[position])
            if target is None:
                target = np.eye(n_vocab)[x]
            else:
                target = np.concatenate((target,np.eye(n_vocab)[x]), axis=0)

            label.extend([ 1 for i in range(bs)])

            
            # negative sampling
            for n in range(bs):
                neg_sample = self.sampler.sample(self.ns)
                context  = np.concatenate((context,np.eye(n_vocab)[neg_sample]), axis=0)

                x_to = np.broadcast_to(x[n],(self.ns,n_vocab))               
                target = np.concatenate((target,x_to), axis=0)
            
                label.extend([ 0 for i in range(self.ns)])

        
        label   = np.array(label,dtype=np.int32)
        
        context = convert(context.astype(np.float32))
        target  = convert(target.astype(np.float32))
        label   = convert(label)
        

        return model(target, context,label) 



#===============
# main process
#===============

# load learning datasets
dataset = np.random.randint(0,10,1000)

counts = collections.Counter(dataset)
cs = [counts[w] for w in range(len(counts))]
n_vocab = max(dataset) + 1
n_data = len(dataset)

print('====================')
print(' vocab size  : {}'.format(n_vocab))
print(' train data  : {}'.format(len(dataset)))


loss_calc = Loss_Calculate(dataset,cs)
model = SkipGram(n_vocab, args.embed)
    
if args.gpu >= 0:
    model.to_gpu()

optimizer = optimizers.Adam()
optimizer.setup(model)


n_win  = args.window
bs = args.batchsize

for i in tqdm(range(args.epoch)):
    indexes = np.random.permutation(n_data)
    for index in range(n_win, n_data-n_win, bs):
        position = dataset[index:index+bs if index+bs < n_data else n_data]
        model.zerograds()
        loss = loss_calc(position)
        loss.backward()  
        optimizer.update()


if args.gpu >= 0:
    w = cuda.to_cpu(model.embed.W.data)
else:
    w = model.embed.W.data


#with open('test_sss.model','wb') as fw:
#    pickle.dump(w,fw)

import chainer.computational_graph as cg
graph = cg.build_computational_graph((loss,), remove_split=True)
with open('./w2v_full.dot', 'w') as fw:
    fw.write(graph.dump())
