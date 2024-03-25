from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, RandomSampler
from torch.optim import SGD
from copy import deepcopy
import gc
import torch
from sklearn.metrics import accuracy_score
import numpy as np
from RNN_net import RNN, NLIRNN
GLOVE_DIM=300
class Learner(nn.Module):
    """
    Meta Learner
    """
    def __init__(self, args):
        """
        :param args:
        """
        super(Learner, self).__init__()
        self.args = args
        self.num_labels = args.num_labels
        self.inner_batch_size = args.inner_batch_size
        self.outer_update_lr  = args.outer_update_lr
        self.old_outer_update_lr = args.outer_update_lr
        self.inner_update_lr  = args.inner_update_lr
        self.inner_update_step = args.inner_update_step
        self.inner_update_step_eval = args.inner_update_step_eval
        self.device =torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.collate_pad_ = self.collate_pad if args.data=='news_data' else self.collate_pad_snli
        if args.data == 'news_data':
            self.meta_model = RNN(
                word_embed_dim=args.word_embed_dim,
                encoder_dim=args.encoder_dim,
                n_enc_layers=args.n_enc_layers,
                dpout_model=0.0,
                dpout_fc=0.0,
                fc_dim=args.fc_dim,
                n_classes=args.n_classes,
                pool_type=args.pool_type,
                linear_fc=args.linear_fc
            )
        if args.data == 'snli':
            self.meta_model = NLIRNN(
                word_embed_dim=args.word_embed_dim,
                encoder_dim=args.encoder_dim,
                n_enc_layers=args.n_enc_layers,
                dpout_model=0.0,
                dpout_fc=0.0,
                fc_dim=args.fc_dim,
                n_classes=args.n_classes,
                pool_type=args.pool_type,
                linear_fc=args.linear_fc
            )
        self.classifier = nn.Sequential(
                nn.Linear(args.encoder_dim, args.fc_dim),
                nn.Tanh(),
                nn.Linear(args.fc_dim, args.fc_dim),
                nn.Tanh(),
                nn.Linear(args.fc_dim, args.n_classes),
            )
        self.outer_optimizer = SGD(self.meta_model.parameters(), lr=self.outer_update_lr, weight_decay=1e-8)
        self.meta_model.train()
        self.gamma = args.gamma
        self.grad_clip = args.grad_clip
        self.no_meta = args.no_meta
        self.criterion = nn.CrossEntropyLoss().to(self.device)

    def forward(self, batch_tasks, training = True, task_batch_id = 0, epoch=0):
        task_accs = []
        task_loss = []
        sum_gradients = []
        num_task = len(batch_tasks)
        num_inner_update_step = self.inner_update_step

        for task_id, task in enumerate(batch_tasks):
            support = task[0]
            query   = task[1]
            if self.no_meta:
                adapter_model = self.classifier
            else:
                adapter_model = deepcopy(self.classifier)
            adapter_model.train()
            self.meta_model.to(self.device)
            adapter_model.to(self.device)
            support_dataloader = DataLoader(support, sampler=RandomSampler(support),
                                            batch_size=self.inner_batch_size, collate_fn=self.collate_pad_)

            inner_optimizer = SGD(adapter_model.parameters(), lr=self.inner_update_lr)
            adapter_model.train()

            all_loss = []
            for i in range(0, num_inner_update_step):
                for inner_step, batch in enumerate(support_dataloader):
                    input,  label_id = batch
                    embedding_features = predict(self.meta_model, input)
                    outputs = adapter_model(embedding_features)
                    loss = self.criterion(outputs, label_id) + 0.001*sum([x.norm().pow(2) for x in adapter_model.parameters()]).sqrt()
                    loss.backward()
                    inner_optimizer.step()
                    inner_optimizer.zero_grad()
                    all_loss.append(loss.item())
                # print('loss/inner_ieteration', loss.item())

            query_dataloader = DataLoader(query, sampler=None, batch_size=len(query), collate_fn=self.collate_pad_)
            query_batch = iter(query_dataloader).__next__()
            q_input, q_label_id = query_batch
            q_embedding_features = predict(self.meta_model, q_input)
            q_outputs = adapter_model(q_embedding_features)

            q_loss = self.criterion(q_outputs, q_label_id)
            if training:
                q_loss.backward()
                if self.no_meta:
                    inner_optimizer.step()
                    inner_optimizer.zero_grad()
                else:
                    for i, params in enumerate(self.meta_model.parameters()):
                        if task_id == 0:
                            sum_gradients.append(deepcopy(params.grad.detach()))
                        else:
                            sum_gradients[i] += deepcopy(params.grad.detach())
            self.outer_optimizer.zero_grad()
            q_logits = F.softmax(q_outputs,dim=1)
            pre_label_id = torch.argmax(q_logits,dim=1)
            pre_label_id = pre_label_id.detach().cpu().numpy().tolist()
            q_label_id = q_label_id.detach().cpu().numpy().tolist()

            acc = accuracy_score(pre_label_id, q_label_id)
            task_loss.append(q_loss.detach().cpu())
            task_accs.append(acc)

            del adapter_model, inner_optimizer
            torch.cuda.empty_cache()
            print(f'{self.args.methods} Task loss: {np.mean(task_loss):.4f}')

        if training and self.no_meta==False:
            # Average gradient across tasks

            for i in range(0, len(sum_gradients)):
                sum_gradients[i] = sum_gradients[i] / float(num_task)

            grad_l2_norm_sq = 0.0
            #Assign gradient for original model, then using optimizer to update its weights
            for i, params in enumerate(self.meta_model.parameters()):
                params.grad = sum_gradients[i]
                if params.grad is None:
                    continue
                grad_l2_norm_sq += torch.sum(sum_gradients[i]*sum_gradients[i])
            grad_l2_norm = torch.sqrt(grad_l2_norm_sq).item()
            clipping_coeff = self.gamma / (1e-10 + grad_l2_norm)
            if self.outer_update_lr < clipping_coeff:
                print('--------- Not clipping ------------')
            if  self.outer_update_lr >= clipping_coeff and self.grad_clip:
                self.outer_update_lr = clipping_coeff
                print('------------ Clipping -------------')
            self.outer_optimizer.step()
            self.outer_optimizer.zero_grad()
            del sum_gradients
            gc.collect()

        self.outer_update_lr = self.old_outer_update_lr
        return np.mean(task_accs), np.mean(task_loss)

    def collate_pad(self, data_points):
        """ Pad data points with zeros to fit length of longest data point in batch. """
        s_embeds = data_points[0] if type(data_points[0])==list else  data_points[1]
        targets = data_points[1] if type(data_points[0])==list else  data_points[0]

        # Get sentences for batch and their lengths.
        s_lens = np.array([sent.shape[0] for sent in s_embeds])
        max_s_len = np.max(s_lens)
        # Encode sentences as glove vectors.
        bs = len(data_points[0])
        s_embed = np.zeros((max_s_len, bs, GLOVE_DIM))
        for i in range(bs):
            e = s_embeds[i]
            if len(e) <= 0:
                s_lens[i] = 1
            s_embed[: len(e), i] = e.copy()
        embeds = torch.from_numpy(s_embed).float().to(self.device)
        targets = torch.LongTensor(targets).to(self.device)
        return (embeds, s_lens), targets

    def collate_pad_snli(self, data_points):
        """ Pad data points with zeros to fit length of longest data point in batch. """
        s_embeds = data_points[0] if type(data_points[0]) == list or type(data_points[0]) == tuple else data_points[1]
        # s_embeds = data_points[0]
        # s2_embeds = data_points[0] if type(data_points[0]) == list or type(data_points[0]) == tuple else data_points[1]
        targets = data_points[1] if type(data_points[0]) == list or type(data_points[0]) == tuple else data_points[0]
        # targets = data_points[1]
        s1_embeds = [x for x in s_embeds[0]]
        s2_embeds = [x for x in s_embeds[1]]
        # targets = [x[1] for x in data_points]

        # Get sentences for batch and their lengths.
        s1_lens = np.array([sent.shape[0] for sent in s1_embeds])
        max_s1_len = np.max(s1_lens)
        s2_lens = np.array([sent.shape[0] for sent in s2_embeds])
        max_s2_len = np.max(s2_lens)
        lens = (s1_lens, s2_lens)

        # Encode sentences as glove vectors.
        bs = len(targets)
        s1_embed = np.zeros((max_s1_len, bs, GLOVE_DIM))
        s2_embed = np.zeros((max_s2_len, bs, GLOVE_DIM))
        for i in range(bs):
            e1 = s1_embeds[i]
            e2 = s2_embeds[i]
            s1_embed[: len(e1), i] = e1.copy()
            s2_embed[: len(e2), i] = e2.copy()
            if len(e1) <= 0:
                s1_lens[i] = 1
            if len(e2) <= 0:
                s2_lens[i] = 1
        embeds = (
            torch.from_numpy(s1_embed).float().to(self.device), torch.from_numpy(s2_embed).float().to(self.device)
        )

        # Convert targets to tensor.
        targets = torch.LongTensor(targets).to(self.device)

        return (embeds, lens), targets

def predict(net, inputs):
    """ Get predictions for a single batch. """
    # snli dataaset
    (s1_embed, s2_embed), (s1_lens, s2_lens) = inputs
    outputs = net((s1_embed, s1_lens), (s2_embed, s2_lens))
    # s_embed, s_lens = inputs
    # outputs = net((s_embed, s_lens))
    return outputs
