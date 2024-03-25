from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, RandomSampler
from torch.optim import SGD
import gc
import torch
from sklearn.metrics import accuracy_score
import numpy as np
from RNN_net import RNN, NLIRNN

GLOVE_DIM = 300

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
        self.xi = args.xi
        self.data=args.data
        self.inner_batch_size = args.inner_batch_size
        self.outer_update_lr = args.outer_update_lr
        self.old_outer_update_lr = args.outer_update_lr
        self.inner_update_lr = args.inner_update_lr
        self.inner_update_step = args.inner_update_step
        self.inner_update_step_eval = args.inner_update_step_eval
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
        self.adapter_model = nn.Sequential(
            nn.Linear(args.encoder_dim, args.fc_dim),
            nn.Tanh(),
            nn.Linear(args.fc_dim, args.fc_dim),
            nn.Tanh(),
            nn.Linear(args.fc_dim, args.n_classes),
        )
        param_count = 0
        for param in self.adapter_model.parameters():
            param_count += param.numel()
        self.z_params = torch.randn(param_count, 1)
        self.z_params = nn.init.xavier_uniform_(self.z_params).to(self.device)

        self.hyper_momentum = [torch.zeros(param.size()).to(self.device) for param in self.meta_model.parameters()]
        self.outer_optimizer = SGD(self.meta_model.parameters(), lr=self.outer_update_lr)
        self.inner_optimizer = SGD(self.adapter_model.parameters(), lr=self.inner_update_lr)
        self.meta_model.train()
        self.adapter_model.train()
        self.gamma = args.gamma
        self.beta = args.beta
        self.nu = args.nu
        self.y_warm_start = args.y_warm_start
        self.normalized = args.grad_normalized
        self.grad_clip = args.grad_clip
        self.no_meta = args.no_meta
        self.criterion = nn.CrossEntropyLoss().to(self.device)

    def forward(self, batch_tasks, training=True, task_batch_id=0, train_batch=None, epoch=0):
        task_accs = []
        task_loss = []
        sum_gradients = []
        num_task = len(batch_tasks)
        num_inner_update_step = self.inner_update_step

        for taskid, task in enumerate(batch_tasks):
            support = task[0]
            query = task[1]
            self.meta_model.to(self.device)
            self.adapter_model.to(self.device)
            support_dataloader = DataLoader(support, sampler=RandomSampler(support),
                                            batch_size=self.inner_batch_size, collate_fn=self.collate_pad_)

            ''' update lower variables'''
            for i in range(0, num_inner_update_step):
                batch = iter(support_dataloader).__next__()
                input, label_id = batch
                embedding_features = predict(self.meta_model, input)
                outputs = self.adapter_model(embedding_features)
                inner_loss = self.criterion(outputs, label_id) + 0.0001 * sum(
                    [x.norm().pow(2) for x in self.adapter_model.parameters()]).sqrt()
                self.inner_optimizer.zero_grad()
                inner_loss.backward()
                self.inner_optimizer.step()

            if training:
                input, label_id = next(iter(support_dataloader))
                embedding_features = predict(self.meta_model, input)
                outputs = self.adapter_model(embedding_features)
                inner_loss = self.criterion(outputs, label_id) + 0.0001 * sum(
                    [x.norm().pow(2) for x in self.adapter_model.parameters()]).sqrt()
                inner_loss.backward(retain_graph=True, create_graph=True)
                gy_grad = [g_param.grad.view(-1) for g_param in self.adapter_model.parameters()]
                gy_grad_flat = torch.unsqueeze(torch.reshape(torch.hstack(gy_grad), [-1]), 1)

                jacob_gy2z = torch.autograd.grad(gy_grad_flat, self.adapter_model.parameters(), grad_outputs=self.z_params, retain_graph=True)
                jacob_gy2z = [j_param.detach().view(-1) for j_param in jacob_gy2z]
                jacob_flat_gy2z = torch.unsqueeze(torch.reshape(torch.hstack(jacob_gy2z), [-1]), 1)
                jacob_gxyz = torch.autograd.grad(gy_grad_flat, self.meta_model.parameters(), grad_outputs=self.z_params)

            self.inner_optimizer.zero_grad()
            self.outer_optimizer.zero_grad()
            query_dataloader = DataLoader(query, sampler=None, batch_size=len(query), collate_fn=self.collate_pad_)
            query_batch = iter(query_dataloader).__next__()
            q_input, q_label_id = query_batch
            q_outputs = self.adapter_model(predict(self.meta_model, q_input))
            q_loss = self.criterion(q_outputs, q_label_id)
            if training:
                self.hyper_momentum, self.z_params = hypergradient(self.args, jacob_flat_gy2z, jacob_gxyz,  self.hyper_momentum, self.z_params, inner_loss, \
                                                                   q_loss, self.meta_model, self.adapter_model)
                # print(f'Task loss: {np.mean(all_loss):.4f}')
                if self.no_meta:
                    self.outer_optimizer.step()
                    self.outer_optimizer.zero_grad()
                else:
                    for i, params in enumerate(self.hyper_momentum):
                        if taskid == 0:
                            sum_gradients.append(params.detach())
                        else:
                            sum_gradients[i] += params.detach()
            q_logits = F.softmax(q_outputs, dim=1)
            pre_label_id = torch.argmax(q_logits, dim=1)
            pre_label_id = pre_label_id.detach().cpu().numpy().tolist()
            q_label_id = q_label_id.detach().cpu().numpy().tolist()
            self.outer_optimizer.zero_grad()
            acc = accuracy_score(pre_label_id, q_label_id)
            task_accs.append(acc)
            task_loss.append(q_loss.detach().cpu())
            torch.cuda.empty_cache()
            print(f'{self.args.methods} Task loss: {np.mean(task_loss):.4f}')

        if training and self.no_meta == False:
            # Average gradient across tasks
            for (grad, param) in zip(sum_gradients, self.meta_model.parameters() ):
                grad = grad / float(num_task)
                param.grad = grad
            # for i, params in enumerate(self.meta_model.parameters()):
            #     params.data -= self.outer_update_lr * sum_gradients[i]
            self.outer_optimizer.step()
            self.outer_optimizer.zero_grad()
            del sum_gradients
            gc.collect()

        return np.mean(task_accs),  np.mean(task_loss)

    def collate_pad(self, data_points):
        """ Pad data points with zeros to fit length of longest data point in batch. """
        s_embeds = data_points[0] if type(data_points[0]) == list or type(data_points[0]) == tuple else data_points[1]
        targets = data_points[1] if type(data_points[0]) == list or type(data_points[0]) == tuple  else data_points[0]

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


def hypergradient(args, jacob_flat_gy2z, jacob_gxyz, hyper_momentum, z_params,  inner_loss, loss, meta_model, adapter_model):
    loss.backward()

    Fy_gradient = [g_param.grad.detach().view(-1) for g_param in adapter_model.parameters()]
    Fx_gradient = [g_param.grad.detach() for g_param in meta_model.parameters()]
    Fy_gradient_flat = torch.unsqueeze(torch.reshape(torch.hstack(Fy_gradient), [-1]), 1)
    z_params -= args.nu * (jacob_flat_gy2z - Fy_gradient_flat)

    hyper_momentum = [args.beta * h + (1-args.beta) * (Fx - Gyxz) for (h, Fx, Gyxz) in zip(hyper_momentum, Fx_gradient, jacob_gxyz)]

    return hyper_momentum, z_params

