from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, RandomSampler
from torch.optim import SGD
import gc
import torch
from sklearn.metrics import accuracy_score
import numpy as np
from .RNN_net import RNN

GLOVE_DIM = 300

class Learner(nn.Module):
    """
    Meta Learner
    """
    def __init__(self, args, training_size):
        """
        :param args:
        """
        super(Learner, self).__init__()
        self.args = args
        self.num_labels = args.num_labels
        self.outer_update_lr = args.outer_update_lr
        self.old_outer_update_lr = args.outer_update_lr
        self.inner_update_lr = args.inner_update_lr
        self.inner_update_step = args.inner_update_step
        self.inner_update_step_eval = args.inner_update_step_eval
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.lamb = args.lamb
        self.nu = args.nu
        self.training_size = training_size
        if args.data == 'news_data':
            self.inner_model = RNN(
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
        if args.data == 'snli' or 'sentment140':
            self.inner_model_y = RNN(
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
            self.inner_model_z = RNN(
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
        self.lambda_x =  torch.ones((self.training_size)).to(self.device)
        self.lambda_x.requires_grad=True

        self.outer_optimizer = SGD([self.lambda_x], lr=self.outer_update_lr)
        self.inner_optimizer_y = SGD(self.inner_model_y.parameters(), lr=self.inner_update_lr)
        self.inner_optimizer_z = SGD(self.inner_model_z.parameters(), lr=self.nu)
        self.inner_stepLR_y = torch.optim.lr_scheduler.StepLR(self.inner_optimizer_y, step_size=args.epoch, gamma=0.8)
        self.inner_stepLR_z = torch.optim.lr_scheduler.StepLR(self.inner_optimizer_z, step_size=args.epoch, gamma=0.8)
        self.outer_stepLR = torch.optim.lr_scheduler.StepLR(self.outer_optimizer, step_size=args.epoch, gamma=0.8)
        self.inner_model_y.train()
        self.inner_model_z.train()
        self.gamma = args.gamma
        self.nu = args.nu
        self.normalized = args.grad_normalized
        self.no_meta = args.no_meta
        self.criterion = nn.CrossEntropyLoss(reduction='none').to(self.device)

    def forward(self, train_loader, training=True, epoch=0):
        task_accs = []
        task_loss = []
        sum_gradients = []
        num_inner_update_step = self.inner_update_step

        for step, data in enumerate(train_loader):
            self.inner_model_y.to(self.device)
            self.inner_model_z.to(self.device)

            all_loss = []
            inner_loss_y = torch.zeros(1).to(self.device)
            inner_loss_z = torch.zeros(1).to(self.device)
            inner_loss_ = torch.zeros(1).to(self.device)
            # inner_loss = torch.zeros(1).to(self.device)

            input_y, label_id_y, data_indx_y = next(iter(train_loader))
            input_z, label_id_z, data_indx_z = next(iter(train_loader))
            input_, label_id_, data_indx_ = next(iter(train_loader))
            outputs_y = predict(self.inner_model_y, input_y)
            outputs_z = predict(self.inner_model_z, input_z)
            outputs_ = predict(self.inner_model_y, input_)
            inner_loss_z += torch.mean(torch.sigmoid(self.lambda_x[data_indx_y])*self.criterion(outputs_z, label_id_z.to(self.device)))\
                            + 0.0001 * sum([x.norm().pow(2) for x in self.inner_model_z.parameters()]).sqrt()
            inner_loss_y += torch.mean(torch.sigmoid(self.lambda_x[data_indx_z])*self.criterion(outputs_y, label_id_y.to(self.device)))\
                            + 0.0001 * sum([x.norm().pow(2) for x in self.inner_model_y.parameters()]).sqrt()
            inner_loss_ += torch.mean(self.criterion(outputs_, label_id_.to(self.device)))

            inner_loss_z.backward()
            self.inner_optimizer_z.step()
            self.inner_optimizer_z.zero_grad()
            self.inner_optimizer_y.zero_grad()
            gy_grad = torch.autograd.grad(inner_loss_y, self.inner_model_y.parameters())
            fy_grad = torch.autograd.grad(inner_loss_,  self.inner_model_y.parameters())
            all_loss.append(inner_loss_y.item())

            for param, fy_g, gy_g in zip(self.inner_model_y.parameters(), fy_grad, gy_grad):
                param.grad =  fy_g + self.lamb * gy_g
            self.inner_optimizer_y.step()
            self.inner_optimizer_y.zero_grad()

            q_input, q_label_id, q_data_indx = data
            q_outputs = predict(self.inner_model_y, q_input)
            loss = torch.mean(self.criterion(q_outputs, q_label_id.to(self.device)))
            task_loss.append(loss.detach().cpu())
            print(f'Task loss: {np.mean(task_loss):.4f}')

            self.outer_optimizer.zero_grad()
            input, label_id, data_indx = next(iter(train_loader))
            outputs = predict(self.inner_model_y, input)
            loss = torch.mean(torch.sigmoid(self.lambda_x[data_indx])*self.criterion(outputs, label_id.to(self.device)))\
                            + 0.0001 * sum([x.norm().pow(2) for x in self.inner_model_y.parameters()]).sqrt()
            g_xy = torch.autograd.grad(loss, self.lambda_x)

            input, label_id, data_indx = next(iter(train_loader))
            outputs = predict(self.inner_model_z, input)
            loss = torch.mean(torch.sigmoid(self.lambda_x[data_indx])*self.criterion(outputs, label_id.to(self.device)))\
                            + 0.0001 * sum([x.norm().pow(2) for x in self.inner_model_z.parameters()]).sqrt()
            g_xz = torch.autograd.grad(loss, self.lambda_x)

            self.lambda_x.grad =  self.lamb * (g_xy[0]- g_xz[0])
            self.outer_optimizer.step()
            self.outer_optimizer.zero_grad()


            q_logits = F.softmax(q_outputs, dim=1)
            pre_label_id = torch.argmax(q_logits, dim=1)
            pre_label_id = pre_label_id.detach().cpu().numpy().tolist()
            q_label_id = q_label_id.detach().cpu().numpy().tolist()
            self.outer_optimizer.zero_grad()
            acc = accuracy_score(pre_label_id, q_label_id)
            task_accs.append(acc)
            torch.cuda.empty_cache()

        self.lamb += self.args.incre_step
        self.inner_stepLR_y.step()
        self.inner_stepLR_y.step()
        self.outer_stepLR.step()
        return np.mean(task_accs), np.mean(task_loss)

    def test(self, test_loader):
        task_accs = []
        task_loss = []

        self.inner_model_y.to(self.device)
        for step, data in enumerate(test_loader):
            q_input, q_label_id, q_data_indx = data
            q_outputs = predict(self.inner_model_y, q_input)
            q_loss = torch.mean(self.criterion(q_outputs, q_label_id.to(self.device)))

            q_logits = F.softmax(q_outputs, dim=1)
            pre_label_id = torch.argmax(q_logits, dim=1)
            pre_label_id = pre_label_id.detach().cpu().numpy().tolist()
            q_label_id = q_label_id.detach().cpu().numpy().tolist()
            acc = accuracy_score(pre_label_id, q_label_id)
            task_accs.append(acc)
            task_loss.append(q_loss.detach().cpu())
            torch.cuda.empty_cache()
            print(f'Task loss: {np.mean(task_loss):.4f}, Task acc: {np.mean(task_accs):.4f}')
        return np.mean(task_accs), np.mean(task_loss)

    def collate_pad_(self, data_points):
        """ Pad data points with zeros to fit length of longest data point in batch. """
        s_embeds = data_points[0] if type(data_points[0]) == list else data_points[1]
        targets = data_points[1] if type(data_points[0]) == list else data_points[0]

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

def predict(net, inputs):
    """ Get predictions for a single batch. """
    s_embed, s_lens = inputs
    outputs = net((s_embed.cuda(), s_lens))
    return outputs



