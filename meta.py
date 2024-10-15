import  torch
from    torch import nn
from    torch import optim
from    torch.nn import functional as F
from    torch.utils.data import TensorDataset, DataLoader
from    torch import optim
import  numpy as np

from    learner import Learner
from    copy import deepcopy
from    loss_fn import *

def correct_fn(logits_q, y_qry_i):
    # [setsz]
    pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
    # scalar
    res = torch.eq(pred_q, y_qry_i).sum().item()
    return res

def custom_loss(logits_q, y_qry_i):
    # [setsz]
    # cross_entropy = F.cross_entropy(logits_q, y_qry_i)
    # shannon entropy
    # entropy = shannon_entropy(logits_q)
    dice = dice_coefficient_time_series(logits_q, y_qry_i)
    # smooth l1 loss
    smooth_l1 = smooth_l1_loss(logits_q, y_qry_i)

    return  dice + smooth_l1


class Meta(nn.Module):
    """
    Meta Learner
    """
    def __init__(self, args, config):
        """

        :param args:
        """
        super(Meta, self).__init__()

        self.update_lr = args.update_lr
        self.meta_lr = args.meta_lr
        self.n_way = args.n_way
        self.k_spt = args.k_spt
        self.k_qry = args.k_qry
        self.task_num = args.task_num
        self.update_step = args.update_step
        self.update_step_test = args.update_step_test
        self.segmentation = args.segmentation
        if args.segmentation == True:
            self.loss_fn = smooth_l1_loss # AdjustSmoothL1Loss(num_features=args.main_length) #mse_loss 
            self.correct_fn = compute_alls_in_segmentation
        else:
            self.loss_fn = F.cross_entropy
            self.correct_fn = correct_fn

        self.net = Learner(config)
        self.meta_optim = optim.Adam(self.net.parameters(), lr=self.meta_lr)




    def clip_grad_by_norm_(self, grad, max_norm):
        """
        in-place gradient clipping.
        :param grad: list of gradients
        :param max_norm: maximum norm allowable
        :return:
        """

        total_norm = 0
        counter = 0
        for g in grad:
            param_norm = g.data.norm(2)
            total_norm += param_norm.item() ** 2
            counter += 1
        total_norm = total_norm ** (1. / 2)

        clip_coef = max_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            for g in grad:
                g.data.mul_(clip_coef)

        return total_norm/counter


    def forward(self, x_spt, y_spt, x_qry, y_qry):
        """

        :param x_spt:   [b, setsz, c_, h,]
        :param y_spt:   [b, setsz] or [b, 1, setsz] if segmentation
        :param x_qry:   [b, querysz, c_, h]
        :param y_qry:   [b, querysz] or [b, 1, setsz] if segmentation
        :return:
        """
        # NOTE: time series has 4 size and no reisze
        task_num, setsz, c_, h = x_spt.size()

        # task_num, setsz, c_, h, w = x_spt.size()
        querysz = x_qry.size(1)

        losses_q = [0 for _ in range(self.update_step + 1)]  # losses_q[i] is the loss on step i
        corrects = [0 for _ in range(self.update_step + 1)]


        for i in range(task_num):

            # 1. run the i-th task and compute loss for k=0
            # for example: x_spt[i]: torch.Size([3, 200, 6]) y_spt[i]: torch.Size([1, 50]) logits: torch.Size([3, 50])
            logits = self.net(x_spt[i], vars=None, bn_training=True)
            loss = self.loss_fn(logits, y_spt[i])
            grad = torch.autograd.grad(loss, self.net.parameters())
            fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, self.net.parameters())))

            # this is the loss and accuracy before first update
            with torch.no_grad():
                # [setsz, nway]
                logits_q = self.net(x_qry[i], self.net.parameters(), bn_training=True) #NOTE: No fast weights
                loss_q = self.loss_fn(logits_q, y_qry[i])
                losses_q[0] += loss_q

                correct = self.correct_fn(logits_q, y_qry[i])
                corrects[0] = corrects[0] + correct

            # this is the loss and accuracy after the first update
            with torch.no_grad():
                # [setsz, nway]
                logits_q = self.net(x_qry[i], fast_weights, bn_training=True) #NOTE: fast_weights
                loss_q = self.loss_fn(logits_q, y_qry[i])
                losses_q[1] += loss_q
                # [setsz]
                correct = self.correct_fn(logits_q, y_qry[i])
                corrects[1] = corrects[1] + correct

            for k in range(1, self.update_step):
                # 1. run the i-th task and compute loss for k=1~K-1
                logits = self.net(x_spt[i], fast_weights, bn_training=True)
                loss = self.loss_fn(logits, y_spt[i])
                # 2. compute grad on theta_pi
                grad = torch.autograd.grad(loss, fast_weights)
                # 3. theta_pi = theta_pi - train_lr * grad
                fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))

                logits_q = self.net(x_qry[i], fast_weights, bn_training=True)
                # loss_q will be overwritten and just keep the loss_q on last update step.
                loss_q = self.loss_fn(logits_q, y_qry[i])
                losses_q[k + 1] += loss_q

                with torch.no_grad():
                    correct = self.correct_fn(logits_q, y_qry[i])  # convert to numpy
                    corrects[k + 1] = corrects[k + 1] + correct



        # end of all tasks
        # sum over all losses on query set across all tasks
        loss_q = losses_q[-1] / task_num

        # optimize theta parameters
        self.meta_optim.zero_grad()
        loss_q.backward()
        # print('meta update')
        # for p in self.net.parameters()[:5]:
        # 	print(torch.norm(p).item())
        self.meta_optim.step()
        accs = np.array(corrects) / (querysz * task_num)
        if self.segmentation:
            # no need to divide bc it has been averaged on the loss_fn
            accs = np.array(corrects) / task_num

        return accs


    def finetunning(self, x_spt, y_spt, x_qry, y_qry):
        """

        :param x_spt:   [setsz, c_, h, w]
        :param y_spt:   [setsz]
        :param x_qry:   [querysz, c_, h, w]
        :param y_qry:   [querysz]
        :return:
        """
        assert len(x_spt.shape) == 4 or len(x_spt.shape) == 3, 'x_spt shape should be 3 or 4 dimensional but got %d' % len(x_spt.shape)

        querysz = x_qry.size(0)

        corrects = [0 for _ in range(self.update_step_test + 1)]

        # in order to not ruin the state of running_mean/variance and bn_weight/bias
        # we finetunning on the copied model instead of self.net
        net = deepcopy(self.net)

        # 1. run the i-th task and compute loss for k=0
        logits = net(x_spt)
        loss = self.loss_fn(logits, y_spt)
        grad = torch.autograd.grad(loss, net.parameters())
        fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, net.parameters())))

        # this is the loss and accuracy before first update
        with torch.no_grad():
            # [setsz, nway]
            logits_q = net(x_qry, net.parameters(), bn_training=True)
            # [setsz]
            correct = self.correct_fn(logits_q, y_qry)
            corrects[0] = corrects[0] + correct

        # this is the loss and accuracy after the first update
        with torch.no_grad():
            # [setsz, nway]
            logits_q = net(x_qry, fast_weights, bn_training=True)
            correct = self.correct_fn(logits_q, y_qry)
            corrects[1] = corrects[1] + correct

        for k in range(1, self.update_step_test):
            # 1. run the i-th task and compute loss for k=1~K-1
            logits = net(x_spt, fast_weights, bn_training=True)
            loss = self.loss_fn(logits, y_spt)
            # 2. compute grad on theta_pi
            grad = torch.autograd.grad(loss, fast_weights)
            # 3. theta_pi = theta_pi - train_lr * grad
            fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))

            logits_q = net(x_qry, fast_weights, bn_training=True)
            # loss_q will be overwritten and just keep the loss_q on last update step.
            loss_q = self.loss_fn(logits_q, y_qry)

            with torch.no_grad():
                correct = self.correct_fn(logits_q, y_qry)
                corrects[k + 1] = corrects[k + 1] + correct


        del net
        
        accs = np.array(corrects) / querysz

        if self.segmentation:
            # no need to divide bc it has been averaged on the loss_fn
            accs = np.array(corrects)

        return accs

    def meta_finetune(self, x_spt, y_spt):
        # meta test

        # task_num, setsz, c_, h = x_spt.size()

        net = deepcopy(self.net)
        for k in range(1, self.update_step_test):
            # 1. run the i-th task and compute loss for k=1~K-1
            logits = net(x_spt)
            loss = self.loss_fn(logits, y_spt)
            # 2. compute grad on theta_pi
            grad = torch.autograd.grad(loss, net.parameters())
            # 3. theta_pi = theta_pi - train_lr * grad
            fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, net.parameters())))
            logits = net(x_spt, fast_weights)
            with torch.no_grad():
                correct = self.correct_fn(logits, y_spt)
                # print(correct)
                
        return net 
    
    def meta_test(self, x_qry, y_qry, net=None):
        # meta test
        if net is None:
            net = deepcopy(self.net)
        querysz = x_qry.size(0)
        # net = deepcopy(self.net)
        for k in range(1, self.update_step_test):
            logits = net(x_qry)
            with torch.no_grad():
                correct = self.correct_fn(logits, y_qry)
                print(correct)
        return net
        


    def finetune_load(self, x_spt, y_spt, x_qry, y_qry):
        assert len(x_spt.shape) == 4 or len(x_spt.shape) == 3, 'x_spt shape should be 3 or 4 dimensional but got %d' % len(x_spt.shape)

        querysz = x_qry.size(0)

        # in order to not ruin the state of running_mean/variance and bn_weight/bias
        # we finetunning on the copied model instead of self.net
        net = deepcopy(self.net)

        # 1. run the i-th task and compute loss for k=0
        logits = net(x_spt)
        loss = self.loss_fn(logits, y_spt)
        grad = torch.autograd.grad(loss, net.parameters())
        fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, net.parameters())))

        for k in range(1, self.update_step_test):
            # 1. run the i-th task and compute loss for k=1~K-1
            logits = net(x_spt, fast_weights, bn_training=True)
            loss = self.loss_fn(logits, y_spt)
            # 2. compute grad on theta_pi
            grad = torch.autograd.grad(loss, fast_weights)
            # 3. theta_pi = theta_pi - train_lr * grad
            fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))

            logits_q = net(x_qry, fast_weights, bn_training=True)
            # loss_q will be overwritten and just keep the loss_q on last update step.
            loss_q = self.loss_fn(logits_q, y_qry)
            with torch.no_grad():
                correct = self.correct_fn(logits_q, y_qry)
                print(correct)
        return net

        



def main():
    pass


if __name__ == '__main__':
    main()
