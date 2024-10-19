import datetime
from matplotlib import pyplot as plt
import  torch, os
import  numpy as np
import  scipy.stats
from    torch.utils.data import DataLoader
from    torch.optim import lr_scheduler
import  random, sys, pickle
import  argparse
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import wandb
from datasets.DenseLabelTaskSampler import DenseLabelTaskSampler
from datasets.PhysiQ import PhysiQ
from learner import Learner
from meta import Meta
# from databuilder import TimeSeriesDataset
# from torch_dataset import *
# from utilities import validation2, validation3
from model_configs import *

def mean_confidence_interval(accs, confidence=0.95):
    n = accs.shape[0]
    m, se = np.mean(accs), scipy.stats.sem(accs)
    h = se * scipy.stats.t._ppf((1 + confidence) / 2, n - 1)
    return m, h


def main(args, config, string: str = None):
    # add time stamp to the string
    wandb.init(
    # set the wandb project where this run will be logged
    project="EXACT",

    # track hyperparameters and run metadata
    config=vars(args)
)
    # string_time =  str(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    # writer = SummaryWriter(log_dir=f'./runs/{string}/{string_time}')
    #TODO connect back to my seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    print(args)

    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    maml = Meta(args, config)
    # data parallel:
    # maml = torch.nn.DataParallel(maml)
    maml.to(device)

    tmp = filter(lambda x: x.requires_grad, maml.parameters())
    num = sum(map(lambda x: np.prod(x.shape), tmp))
    print(maml)
    print('Total trainable tensors:', num)
    train_dataset = PhysiQ(root="data", N_way=2, split="train", window_size=200, bg_fg=4)
    test_dataset = PhysiQ(root="data", N_way=2, split="test", window_size=200, bg_fg=4)
    train_sampler = DenseLabelTaskSampler(
            train_dataset, n_way=2, n_shot=4, batch_size=2, n_query=4, n_tasks=10, threshold_ratio=.25
        )
    test_sampler = DenseLabelTaskSampler(
            test_dataset, n_way=2, n_shot=4, batch_size=2, n_query=4, n_tasks=10, threshold_ratio=.25
        )
    # support_images, support_labels, query_images, query_labels, true_class_ids = next(iter(train_loader))
    # batchsz here means total episode number
    # mini = MiniImagenet('./miniimagenet/', mode='train', n_way=args.n_way, k_shot=args.k_spt,
    #                     k_query=args.k_qry,
    #                     batchsz=10000, resize=args.imgsz)
    # mini = TimeSeriesDataset('spar', mode='train_all', n_way=args.n_way, k_shot=args.k_spt, 
    #                          k_query=args.k_qry,
    #                          segmentation=args.segmentation,
    #                           batchsz=args.batchsz, resize=args.imgsz, main_length=200, args=args) # physiq -> spar; spar -> physiq
    # mini_test = TimeSeriesDataset('spars9x', mode='test_all', n_way=args.n_way, k_shot=args.k_spt, 
    #                          k_query=args.k_qry,
    #                          segmentation=args.segmentation,
    #                           batchsz=args.batchsz//100, resize=args.imgsz, main_length=200, args=args) # "_all" means all the data is used for training or testing
    counter =0
    best_dice = 0.0

    cur_model_dir = os.path.join("save_model", f'seg_maml_{string}')
    if not os.path.exists(cur_model_dir):
        os.makedirs(cur_model_dir)
    print(f"The model will be saved to {cur_model_dir}")
    
    for epoch in tqdm(range(args.epoch//10000)):
        # fetch meta_batchsz num of episode each time
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=train_sampler,
            num_workers=0,
            pin_memory=True,
            collate_fn=train_sampler.episodic_collate_fn,
        )
        # db = DataLoader(mini, args.task_num, shuffle=True, num_workers=0, pin_memory=True)

        for step, (x_spt, y_spt, x_qry, y_qry, true_id) in enumerate(train_loader):

            x_spt, y_spt, x_qry, y_qry = x_spt.float().to(device), y_spt.to(device), x_qry.float().to(device), y_qry.to(device)
            accs = maml(x_spt, y_spt, x_qry, y_qry)

            if step % 30 == 0:
                wandb.log({"dice": accs[-1][0], "f2": accs[-1][1], "iou": accs[-1][2], "recall": accs[-1][3], "specificity": accs[-1][4], "precision": accs[-1][5], "euclidean": accs[-1][6], "loss": accs[-1][7]})
                # writer.add_scalar('dice/train', accs[-1][0], counter)
                # writer.add_scalar('f2/train', accs[-1][1], counter)
                # writer.add_scalar('iou/train', accs[-1][2], counter)
                # writer.add_scalar('recall/train', accs[-1][3], counter)
                # writer.add_scalar('specificity/train', accs[-1][4], counter)
                # writer.add_scalar('precision/train', accs[-1][5], counter)
                # writer.add_scalar('euclidean/train', accs[-1][6], counter)
                # writer.add_scalar('loss/train', accs[-1][7], counter)

            if step % 1000 == 0:  # evaluation
                test_dataloader = DataLoader(
                    test_dataset,
                    batch_sampler=test_sampler,
                    num_workers=0,
                    pin_memory=True,
                    collate_fn=test_sampler.episodic_collate_fn,
                )
                
                accs_all_test = []

                for x_spt, y_spt, x_qry, y_qry, true_id in test_dataloader:
                    x_spt, y_spt, x_qry, y_qry = x_spt.squeeze(0).to(device), y_spt.squeeze(0).to(device), \
                                                 x_qry.squeeze(0).to(device), y_qry.squeeze(0).to(device)

                    accs = maml.finetunning(x_spt, y_spt, x_qry, y_qry)
                    
                    accs_all_test.append(accs)

                # [b, update_step+1]
                accs = np.array(accs_all_test).mean(axis=0).astype(np.float16)
                # print('Test acc:', accs)
                torch.save(maml.state_dict(), os.path.join(cur_model_dir, f'epoch_{epoch}_step_{step}.pt'))
                if float(accs[-1][0]) > best_dice:
                    best_dice = float(accs[-1][0])
                    print(f"Best dice so far: {best_dice}, saving model {cur_model_dir}/best_model.pt")
                    torch.save(maml.state_dict(), os.path.join(cur_model_dir, f'best_model.pt'))


                # writer.add_scalar('dice/test', accs[-1][0], counter)
                # writer.add_scalar('f2/test', accs[-1][1], counter)
                # writer.add_scalar('iou/test', accs[-1][2], counter)
                # writer.add_scalar('recall/test', accs[-1][3], counter)
                # writer.add_scalar('specificity/test', accs[-1][4], counter)
                # writer.add_scalar('precision/test', accs[-1][5], counter)
                # writer.add_scalar('euclidean/test', accs[-1][6], counter)
                # writer.add_scalar('loss/test', accs[-1][7], counter)

            if step % 30 == 0:
                counter+=1


def main_test_only(config, string):
    torch.manual_seed(222)
    torch.cuda.manual_seed_all(222)
    np.random.seed(222)
    #NOTE: Unet:  rotate_unet_trainedonspar_FINALMODEL_1ways_tpe_sl1_lr_0.01


    
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    maml = Meta(args, config).to(device)
    mini_test = TimeSeriesDataset('oppo', mode='test', n_way=args.n_way, k_shot=args.k_spt, 
                             k_query=args.k_qry,
                             segmentation=args.segmentation,
                              batchsz=2, resize=args.imgsz, args=args,
                              one_subject=1) # "_all" means all the data is used for training or testing

    maml.load_state_dict(torch.load('./save_model/seg_maml_{}.pt'.format(string), map_location=device))
    # le = Learner(config).to(device)
    # le.load_state_dict(torch.load('./save_model/best_pretrained_model.pth', map_location=device))
    # maml.net = le

    db_test = DataLoader(mini_test, 1, shuffle=True, num_workers=0, pin_memory=True)


    for x_spt, y_spt, x_qry, y_qry in db_test:
        x_spt, y_spt, x_qry, y_qry = x_spt.squeeze(0).to(device), y_spt.squeeze(0).to(device), \
                                        x_qry.squeeze(0).to(device), y_qry.squeeze(0).to(device)

        net = maml.finetune_load(x_spt, y_spt, x_qry, y_qry)

    val_dataset = EXACTDataset(dataset='oppo', y_label='all', mode='test', slide_windows=True, 
                             window_size=200, window_step=50, 
                             min_rep=19, rep_step=19, n_way=1, 
                             midnoise=-1, side_noise=-1, with_imu=False, shuffle=False, one_subject=1)
    # val_dataset = EXACTDataset(dataset='physiq', y_label='all', mode="test", shuffle=False,
    #                             slide_windows=True, window_size=args.input_length, window_step=args.main_length,
    #                               min_rep=5, rep_step=3, n_way=args.n_way)
    # # val_dataset = EXACTDataset(dataset='spar', y_label='all', mode="test", shuffle=False, slide_windows=True, window_size=args.input_length, 
    #                             # window_step=args.main_length, min_rep=18, rep_step=1, n_way=args.n_way)
    # # sampler = SingleClassSampler(val_dataset, class_idx=1, size_output=3, in_which_index=1)
    
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.k_spt, shuffle=False) #, sampler=sampler)
    # x_sqt, (_, y_spt, _) = next(iter(val_loader))
    # net = maml.meta_finetune(x_sqt, y_spt)
    # validation3(val_loader=val_loader, model=net, device=device, window_size=args.input_length, look_back_length=args.input_length-args.main_length)
    # validation2(val_loader=val_loader, model=net, device=device, window_size=args.input_length, look_back_length=args.input_length-args.main_length, vis=True)



        

if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--epoch', type=int, help='epoch number', default=30000)
    argparser.add_argument('--n_way', type=int, help='n way', default=1)
    argparser.add_argument('--k_spt', type=int, help='k shot for support set', default=10) # slightly better than k_s=1, k_q = 15
    argparser.add_argument('--k_qry', type=int, help='k shot for query set', default=20)
    argparser.add_argument('--imgsz', type=int, help='imgsz', default=84)
    argparser.add_argument('--batchsz', type=int, help='batch size', default=10000)
    argparser.add_argument('--imgc', type=int, help='imgc', default=3)
    argparser.add_argument('--task_num', type=int, help='meta batch size, namely task num', default=4)
    argparser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=1e-3)
    argparser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=0.01)
    argparser.add_argument('--update_step', type=int, help='task-level inner update steps', default=5)
    argparser.add_argument('--update_step_test', type=int, help='update steps for finetunning', default=10) # 10
    argparser.add_argument('--main_length', type=int, help='this length is the length to be returned for segmentation', default=50) # this length is the output length of the model
    argparser.add_argument('--input_length', type=int, help='this length is the length to be returned for segmentation', default=300) # this length is the input length of the model
    argparser.add_argument('--segmentation', type=bool, help='bool to make the deep learning segmentation problem', default=True)
    
    # noise:
    argparser.add_argument('--midnoise', type=int, help='bool to add midnoise (0: no noise, -1: random length of noise point [up to 50], >0: constant)', default=-1)
    argparser.add_argument('--side_noise', type=int, help='bool to add side noise (0: no noise, -1: random length of noise point [up to 200], >0: constant)', default=-1)
    argparser.add_argument('--with_imu', type=bool, help='using imu quaternion to simulate imu noise', default=False)
    argparser.add_argument('--rotation_degree', type=int, help='rotation degree for imu noise', default=0)
    
    # augmentation:
    argparser.add_argument('--augmentation', type=float, help='rotation augmentation for training', default=1)
    # seed:
    argparser.add_argument('--seed', type=int, help='seed for random', default=73054770) # default = 222

    args = argparser.parse_args()
    # simple model:
    # main('1_1supp_5ways_cnn_mse_lr_0.01')
    # string = 'trainedonphysiq_FINALMODEL_1ways_cnn_sl1_lr_0.01'
    # main(string)
    # main_test_only(string+'_2083')
    # string = 'trainedonspar_FINALMODEL_1ways_tpe_sl1_lr_0.01'
    # main(string)
    # main_test_only(string+'_1159')

    # string = 'invert_trainedonspar_FINALMODEL_1ways_tpe_sl1_lr_0.01'
    # main(string)
    # main_test_only(string+'_924')

    # string = 'invert_unet_trainedonspar_FINALMODEL_1ways_tpe_sl1_lr_0.01'
    # main(string)
    # main_test_only(string+'_')
    

    string = 'EXACT_SPAR_testSPAR9x'
    config = EXACT_config
    args.midnoise = -1
    args.side_noise = -1
    args.augmentation = 1.0
    args.seed = 222
    main(args, config, string)

    # string = 'EXACT_SPAR_testSPAR9x_Noise0.5'
    # config = EXACT_config
    # args.midnoise = -1
    # args.side_noise = -1
    # args.augmentation = 1
    # main(config, string)

    # string = 'EXACT_SPAR_testSPAR9x_Noise0.5'
    # config = EXACT_config
    # args.midnoise = -1
    # args.side_noise = -1
    # args.augmentation = 1
    # main(config, string)

    #NOTE: EXACT trained on spar only with noise data with Perc augmentation:======================================================================================
   # string = 'EXACT_SPAR_testSPAR9x_aug0'
    #config = EXACT_config
   # args.midnoise = -1
   # args.side_noise = -1
   # args.augmentation = 0
   # main(config, string)  # seed =73054772
    # main_test_only(config, string+'_504')
    #==============================================

    #NOTE: BASELINE trained on spar  only without noise data:======================================================================================
    # string = 'baselineCNN_SPAR_NoNoise'
    # config = BASELINE_config
    # args.midnoise = 0
    # args.side_noise = 0
    # main(config, string)  # seed =73054772
    # main_test_only(config, string+'_504')
    #======================================================================================================================================

    # #NOTE: BASELINE trained on spar  only with noise data:======================================================================================
    # string = 'baselineCNN_SPAR_Noise'
    # config = BASELINE_config
    # args.midnoise = -1
    # args.side_noise = -1
    # main(config, string)  # seed =73054772
    # # main_test_only(config, string+'_504')
    # #======================================================================================================================================


    #NOTE: ablation w/o tpe only with noise data:======================================================================================
    # string = 'w/o_TPE_EXACT_Noise'
    # config = WITHOUT_TPE_config
    # args.midnoise = -1
    # args.side_noise = -1
    # main(config, string)  # seed =73054772
    # main_test_only(config, string+'_504')
    #======================================================================================================================================
    
    #NOTE: ablation w/o aspp only with noise data:======================================================================================
    # string = 'w/o_ASPP_EXACT_Noise'
    # config = WITHOUT_ASPP_config
    # args.midnoise = -1
    # args.side_noise = -1
    # main(config, string)  # seed =73054772
    # main_test_only(config, string+'_504')
    #======================================================================================================================================




    #NOTE: EXACTSEG trained on spar  only with noise data:======================================================================================
    # string = 'EXACTSeg_SPAR_Noise'
    # config = EXACT_config
    # args.midnoise = -1ss as
    # args.side_noise = -1
    # main(config, string)  # seed =73054772
    # main_test_only(config, string+'/best_model.pt')
    #======================================================================================================================================


    #NOTE: EXACTSEG trained on spar  only without noise data:======================================================================================
    # string = 'EXACTSeg_SPAR_NoNoise'
    # config = EXACT_config
    # args.midnoise = 0
    # args.side_noise = 0
    # main(config, string)  # seed =222
    # main_test_only(config, string+'/best_model.pt')
    #======================================================================================================================================

    #NOTE: trained on physiq only without noise data
    # string = 'rotate_unet_trainedonphysiq_FINALMODEL_1ways_tpe_sl1_lr_0.01'
    # main(string) 
    # main_test_only(string+'_504')

    #NOTE: trained on physiq only with noise data
    # string = 'rotate_unet_trainedonphysiq_noisy_FINALMODEL_1ways_tpe_sl1_lr_0.01'
    # main(string)
    # string = 'rotate_unet_trainedonphysiq_noisy_FINALMODEL_1ways_tpe'
    # main_test_only(string + '_235')






    # main('pretrained')
    # main_test_only('pretrained')
    # aspp attn model:
    # main('sl1_big_model_5ways_attn_aspp_lr_-0.075')

    # bigger model of aspp attn:
    # main('_5ways_unet_dice_sl1_lr_-.0075')

    # main('testing')
    # main_test_only('_5ways_attn_aspp_lr_-.0075_1226')
