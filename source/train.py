import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from metric import compress, compress_wiki, calculate_map, calculate_top_map
import datasets
import settings
from models import ImgNet, TxtNet, DIS
import os.path as osp
import os
import numpy as np

def load_checkpoints(logger, CodeNet_I, CodeNet_T, file_name='latest.pth'):
    ckp_path = osp.join(settings.MODEL_DIR, file_name)
    try:
        obj = torch.load(ckp_path, map_location=lambda storage, loc: storage.cuda())
        logger.info('**************** Load checkpoint %s ****************' % ckp_path)
    except IOError:
        logger.error('********** No checkpoint %s!*********' % ckp_path)
        return
    CodeNet_I.load_state_dict(obj['ImgNet'])
    CodeNet_T.load_state_dict(obj['TxtNet'])
   
    logger.info('********** The loaded model has been trained for %d epochs.*********' % obj['step'])


def save_checkpoints(logger, CodeNet_I, CodeNet_T, step, file_name='latest.pth'):
    ckp_path = osp.join(settings.MODEL_DIR, file_name)
    obj = {
        'ImgNet': CodeNet_I.state_dict(),
        'TxtNet': CodeNet_T.state_dict(),
    
        'step': step,
    }
    torch.save(obj, ckp_path)
    logger.info('**********Save the trained model successfully.**********')


def eval(logger, CodeNet_I, CodeNet_T, database_loader, test_loader, database_dataset, test_dataset):
    logger.info('--------------------Evaluation: Calculate top MAP-------------------')

    # Change model to 'eval' mode (BN uses moving mean/var).
    CodeNet_I.eval().cuda()
    CodeNet_T.eval().cuda()

    if settings.DATASET == "WIKI":
        re_BI, re_BT, re_L, qu_BI, qu_BT, qu_L = compress_wiki(database_loader, test_loader, CodeNet_I,
                                                               CodeNet_T, database_dataset, test_dataset)
                                                        

    if settings.DATASET == "MIRFlickr" or settings.DATASET == "NUSWIDE":
        re_BI, re_BT, re_L, qu_BI, qu_BT, qu_L = compress(database_loader, test_loader, CodeNet_I,
                                                          CodeNet_T, database_dataset, test_dataset)

    
    MAP_I2T = calculate_top_map(qu_B=qu_BI, re_B=re_BT, qu_L=qu_L, re_L=re_L, topk=50)
    MAP_T2I = calculate_top_map(qu_B=qu_BT, re_B=re_BI, qu_L=qu_L, re_L=re_L, topk=50)
    
   
    return MAP_I2T, MAP_T2I, re_BI, re_BT, re_L, qu_BI, qu_BT, qu_L


def train(epoch, logger, CodeNet_I, CodeNet_T, ComNet, opt_I, opt_T, opt_com, train_dataset, train_loader):
    CodeNet_I.cuda().train()
    CodeNet_T.cuda().train()
    ComNet.cuda().train()

    CodeNet_I.set_alpha(epoch)
    CodeNet_T.set_alpha(epoch)

    logger.info('Epoch [%d/%d], ---------alpha for ImgNet: %.3f, alpha for TxtNet: %.3f' % (
        epoch + 1, settings.NUM_EPOCH, CodeNet_I.alpha, CodeNet_T.alpha))

    for idx, (img, F_T, labels, _) in enumerate(train_loader):
        img = img.cuda()  # img tensor:(batch,3,224,224)  F_T tensor:(batch,1386)  label tensor:(batch,24)
        F_T = torch.FloatTensor(F_T.numpy()).cuda()
        labels = labels.cuda()

        opt_I.zero_grad()
        opt_T.zero_grad()
        opt_com.zero_grad()
        # F_I tensor:(batch,4096)
        # F_I , _, _ = self.FeatNet_I(img)
        F_I, hid_I, code_I = CodeNet_I(img)
        _, hid_T, code_T = CodeNet_T(F_T)

        #hid_concat = code_T + code_I
        #code_com = ComNet(hid_concat)
        com_f = ComNet(F_I, F_T)

        F_I = F.normalize(F_I)  # S_I tensor:(batch,batch)
        S_I = F_I.mm(F_I.t())
        S_I = S_I * 2 - 1

        F_T = F.normalize(F_T)
        S_T = F_T.mm(F_T.t())
        S_T = S_T * 2 - 1

        B_I = F.normalize(code_I)
        B_T = F.normalize(code_T)
        B_C = F.normalize(com_f)

        BI_BI = B_I.mm(B_I.t())
        BT_BT = B_T.mm(B_T.t())
        BI_BT = B_I.mm(B_T.t())
        BC_BC = B_C.mm(B_C.t())

        S_tilde = settings.BETA * S_I + (1 - settings.BETA) * S_T
        S = (1 - settings.ETA) * S_tilde + settings.ETA * S_tilde.mm(S_tilde) / settings.BATCH_SIZE
        S = S * settings.MU

        loss1 = F.mse_loss(BI_BI, S)
        loss2 = F.mse_loss(BI_BT, S)
        loss3 = F.mse_loss(BT_BT, S)
        loss4 = F.mse_loss(B_I, B_C) + F.mse_loss(BI_BT, BC_BC) + F.mse_loss(B_T, B_C)
        loss5 = F.mse_loss(BC_BC, S)
        loss = settings.LAMBDA1 * loss1 + 1 * loss2 + settings.LAMBDA2 * loss3 + settings.GAMMA * loss4 + settings.LAMBDA3 * loss5

        loss.backward()
        opt_I.step()
        opt_T.step()
        opt_com.step()

        if (idx + 1) % (len(train_dataset) // settings.BATCH_SIZE / settings.EPOCH_INTERVAL) == 0:
            logger.info(
                'Epoch [%d/%d], Iter [%d/%d] Loss1: %.4f Loss2: %.4f Loss3: %.4f Loss4: %.4f Loss5: %.4f Total Loss: %.4f'
                % (epoch + 1, settings.NUM_EPOCH, idx + 1, len(train_dataset) // settings.BATCH_SIZE,
                   loss1.item(), loss2.item(), loss3.item(), loss4.item(), loss5.item(), loss.item()))


def main():
    logger = settings.logger
    start = time.time()
    torch.cuda.set_device(settings.GPU_ID)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    
    if settings.DATASET == "WIKI":
        train_dataset = datasets.WIKI(root=settings.DATA_DIR, train=True, transform=datasets.wiki_train_transform)
        test_dataset = datasets.WIKI(root=settings.DATA_DIR, train=False, transform=datasets.wiki_test_transform)
        database_dataset = datasets.WIKI(root=settings.DATA_DIR, train=True, transform=datasets.wiki_test_transform)


    if settings.DATASET == "MIRFlickr":
        train_dataset = datasets.MIRFlickr(train=True, transform=datasets.mir_train_transform)
        test_dataset = datasets.MIRFlickr(train=False, database=False, transform=datasets.mir_test_transform)
        database_dataset = datasets.MIRFlickr(train=False, database=True, transform=datasets.mir_test_transform)

    if settings.DATASET == "NUSWIDE":
        train_dataset = datasets.NUSWIDE(train=True, transform=datasets.nus_train_transform)
        test_dataset = datasets.NUSWIDE(train=False, database=False, transform=datasets.nus_test_transform)
        database_dataset = datasets.NUSWIDE(train=False, database=True, transform=datasets.nus_test_transform)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=settings.BATCH_SIZE,
                                               shuffle=True,
                                               num_workers=settings.NUM_WORKERS,
                                               drop_last=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=settings.BATCH_SIZE,
                                              shuffle=False,
                                              num_workers=settings.NUM_WORKERS)
    database_loader = torch.utils.data.DataLoader(dataset=database_dataset,
                                                  batch_size=settings.BATCH_SIZE,
                                                  shuffle=False,
                                                  num_workers=settings.NUM_WORKERS)
    CodeNet_I = ImgNet(code_len=settings.CODE_LEN)
    txt_feat_len = datasets.txt_feat_len
    CodeNet_T = TxtNet(code_len=settings.CODE_LEN, txt_feat_len=txt_feat_len)
    ComNet = DIS(hash_dim=settings.CODE_LEN, txt_feat_len=txt_feat_len)

    params = filter(lambda p: p.requires_grad, CodeNet_I.parameters())
    opt_I = torch.optim.SGD(params, lr=settings.LR_IMG, momentum=settings.MOMENTUM,
                            weight_decay=settings.WEIGHT_DECAY)

    opt_T = torch.optim.SGD(CodeNet_T.parameters(), lr=settings.LR_TXT, momentum=settings.MOMENTUM,
                            weight_decay=settings.WEIGHT_DECAY)
    opt_com = torch.optim.SGD(ComNet.parameters(), lr=settings.LR_COM, momentum=settings.MOMENTUM, 
                              weight_decay=settings.WEIGHT_DECAY)
                              
    best_it = best_ti = 0

    if settings.EVAL == True:
        # load_checkpoints(logger, CodeNet_I, CodeNet_T)
        # eval(start, epoch, logger, CodeNet_I, CodeNet_T, database_loader, test_loader, database_dataset, test_dataset)

    else:
        for epoch in range(settings.NUM_EPOCH):
            # train the Model
            train(epoch, logger, CodeNet_I, CodeNet_T, ComNet, opt_I, opt_T, opt_com, train_dataset, train_loader)
            # eval the Model
            if (epoch + 1) % settings.EVAL_INTERVAL == 0:
                MAP_I2T, MAP_T2I, re_BI, re_BT, re_L, qu_BI, qu_BT, qu_L = eval(logger, CodeNet_I, CodeNet_T, database_loader, test_loader, database_dataset, test_dataset)
                
                logger.info('mAP@50 I->T: %.3f, mAP@50 T->I: %.3f' % (MAP_I2T, MAP_T2I))

                if (best_it + best_ti) < (MAP_I2T + MAP_T2I):
                    best_it, best_ti = MAP_I2T, MAP_T2I
                    logger.info('Best MAP of I->T: %.3f, Best mAP of T->I: %.3f' % (best_it, best_ti))
                    
                logger.info('--------------------------------------------------------------------')
            
            # save the model
            # if epoch + 1 == settings.NUM_EPOCH:
            #     save_checkpoints(logger, CodeNet_I, CodeNet_T, step=epoch + 1, file_name='latest.pth')


if __name__ == '__main__':
    main()