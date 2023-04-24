import logging
import time
import os.path as osp

# EVAL = True: just test, EVAL = False: train and eval
EVAL = False

# dataset can be 'WIKI', 'MIRFlickr' or 'NUSWIDE'
DATASET = 'NUSWIDE'

if DATASET == 'WIKI':
    DATA_DIR = '/hdd/gyr/projects/datasets/wikipedia_dataset/images'
    LABEL_DIR = '/hdd/gyr/projects/datasets/wikipedia_dataset/raw_features.mat'
    TRAIN_LABEL = '/hdd/gyr/projects/datasets/wikipedia_dataset/trainset_txt_img_cat.list'
    TEST_LABEL = '/hdd/gyr/projects/datasets/wikipedia_dataset/testset_txt_img_cat.list'

    BETA = 0.3  # 图像相似度的权重
    LAMBDA1 = 0.3  # 相似度保持的loss权重
    LAMBDA2 = 0.3
    LAMBDA3 = 0.1

    LR_IMG = 0.01
    LR_TXT = 0.01
    LR_COM = 0.1

    NUM_EPOCH = 100
    EVAL_INTERVAL = 50

if DATASET == 'MIRFlickr':
    LABEL_DIR = '/hdd/gyr/projects/datasets/mirflickr/LAll/mirflickr25k-lall.mat'
    TXT_DIR = '/hdd/gyr/projects/datasets/mirflickr/YAll/mirflickr25k-yall.mat'
    IMG_DIR = '/hdd/gyr/projects/datasets/mirflickr/IAll/mirflickr25k-iall.mat'

    BETA = 0.9  # 图像相似度的权重
    LAMBDA1 = 0.1  # 相似度保持的loss权重
    LAMBDA2 = 0.1
    LAMBDA3 = 0.05

    LR_IMG = 0.01
    LR_TXT = 0.1
    LR_COM = 0.01

    NUM_EPOCH = 60
    EVAL_INTERVAL = 20

if DATASET == 'NUSWIDE':
    LABEL_DIR = '/hdd/gyr/projects/datasets/nuswide/nus-wide-tc10-lall.mat'
    TXT_DIR = '/hdd/gyr/projects/datasets/nuswide/nus-wide-tc10-yall.mat'
    IMG_DIR = '/hdd/gyr/projects/datasets/nuswide/nus-wide-tc10-iall.mat'

    BETA = 0.6  # 图像相似度的权重
    LAMBDA1 = 0.1  # 相似度保持的loss权重
    LAMBDA2 = 0.1
    LAMBDA3 = 0.05

    LR_IMG = 0.01
    LR_TXT = 0.1
    LR_COM = 0.01

    NUM_EPOCH = 60
    EVAL_INTERVAL = 60

BATCH_SIZE = 64
CODE_LEN = 32

ETA = 0.4  # 1-ETA 是联合S 的系数
MU = 1.5  # loss里相似矩阵缩放的系数
GAMMA = 4  # 哈希码损失的 系数

WEIGHT_DIR = "/hdd/gyr/projects/pretrain_weight/swin_tiny_patch4_window7_224.pth"

joo = 0.9  # 图像哈希码融合的权重
zoo = 0.2  # 文本的权重

MOMENTUM = 0.9  # 优化器的动量参数，加速收敛
WEIGHT_DECAY = 5e-4  # 优化器的权值衰减，防止过拟合

GPU_ID = 0
NUM_WORKERS = 1
EPOCH_INTERVAL = 2

MODEL_DIR = './checkpoint'
save_path = './save'

logger = logging.getLogger('train')
logger.setLevel(logging.INFO)
now = time.strftime("%Y-%m-%d___%H-%M-%S", time.localtime(time.time()))
log_name = now + '___log.txt'
log_dir = './log'
txt_log = logging.FileHandler(osp.join(log_dir, log_name))
txt_log.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
txt_log.setFormatter(formatter)
logger.addHandler(txt_log)

stream_log = logging.StreamHandler()
stream_log.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
stream_log.setFormatter(formatter)
logger.addHandler(stream_log)

logger.info('--------------------------Current Settings--------------------------')
logger.info('eval = %s' % EVAL)
logger.info('dataset = %s' % DATASET)
logger.info('beta = %.4f' % BETA)
logger.info('eta = %.4f' % ETA)
logger.info('mu = %.4f' % MU)
logger.info('lambda1 = %.4f' % LAMBDA1)
logger.info('lambda2 = %.4f' % LAMBDA2)
logger.info('lambda3 = %.4f' % LAMBDA3)
logger.info('gamma = %.4f' % GAMMA)
logger.info('lr_img = %.4f' % LR_IMG)
logger.info('lr_txt = %.4f' % LR_TXT)
logger.info('lr_com = %.4f' % LR_COM)
logger.info('batch_size = %d' % BATCH_SIZE)
logger.info('code_len = %d' % CODE_LEN)
logger.info('joo = %.4f' % joo)
logger.info('zoo = %.4f' % zoo)
logger.info('momentum = %.4f' % MOMENTUM)
logger.info('weight_decay = %.4f' % WEIGHT_DECAY)
logger.info('gpu_id=  %d' % GPU_ID)
logger.info('num_workers = %d' % NUM_WORKERS)
logger.info('epoch_interval = %d' % EPOCH_INTERVAL)
logger.info('num_epoch = %d' % NUM_EPOCH)
logger.info('eval_interval = %d' % EVAL_INTERVAL)
logger.info('--------------------------------------------------------------------')