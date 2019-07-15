import argparse
import torch
from PIL import Image
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.autograd import Function
import torch.backends.cudnn as cudnn
import os
import numpy as np
from tqdm import tqdm
import operator
import numpy as np
from sklearn.model_selection import KFold
from scipy import interpolate
import matplotlib.pyplot as plt 
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score

def get_performance(y_true, y_pred, avg = 'binary'):

    f1 = f1_score(y_true, y_pred, average = avg)
    acc = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred, average = avg)
    precision = precision_score(y_true, y_pred, average = avg)
    return acc, recall, precision, f1    

def calculate_roc(thresholds, distances, labels, nrof_folds=10):

    nrof_pairs = min(len(labels), len(distances))
    nrof_thresholds = len(thresholds)
    # k_fold = KFold(n_splits=nrof_folds, shuffle=False)

    #tprs = np.zeros((nrof_folds,nrof_thresholds))
    #fprs = np.zeros((nrof_folds,nrof_thresholds))
    accuracy = np.zeros((nrof_thresholds))

    indices = np.arange(nrof_pairs)

    # for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):

        # Find the best threshold for the fold
    acc_train = np.zeros((nrof_thresholds))
    for threshold_idx, threshold in enumerate(thresholds):
        _, _, accuracy[threshold_idx] = calculate_accuracy(threshold, distances, labels)
    best_threshold_index = np.argmax(accuracy)
    best_accuracy = max(accuracy)
        #for threshold_idx, threshold in enumerate(thresholds):
            #tprs[fold_idx,threshold_idx], fprs[fold_idx,threshold_idx], _ = calculate_accuracy(threshold, distances[test_set], labels[test_set])
        #_, _, accuracy[fold_idx] = calculate_accuracy(thresholds[best_threshold_index], distances[test_set], labels[test_set])

        #tpr = np.mean(tprs,0)
        #fpr = np.mean(fprs,0)
    # return tpr, fpr, accuracy
    return accuracy, best_accuracy, thresholds[best_threshold_index]

def calculate_accuracy(threshold, dist, actual_issame):

    predict_issame = np.less(dist, threshold)
    tp = np.sum(np.logical_and(predict_issame, actual_issame))
    fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    tn = np.sum(np.logical_and(np.logical_not(predict_issame), np.logical_not(actual_issame)))
    fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))
    tpr = 0 if (tp+fn==0) else float(tp) / float(tp+fn)
    fpr = 0 if (fp+tn==0) else float(fp) / float(fp+tn)
    acc = float(tp+tn)/dist.size
    return tpr, fpr, acc



def calculate_val(thresholds, distances, labels, far_target=1e-3, nrof_folds=10):
    nrof_pairs = min(len(labels), len(distances))
    nrof_thresholds = len(thresholds)
    k_fold = KFold(n_splits=nrof_folds, shuffle=False)
    val = np.zeros(nrof_folds)
    far = np.zeros(nrof_folds)
    indices = np.arange(nrof_pairs)
    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
        # Find the threshold that gives FAR = far_target
        far_train = np.zeros(nrof_thresholds)
        for threshold_idx, threshold in enumerate(thresholds):
            _, far_train[threshold_idx] = calculate_val_far(threshold, distances[train_set], labels[train_set])
        if np.max(far_train)>=far_target:
            f = interpolate.interp1d(far_train, thresholds, kind='slinear')
            threshold = f(far_target)
        else:
            threshold = 0.0
        val[fold_idx], far[fold_idx] = calculate_val_far(threshold, distances[test_set], labels[test_set])
    val_mean = np.mean(val)
    far_mean = np.mean(far)
    val_std = np.std(val)
    return val_mean, val_std, far_mean


def calculate_val_far(threshold, dist, actual_issame):
    predict_issame = np.less(dist, threshold)
    true_accept = np.sum(np.logical_and(predict_issame, actual_issame))
    false_accept = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    n_same = np.sum(actual_issame)
    n_diff = np.sum(np.logical_not(actual_issame))
    if n_diff == 0:
        n_diff = 1
    if n_same == 0:
        return 0,0
    val = float(true_accept) / float(n_same)
    far = float(false_accept) / float(n_diff)
    return val, far

class Scale(object):
    """Rescales the input PIL.Image to the given 'size'.
    If 'size' is a 2-element tuple or list in the order of (width, height), it will be the exactly size to scale.
    If 'size' is a number, it will indicate the size of the smaller edge.
    For example, if height > width, then image will be
    rescaled to (size * height / width, size)
    size: size of the exactly size or the smaller edge
    interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(self, size, interpolation=Image.BILINEAR):
        #assert isinstance(size, int) or (isinstance(size, collections.Iterable) and len(size) == 2)
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        if isinstance(self.size, int):
            w, h = img.size
            if (w <= h and w == self.size) or (h <= w and h == self.size):
                return img
            if w < h:
                ow = self.size
                oh = int(self.size * h / w)
                return img.resize((ow, oh), self.interpolation)
            else:
                oh = self.size
                ow = int(self.size * w / h)
                return img.resize((ow, oh), self.interpolation)
        else:
            return img.resize(self.size, self.interpolation)

def adjust_learning_rate(optimizer, args):
    """Updates the learning rate given the learning rate decay.
    The routine has been implemented according to the original Lua SGD optimizer
    """
    for group in optimizer.param_groups:
        if 'step' not in group:
            group['step'] = 0
        group['step'] += 1

        group['lr'] = args.lr / (1 + group['step'] * args.lr_decay)


def create_optimizer(model, new_lr, args):
    # setup optimizer
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=new_lr,
                              momentum=0.9, dampening=0.9,
                              weight_decay=args.wd)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=new_lr,
                               weight_decay=args.wd)
    elif args.optimizer == 'adagrad':
        optimizer = optim.Adagrad(model.parameters(),
                                  lr=new_lr,
                                  lr_decay=args.lr_decay,
                                  weight_decay=args.wd)
    return optimizer


def plot_roc(fpr,tpr,figure_name="roc.png"):
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc
    roc_auc = auc(fpr, tpr)
    fig = plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    fig.savefig(os.path.join(LOG_DIR,figure_name), dpi=fig.dpi)

def denormalize(tens):
    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]

    img_1 = tens.clone()
    for t, m, s in zip(img_1, mean, std):
        t.mul_(s).add_(m)
    img_1 = img_1.numpy().transpose(1,2,0)
    return img_1

def evaluate(distances, labels, nrof_folds=10):
    # Calculate evaluation metrics
    thresholds = np.arange(0, 30, 0.01)
    # tpr, fpr, accuracy = calculate_roc(thresholds, distances,
    #    labels, nrof_folds=nrof_folds)
    accuracy, best_accuracy, best_threshold_index = calculate_roc(thresholds, distances,
        labels, nrof_folds=nrof_folds)

    thresholds = np.arange(0, 30, 0.001)
    val, val_std, far = calculate_val(thresholds, distances,
        labels, 1e-3, nrof_folds=nrof_folds)
    return accuracy, best_accuracy, best_threshold_index, val, val_std, far


class TripletMarginLoss(Function):
    """Triplet loss function.
    """
    def __init__(self, margin):
        super(TripletMarginLoss, self).__init__()
        self.margin = margin
        self.pdist = PairwiseDistance(2)  # norm 2

    def forward(self, anchor, positive, negative):
        d_p = self.pdist.forward(anchor, positive)
        d_n = self.pdist.forward(anchor, negative)

        dist_hinge = torch.clamp(self.margin + d_p - d_n, min=0.0)
        loss = torch.mean(dist_hinge)
        return loss

class PairwiseDistance(Function):
    def __init__(self, p):
        super(PairwiseDistance, self).__init__()
        self.norm = p

    def forward(self, x1, x2):
        assert x1.size() == x2.size()
        eps = 1e-4 / x1.size(1)
        diff = torch.abs(x1 - x2)
        out = torch.pow(diff, self.norm).sum(dim=1)
        return torch.pow(out + eps, 1. / self.norm)

def display_triplet_distance(model,train_loader,name):
    #plt.ioff()
    #f, axarr = plt.subplots(3,figsize=(10,10))
    #f.tight_layout()
    l2_dist = PairwiseDistance(2)

    for batch_idx, sample in enumerate(train_loader):
        (data_a, data_p, data_n,c1,c2) = sample['img_a'], sample['img_p'], sample['img_n'], sample['c1'], sample['c2']
        try:
            data_a_c, data_p_c,data_n_c = data_a.cuda(), data_p.cuda(), data_n.cuda()
            data_a_v, data_p_v, data_n_v = Variable(data_a_c, volatile=True), \
                                    Variable(data_p_c, volatile=True), \
                                    Variable(data_n_c, volatile=True)

            (out_a,aux_out_a), (out_p,aux_out_p), (out_n,aux_out_n) = model(data_a_v), model(data_p_v), model(data_n_v)
        except Exception as ex:
            print(ex)
            print("ERROR at: {}".format(batch_idx))
            break

        print("Distance (anchor-positive): {}".format(l2_dist.forward(out_a,out_p).data[0]))
        print("Distance (anchor-negative): {}".format(l2_dist.forward(out_a,out_n).data[0]))


        #axarr[0].imshow(denormalize(data_a[0]))
        #axarr[1].imshow(denormalize(data_p[0]))
        #axarr[2].imshow(denormalize(data_n[0]))
        #axarr[0].set_title("Distance (anchor-positive): {}".format(l2_dist.forward(out_a,out_p).data[0]))
        #axarr[2].set_title("Distance (anchor-negative): {}".format(l2_dist.forward(out_a,out_n).data[0]))

        break
    #f.savefig("{}.png".format(name))
    #plt.show()

from sklearn.decomposition import PCA
import numpy as np

def display_triplet_distance_test(model,test_loader,name):
    f, axarr = plt.subplots(5,2,figsize=(10,10))
    f.tight_layout()
    l2_dist = PairwiseDistance(2)

    for batch_idx, (data_a, data_n,label) in enumerate(test_loader):

        if np.all(label.cpu().numpy()):
            continue

        try:
            data_a_c, data_n_c = data_a.cuda(), data_n.cuda()
            data_a_v, data_n_v = Variable(data_a_c, volatile=True), \
                                    Variable(data_n_c, volatile=True)

            out_a, out_n = model(data_a_v), model(data_n_v)

        except Exception as ex:
            print(ex)
            print("ERROR at: {}".format(batch_idx))
            break

        for i in range(5):
            rand_index = np.random.randint(0, label.size(0)-1)
            if i%2 == 0:
                for j in range(label.size(0)):
                    # Choose label == 0
                    rand_index = np.random.randint(0, label.size(0)-1)
                    if label[rand_index] == 0:
                        break

            distance = l2_dist.forward(out_a,out_n).data[rand_index][0]
            print("Distance: {}".format(distance))
            #distance_pca = l2_dist.forward(PCA(128).fit_transform(out_a.data[i].cpu().numpy()),PCA(128).fit_transform(out_n.data[i].cpu().numpy())).data[0]
            #print("Distance(PCA): {}".format(distance_pca))

            axarr[i][0].imshow(denormalize(data_a[rand_index]))
            axarr[i][1].imshow(denormalize(data_n[rand_index]))
            plt.figtext(0.5, i/5.0+0.1,"Distance : {}, Label: {}\n".format(distance,label[rand_index]), ha='center', va='center')


        break
    plt.subplots_adjust(hspace=0.5)

    f.savefig("{}.png".format(name))
    #plt.show()
