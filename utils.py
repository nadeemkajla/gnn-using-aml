# -*- coding: utf-8 -*-
from __future__ import print_function, division

"""
Pytorch useful tools.
"""

import torch
import os
import errno
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score, confusion_matrix, roc_curve, auc

__author__ = 'Pau Riba'
__email__ = 'priba@cvc.uab.cat'


def save_checkpoint(state, directory, file_name):
    if not os.path.isdir(directory):
        os.makedirs(directory)
    checkpoint_file = os.path.join(directory, file_name + '.pth')
    torch.save(state, checkpoint_file)


def load_checkpoint(model_file):
    if os.path.isfile(model_file):
        print("=> loading model '{}'".format(model_file))
        checkpoint = torch.load(model_file)
        print("=> loaded model '{}' (epoch {}, acc {})".format(model_file, checkpoint['epoch'], checkpoint['best_acc']))
        return checkpoint
    else:
        print("=> no model found at '{}'".format(model_file))
        raise OSError(errno.ENOENT, os.strerror(errno.ENOENT), model_file)


def write_gxl(v, am, out_file, directed=False):
    graph_id = os.path.splitext(os.path.basename(out_file))[0]
    with open(out_file, 'w') as file_object:
        file_object.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        file_object.write('<!DOCTYPE gxl SYSTEM "http://www.gupro.de/GXL/gxl-1.0.dtd">\n')
        file_object.write('<gxl xmlns:xlink="http://www.w3.org/1999/xlink">\n')

        if directed:
            file_object.write('\t<graph id="' + graph_id + '" edgeids="false" edgemode="directed">\n')
        else:
            file_object.write('\t<graph id="' + graph_id + '" edgeids="false" edgemode="undirected">\n')

        # Node
        for i in range(v.size(0)):
            file_object.write('\t\t<node id="_' + str(i) + '">\n')
            file_object.write('\t\t\t<attr name="hidden_state">\n')
            file_object.write('\t\t\t\t<vector>')
            file_object.write(str(v[i, 0]))
            for j in range(1, v.size(1)):
                file_object.write(', ' + str(v[i, j]))
            file_object.write('</vector>\n')
            file_object.write('\t\t\t</attr>\n')
            file_object.write('\t\t</node>\n')
        # Edge
        for i in range(v.size(0)):
            start_j = 0 if directed else i
            for j in range(start_j, v.size(0)):
                if am[i, j, :].abs().sum() != 0:
                    if directed:
                        file_object.write('\t\t<edge from="_' + str(i) + '" to="_' + str(j) + '">\n')
                        file_object.write('\t\t\t<attr name="hidden_state">\n')
                        file_object.write('\t\t\t\t<vector>')
                        file_object.write(str(am[i, j, 0]))
                        for k in range(1, am.size(2)):
                            file_object.write(', ' + str(am[i, j, k]))
                        file_object.write('</vector>\n')
                        file_object.write('\t\t\t</attr>\n')
                        file_object.write('\t\t</edge>\n')
                    else:
                        file_object.write('\t\t<edge from="_' + str(i) + '" to="_' + str(j) + '"/>\n')
        file_object.write('\t</graph>\n')
        file_object.write('</gxl>\n')

def roc_auc(output, target):
    import numpy as np
    output = np.array(output)
    target = np.array(target)
    output = np.resize(output, output.shape[0:2])
    target = np.resize(target, target.shape[0:2])
    n_classes = output.shape[1]

    for i in range(len(output)):
        for j in range(output.shape[1]):
            output[i, j] = output[i, j].data[0]
    for i in range(len(target)):
        for j in range(target.shape[1]):
            target[i, j] = target[i, j].data[0]
    target_one_hot = np.zeros((output.shape[0], output.shape[1]), dtype=int)
    for i in range(output.shape[0]):
        target_one_hot[i,target[i,0]] = 1
    target = target_one_hot.astype(int)
    output = output.astype(int)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(target[:, i], output[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(target.ravel(), output.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)
    lw = 2
    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Some extension of Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")
    plt.savefig('roc_auc.png')


def build_confusion_matrix(output, target):
    import numpy as np
    output = np.array(output)
    target = np.array(target)
    output = np.resize(output, output.shape[0:2])
    target = np.resize(target, target.shape[0:2])

    for i in range(len(output)):
        for j in range(output.shape[1]):
            output[i, j] = output[i, j].data[0]
    for i in range(len(target)):
        for j in range(target.shape[1]):
            target[i, j] = target[i, j].data[0]
    output_class = np.zeros((output.shape[0],1), dtype=int)
    print('cm')
    for i in range(output.shape[0]):
        output_class[i] = np.argmax(output[i])
    target = target[:,0].astype(int)
    output_class = output_class[:,0]
    classes = [str(cl) for cl in range(output.shape[1])]
    plot_confusion_matrix(target, output_class, classes=classes)
    plt.savefig('confusion_matrix.png')


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    from sklearn.utils.multiclass import unique_labels
    #classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    plt.figure(figsize=(20,20))
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


def accuracy(output, target):
    return precision_at_k(output, target, topk=(1,))


def precision_at_k(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    pred = pred.type_as(target)
    target = target.type_as(pred)
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def siamese_accuracy(output, target):
    batch_size = target.size(0)

    pred = (output < 0.5).float()
    correct = pred.eq(target).float()
    acc = 100.0 * correct.sum() / batch_size
    return acc


def knn(D, target, train_target, k=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(k)
    batch_size = target.size(0)
    _, pred = D.topk(maxk, largest=False, sorted=True)
    pred = train_target[pred]
    pred = pred.type_as(target)

    res = []
    for ki in k:
        pred_k = nn_prediction(pred[:ki], axis=0)

        pred_k = pred_k.squeeze()
        correct_k = pred_k.eq(target.data).float().sum()

        res.append(correct_k * (100.0 / batch_size))
    return torch.FloatTensor(res)


def meanAveragePrecision(D, target, train_target):
    y_true = train_target == target
    y_score = (1 + D.max()) - D
    if y_true.data.float().sum() == 0:
        return 1.0
    return average_precision_score(y_true.data.cpu().numpy(), y_score.data.cpu().numpy())


def nn_prediction(pred, axis=1):
    scores = torch.from_numpy(np.unique(np.ravel(pred.data.cpu().numpy())))

    testshape = list(pred.size())
    testshape[axis] = 1

    mostfrequent = torch.zeros(testshape)
    mostindex = torch.zeros(testshape).long()
    oldcounts = torch.zeros(testshape)

    if pred.is_cuda:
        mostfrequent = mostfrequent.cuda()
        mostindex = mostindex.cuda()
        oldcounts = oldcounts.cuda()

    for score in scores:
        template = (pred == score).data
        counts = template.float().sum(axis, keepdim=True)

        ind = torch.arange(0, pred.size(0)).expand_as(template)
        if pred.is_cuda:
            ind = ind.cuda()
        ind = ind * template.float() + pred.size(0) * (1 - template.float())
        _, ind = ind.min(0, keepdim=True)

        mostfrequent[(counts > oldcounts) | ((counts == oldcounts) & (ind < mostindex))] = score
        mostindex[(counts > oldcounts) | ((counts == oldcounts) & (ind < mostindex))] = ind[
            (counts > oldcounts) | ((counts == oldcounts) & (ind < mostindex))]

        oldcounts, _ = torch.max(torch.cat([oldcounts.unsqueeze(0), counts.unsqueeze(0)], 1), 1, keepdim=False)

    return mostfrequent.long()


def plot_letters(numpy_all, numpy_labels):
    c = ['#ff0000', '#ffff00', '#00ff00', '#00ffff', '#0000ff',
         '#ff00ff', '#990000', '#999900', '#009900', '#009999', '#0ffff0', '#000099', '#990099', '#099990', '#999999']

    for i in range(14):
        f = numpy_all[np.where(numpy_labels == i)]
        plt.plot(f[:, 0], f[:, 1], '.', c=c[i])
    plt.legend(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14'])
    plt.savefig('result.png')
