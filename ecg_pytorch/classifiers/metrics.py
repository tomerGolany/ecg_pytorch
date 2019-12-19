"""Evaluation metrics for ECG classification models."""
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import numpy as np
import logging
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import confusion_matrix
from bokeh.plotting import figure, output_file, show, ColumnDataSource
import os
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score


def plt_roc_curve(y_true, y_pred, classes, writer, total_iters):
    """Calculate ROC curve from predictions and ground truths.

    writes the roc curve into tensorboard writer object.

    :param y_true:[[1,0,0,0,0], [0,1,0,0], [1,0,0,0,0],...]
    :param y_pred: [0.34,0.2,0.1] , 0.2,...]
    :param classes:5
    :param writer: tensorboardX summary writer.
    :param total_iters: total number of training iterations when the predictions where generated.
    :return: List of area-under-curve (AUC) of the ROC curve.
    """
    logging.info("Entering plot_roc_curve function with params: y_true shape = {}. y_pred shape = {}.".format(
        y_true.shape, y_pred.shape))
    fpr = {}
    tpr = {}
    roc_auc = {}
    roc_auc_res = []
    n_classes = len(classes)
    for i in range(n_classes):
        fpr[classes[i]], tpr[classes[i]], _ = roc_curve(y_true[:, i], y_pred[:, i])
        roc_auc[classes[i]] = auc(fpr[classes[i]], tpr[classes[i]])
        roc_auc_res.append(roc_auc[classes[i]])
        fig = plt.figure()
        lw = 2
        plt.plot(fpr[classes[i]], tpr[classes[i]], color='darkorange',
                 lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[classes[i]])
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic beat {}'.format(classes[i]))
        plt.legend(loc="lower right")
        writer.add_figure('test/roc_curve_beat_{}'.format(classes[i]), fig, total_iters)
        plt.close()
        fig.clf()
        fig.clear()
    return roc_auc_res


def add_roc_curve_pure_tensorboard(y_true, y_pred, classes, writer, total_iters):
    # TODO: feature not supported.
    fpr = {}
    tpr = {}
    roc_auc = {}
    roc_auc_res = []
    n_classes = len(classes)
    for i in range(n_classes):
        fpr[classes[i]], tpr[classes[i]], probabilities = roc_curve(y_true[:, i], y_pred[:, i])
        roc_auc[classes[i]] = auc(fpr[classes[i]], tpr[classes[i]])
        roc_auc_res.append(roc_auc[classes[i]])

        fpr_i = fpr[classes[i]]
        tpr_i = tpr[classes[i]]

        for fp, tp, prob in zip(fpr_i, tpr_i, probabilities):
            writer.add_scalars('ROC_Curve_at_step_{}_beat_{}'.format(total_iters, classes[i]), {'roc_curve': tp}, fp)


def add_roc_curve_bokeh(y_true, y_pred, classes, model_dir, epoch):
    fpr = {}
    tpr = {}
    roc_auc = {}
    roc_auc_res = []
    n_classes = len(classes)
    for i in range(n_classes):
        fpr[classes[i]], tpr[classes[i]], probabilities = roc_curve(y_true[:, i], y_pred[:, i])
        roc_auc[classes[i]] = auc(fpr[classes[i]], tpr[classes[i]])
        roc_auc_res.append(roc_auc[classes[i]])

        fpr_i = fpr[classes[i]]
        tpr_i = tpr[classes[i]]
        tnr_i = [1 - x for x in fpr_i]
        output_file(os.path.join(model_dir, "roc_curve_{}_epoch_{}.html".format(classes[i], epoch)))

        source = ColumnDataSource(data=dict(
            fpr=fpr_i,
            tpr=tpr_i,
            tnr=tnr_i,
            probs=probabilities,
        ))

        TOOLTIPS = [
            ("(fpr ,se.(tpr))", "($x, $y)"),
            ("threshold", "@probs"),
            ("spe.(tnr)", "@tnr")
        ]

        p = figure(plot_width=400, plot_height=400, tooltips=TOOLTIPS,
                   title='Receiver operating characteristic beat {}'.format(classes[i]),
                   x_axis_label='False Positive Rate', y_axis_label='True Positive Rate')

        p.line('fpr', 'tpr', source=source)
        show(p)


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """This function prints and plots the confusion matrix.

    Normalization can be applied by setting `normalize=True`.

    :return: Figure which contains confusion matrix.
    """
    logging.info("Entering plot confusion matrix...")
    logging.info("inputs params: y_true shape: {}, y_pred shape: {}".format(y_true.shape, y_pred.shape))
    logging.info("First element ground truth: {}. First element predictions: {}".format(y_true[0], y_pred[0]))
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    normalize = True
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        logging.info("Normalized confusion matrix")
    else:
        logging.info('Confusion matrix, without normalization')

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
    return fig, ax


def plt_precision_recall_curve(y_true, y_pred, classes, writer, total_iters):
    # For each class
    precision = dict()
    recall = dict()
    n_classes = len(classes)
    # average_precision = dict()
    for i in range(n_classes):
        precision[classes[i]], recall[classes[i]], _ = precision_recall_curve(y_true[:, i],
                                                            y_pred[:, i])

        fig = plt.figure()
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title('Precision Recall graph')
        plt.plot(recall[classes[i]], precision[classes[i]])
        writer.add_figure('test/precision_recall_beat_{}'.format(classes[i]), fig, total_iters)
        plt.close()
        fig.clf()
        fig.clear()

        # average_precision[classes[i]] = average_precision_score(y_true[:, i], y_pred[:, i])

    # A "micro-average": quantifying score on all classes jointly
    # precision["micro"], recall["micro"], _ = precision_recall_curve(Y_test.ravel(),
    #                                                                 y_score.ravel())
    # average_precision["micro"] = average_precision_score(Y_test, y_score,
    #                                                      average="micro")
    # print('Average precision score, micro-averaged over all classes: {0:0.2f}'
    #       .format(average_precision["micro"]))


def plt_precision_recall_bokeh(y_true, y_pred, classes, model_dir, epoch):
    # For each class
    precision = dict()
    recall = dict()
    n_classes = len(classes)
    # average_precision = dict()
    for i in range(n_classes):
        precision[classes[i]], recall[classes[i]], probs = precision_recall_curve(y_true[:, i],
                                                                              y_pred[:, i])

        pre_i = precision[classes[i]]
        rec_i = recall[classes[i]]

        output_file(os.path.join(model_dir, "precision_recall_{}_epoch_{}.html".format(classes[i], epoch)))
        logging.info("saving in {}".format(os.path.join(model_dir, "precision_recall_{}.html".format(classes[i]))))
        source = ColumnDataSource(data=dict(
            prec=pre_i,
            rec=rec_i,
            probs=probs,
        ))

        TOOLTIPS = [
            ("(se.(rec),ppv(prec))", "($x, $y)"),
            ("threshold", "@probs"),
        ]

        p = figure(plot_width=400, plot_height=400, tooltips=TOOLTIPS,
                   title='precision recall beat {}'.format(classes[i]),
                   x_axis_label='Recall', y_axis_label='Precision')

        p.line('rec', 'prec', source=source)

        show(p)
