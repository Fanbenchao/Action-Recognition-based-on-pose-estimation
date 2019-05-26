import os

import numpy as np
import json
import time

from keras.callbacks import Callback

from data_generator.loader import BatchLoader
from data_generator.io_operate import *
from data_generator.ntu import ACTION_LABELS

def eval_singleclip_gt_bbox_generator(model, datagen, num_actions, verbose=1):

    num_blocks = len(model.outputs)
    num_samples = len(datagen)

    start = time.time()
    for i in range(num_samples):
        if verbose:
            printcn('', 'pred %05d/%05d' % (i+1, num_samples))

        [x], [y] = datagen[i]
        if 'y_true' not in locals():
            y_true = np.zeros((num_samples,) + y.shape[1:])
            y_pred = np.zeros((num_samples, num_blocks) + y.shape[1:])

        y_true[i, :] = y
        pred = model.predict(x)
        for b in range(num_blocks):
            y_pred[i, b, :] = pred[b]

    dt = time.time() - start

    if verbose:
        printc(WARNING, 'NTU, single-clip, GT bbox, action acc.%:')
    higher_score = -np.inf
    higher_i = -1
    scores = []
    for b in range(num_blocks):
        correct = np.equal(np.argmax(y_true, axis=-1),
                np.argmax(y_pred[:, b, :], axis=-1), dtype=np.float)
        scores.append(sum(correct) / len(correct))
        if verbose:
            printc(WARNING, ' %.5f ' % (100*scores[-1]))
        if scores[b] >= higher_score:
            higher_score = scores[b]
            higher_i = b
    y_pred_block = y_pred[:,higher_i,:]
    action_scores = {}
    y_pred_act = {}
    y_true_act = {}
    for i in range(num_actions):
        y_pred_act[i] = None
        y_true_act[i] = None
    
    act = lambda x: y_true[x].nonzero()[0][0]
    for i in range(num_samples):
        if y_pred_act[act(i)] is None:
            y_pred_act[act(i)] = np.expand_dims(y_pred_block[i],axis = 0)
            y_true_act[act(i)] = np.expand_dims(y_true[i],axis = 0)
        else:
            y_pred_act[act(i)] = np.concatenate(\
                    [y_pred_act[act(i)], np.expand_dims(y_pred_block[i],axis = 0)], axis=0)
            y_true_act[act(i)] = np.concatenate(\
                    [y_true_act[act(i)], np.expand_dims(y_true[i],axis = 0)], axis=0)
    for i in range(num_actions):
        if y_pred_act[i] is None:
            continue
        correct = np.equal(np.argmax(y_true_act[i], axis=-1),
                np.argmax(y_pred_act[i], axis=-1), dtype=np.float)
        action_score = sum(correct) / len(correct)
        action_scores[ACTION_LABELS[i]] = action_score
        printcn(OKBLUE, '%s: %.5f' % (ACTION_LABELS[i], action_score))
        
    if verbose:
        printcn('', '\n%d samples in %.1f sec: %.1f clips per sec' \
                % (num_samples, dt, num_samples / dt))
        
    return scores,action_scores


class NtuEvalCallback(Callback):

    def __init__(self, data,num_actions, eval_model=None, logdir=None):

        assert type(data) == BatchLoader, \
                'data must be a BatchLoader instance, ' \
                + 'got {} instead'.format(data)

        self.data = data
        self.eval_model = eval_model
        self.scores = {}
        self.blocks = {}
        self.logdir = logdir
        self.num_actions = num_actions
    def on_epoch_end(self, epoch, logs={}):
        if self.eval_model is not None:
            model = self.eval_model
        else:
            model = self.model

        scores,action_scores= eval_singleclip_gt_bbox_generator(model, self.data,self.num_actions)

        epoch += 1
        if self.logdir is not None:
            if not hasattr(self, 'logarray'):
                self.logarray = {}
            self.logarray[epoch] = [scores,action_scores]
            with open(os.path.join(self.logdir, 'ntu_val.json'), 'w') as f:
                json.dump(self.logarray, f)

        cur_best = max(scores)
        block_best = np.argmax(scores)+1
        self.scores[epoch] = cur_best
        self.blocks[epoch] = block_best
        printcn(OKBLUE, 'Best score is %.1f in blocks %d at epoch %d' % \
                (100*self.best_score,self.best_block,self.best_epoch))

    @property
    def best_epoch(self):
        if len(self.scores) > 0:
            # Get the key of the maximum value from a dict
            #return max(self.scores, key=self.scores.get)
            return max(self.scores)
        else:
            return np.inf

    @property
    def best_score(self):
        if len(self.scores) > 0:
            # Get the maximum value from a dict
            return self.scores[self.best_epoch]
        else:
            return 0
    @property
    def best_block(self):
        if len(self.blocks) > 0:
            # Get the maximum value from a dict
            return self.blocks[self.best_epoch]
        else:
            return 0
# Aliases.
eval_singleclip_generator = eval_singleclip_gt_bbox_generator
