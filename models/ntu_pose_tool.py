import os

import numpy as np
import operator
import json

from keras.callbacks import Callback

from models.measures import mean_distance_error
from data_generator.io_operate import *
from data_generator.transform import transform_pose_sequence
from data_generator.pose import *
def eval_ntu_sc_error(model, x, pose, afmat, rootz, action,
        resol_z=2000., batch_size=8, map_to_pa20j=None, logdir=None,
        verbose=True):

    from data_generator.ntu import ACTION_LABELS

    assert len(x) == len(pose) == len(afmat)  == len(action)

    input_shape = model.input_shape
    if len(input_shape) == 5:
        """Video clip processing."""
        num_frames = input_shape[1]
        num_batches = int(len(x) / num_frames)

        x = x[0:num_batches*num_frames]
        x = np.reshape(x, (num_batches, num_frames,) + x.shape[1:])

        pose = pose[0:num_batches*num_frames]
        afmat = afmat[0:num_batches*num_frames]
        rootz = rootz[0:num_batches*num_frames]
        action = action[0:num_batches*num_frames]

    num_blocks = len(model.outputs)
    num_spl = len(x)

    y_true = pose.copy()
    if map_to_pa20j is not None:
        y_true = y_true[:, map_to_pa20j, :]
    y_true[:,:,0:2] = transform_pose_sequence(afmat.copy(), y_true[:, :, 0:2], inverse=True)
    y_true[:, :, 2] = (resol_z * y_true[:, :, 2]) + rootz

    pred = model.predict(x, batch_size=batch_size, verbose=1)
    y_pred_w = []
    """Move the root joints from g.t. poses to origin."""

    if verbose:
        printc(WARNING, 'Avg. mm. error:')

    lower_err = np.inf
    lower_i = -1
    scores = []

    for b in range(num_blocks):

        if num_blocks > 1:
            y_pred = pred[b]
        else:
            y_pred = pred

        if len(input_shape) == 5:
            """Remove the temporal dimension."""
            y_pred = y_pred[:, :, :, 0:3]
            y_pred = np.reshape(y_pred, (-1, y_pred.shape[2], y_pred.shape[3]))
        else:
            y_pred = y_pred[:, :, 0:3]

        """Project normalized coordiates to the image plane."""
        y_pred[:, :, 0:2] = transform_pose_sequence(
            afmat.copy(), y_pred[:, :, 0:2], inverse=True)

        """Recover the absolute Z."""
        y_pred[:, :, 2] = (resol_z * y_pred[:, :, 2]) + rootz
        y_pred_w.append(y_pred)
        err = mean_distance_error(y_true, y_pred)
        scores.append(err)
        if verbose:
            printc(WARNING, ' %.1f' % err)

        """Keep the best prediction and its index."""
        if err < lower_err:
            lower_err = err
            lower_i = b
        
    if verbose:
        printcn('', '')

    if logdir is not None:
        np.save('%s/y_pred.npy' % logdir, y_pred)
        np.save('%s/y_true.npy' % logdir, y_true)

    """Select only the best prediction."""
    y_pred = y_pred_w[lower_i]

    """Compute error per action."""
    num_act = len(ACTION_LABELS)
    y_pred_act = {}
    y_true_act = {}
    for i in range(num_act):
        y_pred_act[i] = None
        y_true_act[i] = None

    act = lambda x: action[x].nonzero()[0][0]
    for i in range(len(y_pred)):
        if y_pred_act[act(i)] is None:
            y_pred_act[act(i)] = np.expand_dims(y_pred[i],axis = 0)
            y_true_act[act(i)] = np.expand_dims(y_true[i],axis = 0)
        else:
            y_pred_act[act(i)] = np.concatenate(
                    [y_pred_act[act(i)], np.expand_dims(y_pred[i],axis = 0)], axis=0)
            y_true_act[act(i)] = np.concatenate(
                    [y_true_act[act(i)], np.expand_dims(y_true[i],axis = 0)], axis=0)

    for i in range(num_act):
        if y_pred_act[i] is None:
            continue
        err = mean_distance_error(y_true_act[i], y_pred_act[i])
        printcn(OKBLUE, '%s: %.1f' % (ACTION_LABELS[i], err))

    printcn(WARNING, 'Final averaged error (mm): %.3f' % lower_err)

    return scores


class NTU_POSE_EvalCallback(Callback):

    def __init__(self, x, p, afmat, rootz, action, batch_size=24,
            eval_model=None, map_to_pa20j=None, logdir=None):

        self.x = x
        self.p = p[:,:,0:3]
        self.afmat = afmat
        self.rootz = rootz
        self.action = action
        self.batch_size = batch_size
        self.eval_model = eval_model
        self.map_to_pa20j = map_to_pa20j
        self.scores = {}
        self.logdir = logdir

    def on_epoch_end(self, epoch, logs={}):
        if self.eval_model is not None:
            model = self.eval_model
        else:
            model = self.model

        scores = eval_ntu_sc_error(model, self.x, self.p, self.afmat,
                self.rootz, self.action, batch_size=self.batch_size,
                map_to_pa20j=self.map_to_pa20j)

        epoch += 1
        if self.logdir is not None:
            if not hasattr(self, 'logarray'):
                self.logarray = {}
            self.logarray[epoch] = scores
            with open(os.path.join(self.logdir, 'ntu_pose_val.json'), 'w') as f:
                json.dump(self.logarray, f)

        cur_best = min(scores)
        self.scores[epoch] = cur_best

        printcn(OKBLUE, 'Best score is %.1f at epoch %d' % \
                (self.best_score, self.best_epoch))


    @property
    def best_epoch(self):
        if len(self.scores) > 0:
            # Get the key of the minimum value from a dict
            return min(self.scores, key=self.scores.get)
        else:
            return np.inf

    @property
    def best_score(self):
        if len(self.scores) > 0:
            # Get the minimum value from a dict
            return self.scores[self.best_epoch]
        else:
            return np.inf

