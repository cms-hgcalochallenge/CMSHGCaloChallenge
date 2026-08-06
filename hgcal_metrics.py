import argparse
import json
import os
import warnings
import random
import copy

import h5py
import jetnet
import numpy as np
import torch
import torch.utils.data as torchdata
from sklearn.calibration import calibration_curve
from sklearn.decomposition import PCA
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from scipy.stats import kstest

import utils
from plotting.plotting_utils import make_hist, make_profile

#set random seed for everything
def set_seed(seed = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Essential for reproducibility, but may impact performance
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def truncate_layer_features(feats_cls, n_layers, max_layer=30):
    """Return feats_cls with per-layer features for layer >= max_layer removed.

    Expected feats_cls column layout:
      [0]                       incident_E          (scalar)
      [1]                       E_ratio             (scalar)
      [2 : 2+n_layers]          E_x_center          (per layer)
      [2+n_layers : 2+2*n_l]    E_x_width           (per layer)
      [2+2*n_l : 2+3*n_l]       E_y_center          (per layer)
      [2+3*n_l : 2+4*n_l]       E_y_width           (per layer)
      [2+4*n_l : 2+5*n_l]       layer_occupancy     (per layer)
      [2+5*n_l : 2+6*n_l]       longitudinal profile(per layer)
      [2+6*n_l : ]              transverse profile  (per ring, kept in full)

    Args:
        feats_cls:  (N, D) combined feature array.
        n_layers:   Total number of calorimeter layers.
        max_layer:  Only keep layers [0, max_layer); drop layer >= max_layer.

    Returns:
        (N, D') array with the high-layer columns removed.
    """
    n_per_layer_blocks = 6  # 5 from compute_feats + 1 longitudinal profile
    keep = np.ones(feats_cls.shape[1], dtype=bool)
    for b in range(n_per_layer_blocks):
        drop_start = 2 + b * n_layers + max_layer
        drop_end   = 2 + (b + 1) * n_layers
        keep[drop_start:drop_end] = False
    n_removed = feats_cls.shape[1] - keep.sum()
    #print(f"Layer truncation: keeping layers 0-{max_layer-1}, "
    #      f"removed {n_removed} columns, {keep.sum()} remaining.")
    return feats_cls[:, keep]


def pca_reduce(feats_reference, feats_model, n_components=50):
    """Fit PCA on the reference (Geant4) sample and project both samples.

    The scaler and PCA are fitted exclusively on *feats_reference* so that
    the projection basis is independent of the model being evaluated.

    Args:
        feats_reference: (N_ref, D) array of reference (Geant4) features.
        feats_model:     (N_mod, D) array of model/generated features.
        n_components:    Number of principal components to retain (default 50).

    Returns:
        ref_pca:  (N_ref, n_components) projected reference features.
        model_pca:(N_mod, n_components) projected model features.
    """
    scaler = StandardScaler()
    ref_scaled   = scaler.fit_transform(feats_reference)
    model_scaled = scaler.transform(feats_model)

    # Fit full PCA to get complete variance spectrum for diagnostics
    pca_full = PCA()
    pca_full.fit(ref_scaled)

    cumvar = np.cumsum(pca_full.explained_variance_ratio_)
    for thresh in (0.90, 0.95, 0.99):
        n = np.searchsorted(cumvar, thresh) + 1
        print(f"  Components for {thresh*100:.0f}% variance: {n}")
    explained_n = cumvar[n_components - 1]
    print(f"PCA: top {n_components} components explain {explained_n*100:.2f}% of variance")

    # Project using only the top n_components eigenvectors
    components = pca_full.components_[:n_components]
    ref_pca   = ref_scaled   @ components.T
    model_pca = model_scaled @ components.T
    return ref_pca, model_pca


def _map_weights_to_layers(weights, n_layers, key_name):
    if len(weights) == n_layers + 1:
        mapped_weights = weights[1:]
    elif len(weights) == n_layers:
        mapped_weights = weights
    else:
        raise ValueError(
            "%s length %i does not match expected %i or %i"
            % (key_name, len(weights), n_layers, n_layers + 1)
        )
    return np.asarray(mapped_weights, dtype=np.float32)


def load_layer_weights_from_json(weights_file, key_name, n_layers):
    """Load layer weights from JSON and map to shower layer indexing."""
    if not os.path.exists(weights_file):
        raise OSError("Layer-weights JSON not found: %s" % weights_file)

    with open(weights_file, "r") as handle:
        payload = json.load(handle)

    weights = payload.get(key_name)
    if weights is None:
        raise ValueError("Missing `%s` in %s" % (key_name, weights_file))
    return _map_weights_to_layers(weights, n_layers, key_name)


def train_and_evaluate_cls(model, data_train, data_test, optim, arg):
    """ train the model and evaluate along the way"""
    best_eval_acc = float('-inf')
    arg.best_epoch = -1
    try:
        for i in range(arg.cls_n_epochs):
            train_cls(model, data_train, optim, i, arg)
            with torch.no_grad():
                eval_acc, _, _, _ = evaluate_cls(model, data_test, arg)
            if eval_acc > best_eval_acc:
                best_eval_acc = eval_acc
                arg.best_epoch = i+1
                #filename = arg.mode + '_' + arg.dataset + '.pt'
                #torch.save({'model_state_dict':model.state_dict()},
                           #os.path.join(arg.output_dir, filename))
            if eval_acc == 1.:
                break
    except KeyboardInterrupt:
        # training can be cut short with ctrl+c, for example if overfitting between train/test set
        # is clearly visible
        pass
    return model

def train_cls(model, data_train, optim, epoch, arg):
    """ train one step """
    model.train()
    for i, data_batch in enumerate(data_train):
        if arg.save_mem:
            data_batch = data_batch[0].to(arg.device)
        else:
            data_batch = data_batch[0]
        #input_vector, target_vector = torch.split(data_batch, [data_batch.size()[1]-1, 1], dim=1)
        input_vector, target_vector = data_batch[:, :-1], data_batch[:, -1]
        output_vector = model(input_vector)
        criterion = torch.nn.BCEWithLogitsLoss()
        loss = criterion(output_vector, target_vector.unsqueeze(1))

        optim.zero_grad()
        loss.backward()
        optim.step()

        if i % (len(data_train)//2) == 0:
            print('Epoch {:3d} / {}, step {:4d} / {}; loss {:.4f}'.format(
                epoch+1, arg.cls_n_epochs, i, len(data_train), loss.item()))
        # PREDICTIONS
        pred = torch.round(torch.sigmoid(output_vector.detach()))
        target = torch.round(target_vector.detach())
        if i == 0:
            res_true = target
            res_pred = pred
        else:
            res_true = torch.cat((res_true, target), 0)
            res_pred = torch.cat((res_pred, pred), 0)

    try:
        print("Accuracy on training set is",
          accuracy_score(res_true.cpu(), np.clip(res_pred.cpu(), 0., 1.0)))
    except:
        print("Nans")

def evaluate_cls(model, data_test, arg, final_eval=False, calibration_data=None):
    """ evaluate on test set.

    Returns (eval_acc, eval_auc, scores, labels), where `scores` is the
    continuous predicted probability per event (calibrated when
    final_eval=True) and `labels` is the true 0/1 array. Keeping the
    continuous probability (rather than rounding to 0/1 before scoring)
    is required for a true ROC AUC and for any downstream use of the
    score as a density-ratio estimate (e.g. OOD fraction, reweighting).
    """
    model.eval()
    for j, data_batch in enumerate(data_test):
        if arg.save_mem:
            data_batch = data_batch[0].to(arg.device)
        else:
            data_batch = data_batch[0]
        input_vector, target_vector = data_batch[:, :-1], data_batch[:, -1]
        output_vector = model(input_vector)
        pred = output_vector.reshape(-1)
        target = target_vector.double()
        if j == 0:
            result_true = target
            result_pred = pred
        else:
            result_true = torch.cat((result_true, target), 0)
            result_pred = torch.cat((result_pred, pred), 0)
    BCE = torch.nn.BCEWithLogitsLoss()(result_pred, result_true)
    prob_pred = torch.sigmoid(result_pred).cpu().numpy()
    result_true = result_true.cpu().numpy().astype(np.float32)
    hard_pred = np.clip(np.round(prob_pred), 0., 1.0)
    try:
        eval_acc = accuracy_score(result_true, hard_pred)
    except:
        print("Nans")
        prob_pred[np.isnan(prob_pred)] = 0.5
        hard_pred = np.clip(np.round(prob_pred), 0., 1.0)
        eval_acc = accuracy_score(result_true, hard_pred)
    print("Accuracy on test set is", eval_acc)
    eval_auc = roc_auc_score(result_true, prob_pred)
    print("AUC on test set is", eval_auc)
    scores = prob_pred
    if final_eval:
        prob_true, prob_pred_curve = calibration_curve(result_true, prob_pred, n_bins=10)
        print("unrescaled calibration curve:", prob_true, prob_pred_curve)
        calibrator = calibrate_classifier(model, calibration_data, arg)
        rescaled_pred = calibrator.predict(prob_pred)
        eval_acc = accuracy_score(result_true, np.clip(np.round(rescaled_pred), 0., 1.0))
        print("Rescaled accuracy is", eval_acc)
        eval_auc = roc_auc_score(result_true, rescaled_pred)
        scores = rescaled_pred
    return eval_acc, eval_auc, scores, result_true

def calibrate_classifier(model, calibration_data, arg):
    """ reads in calibration data and performs a calibration with isotonic regression"""
    model.eval()
    assert calibration_data is not None, ("Need calibration data for calibration!")
    for j, data_batch in enumerate(calibration_data):
        if arg.save_mem:
            data_batch = data_batch[0].to(arg.device)
        else:
            data_batch = data_batch[0]
        input_vector, target_vector = data_batch[:, :-1], data_batch[:, -1]
        output_vector = model(input_vector)
        pred = torch.sigmoid(output_vector).reshape(-1)
        target = target_vector.to(torch.float64)
        if j == 0:
            result_true = target
            result_pred = pred
        else:
            result_true = torch.cat((result_true, target), 0)
            result_pred = torch.cat((result_pred, pred), 0)
    result_true = result_true.cpu().numpy()
    result_pred = result_pred.cpu().numpy()
    iso_reg = IsotonicRegression(out_of_bounds='clip', y_min=1e-6, y_max=1.-1e-6).fit(result_pred,
                                                                                      result_true)
    return iso_reg


class DNN(torch.nn.Module):
    """ NN for vanilla classifier. Does not have sigmoid activation in last layer, should
        be used with torch.nn.BCEWithLogitsLoss()
    """
    def __init__(self, num_layer, num_hidden, input_dim, dropout_probability=0.):
        super(DNN, self).__init__()

        self.dpo = dropout_probability

        self.inputlayer = torch.nn.Linear(input_dim, num_hidden)
        self.outputlayer = torch.nn.Linear(num_hidden, 1)

        all_layers = [self.inputlayer, torch.nn.LeakyReLU(), torch.nn.Dropout(self.dpo)]
        for _ in range(num_layer):
            all_layers.append(torch.nn.Linear(num_hidden, num_hidden))
            all_layers.append(torch.nn.LeakyReLU())
            all_layers.append(torch.nn.Dropout(self.dpo))

        all_layers.append(self.outputlayer)
        self.layers = torch.nn.Sequential(*all_layers)

    def forward(self, x):
        """ Forward pass through the DNN """
        x = self.layers(x)
        return x

def get_feat_names(nLayers):
    feat_names = ['Incident Energy', 'Energy Ratio']
    # removed: redundant with longitudinal profile (fraction or absolute via --use_absolute)
    #for i in range(nLayers): feat_names.append("Log Energy Layer %i" % i)
    for i in range(nLayers): feat_names.append("X Center Layer %i" % i)
    for i in range(nLayers): feat_names.append("X Width Layer %i" % i)
    for i in range(nLayers): feat_names.append("Y Center Layer %i" % i)
    for i in range(nLayers): feat_names.append("Y Width Layer %i" % i)
    for i in range(nLayers): feat_names.append("Occupancy Layer %i" % i)

    return feat_names


def compute_profiles(showers, geom, n_rings):
    """Compute longitudinal and transverse shower profiles.

    Callers must pass showers that are already free of non-finite voxels
    (see the finite-mask filter in LoadSample); this keeps the profile row
    count identical to that of compute_feats on the same showers.
    """
    eps = 1e-8

    E_total = np.sum(showers, axis=(1,2)).reshape(showers.shape[0], 1)
    E_layer = np.sum(showers, axis=(2))
    longitudinal_profile = E_layer / (E_total + eps)

    ring_vals = geom.ring_map[:, :geom.max_ncell]
    ring_vals = np.where(ring_vals >= 0, ring_vals, -1).astype(np.int32)
    ring_onehot = (ring_vals[None, :, :] == np.arange(n_rings)[:, None, None]).astype(np.float32)
    ring_energies = np.einsum('rlc,slc->sr', ring_onehot, showers)
    transverse_profile = ring_energies / (E_total + eps)

    return longitudinal_profile.astype(np.float32), transverse_profile.astype(np.float32)


def compute_feats(showers, incident_E, geom):

    eps = 1e-8
    E_total = np.sum(showers, axis=(1,2)).reshape(showers.shape[0], 1)
    E_ratio = E_total / incident_E
    # removed: redundant with longitudinal profile (fraction or absolute via --use_absolute)
    #E_layer = np.sum(showers, axis=(2))
    #E_per_layer = np.log10(E_layer + eps)
    x_vals = geom.xmap[:, :geom.max_ncell]
    E_x_center = utils.WeightedMean(x_vals, showers, axis=(2))
    E_x2_center = utils.WeightedMean(x_vals, showers, power=2, axis=(2))
    E_x_width = utils.GetWidth(E_x_center, E_x2_center)

    y_vals = geom.ymap[:, :geom.max_ncell]
    E_y_center = utils.WeightedMean(y_vals, showers, axis=(2))
    E_y2_center = utils.WeightedMean(y_vals, showers, power=2, axis=(2))
    E_y_width = utils.GetWidth(E_y_center, E_y2_center)

    layer_voxels = np.reshape(showers,(showers.shape[0],showers.shape[1],-1))
    layer_occupancy = np.sum(layer_voxels > eps, axis = -1)

    feats = np.concatenate(
        [
            incident_E,
            E_ratio,
            #E_per_layer,  # removed: redundant with profiles
            E_x_center,
            E_x_width,
            E_y_center,
            E_y_width,
            layer_occupancy,
        ],
        axis=-1,
    ).astype(np.float32)

    return feats


def ttv_split(data1, split=np.array([0.7, 0.2, 0.1])):
    """ splits data1 and data2 in train/test/val according to split,
        returns shuffled and merged arrays
    """
    num_events = (len(data1) * split).astype(int)
    np.random.shuffle(data1)
    train1, test1, val1 = np.split(data1, num_events.cumsum()[:-1])
    np.random.shuffle(train1)
    np.random.shuffle(test1)
    np.random.shuffle(val1)
    return train1, test1, val1


def compute_metrics(flags):

    utils.SetStyle()
    nevts = int(flags.nevts)
    dataset_config = utils.LoadJson(flags.config)
    geom_file = dataset_config.get('BIN_FILE', '')
    dataset_num = dataset_config.get('DATASET_NUM', 2)
    hgcal = dataset_config.get('HGCAL', False)
    max_cells = dataset_config.get('MAX_CELLS', None)
    max_layer_eval = dataset_config.get('MAX_LAYER_EVAL', None)

    if(torch.cuda.is_available()): device = torch.device('cuda')
    else: device = torch.device('cpu')
    flags.device = device


    geom = utils.load_geom(geom_file)


    shape_plot = dataset_config['SHAPE_ORIG']
    valid_rings = geom.ring_map[:, :geom.max_ncell]
    valid_rings = valid_rings[valid_rings >= 0]
    nRings = int(np.amax(valid_rings)) + 1 if valid_rings.size else 0

    print("Data shape", shape_plot)
    print("Feature rings", nRings)

    if(not os.path.exists(flags.plot_folder)): os.makedirs(flags.plot_folder, exist_ok=True)

    layer_weights = None
    if flags.apply_layer_weights:
        layer_weights = load_layer_weights_from_json(
            flags.layer_weights_file,
            flags.layer_weights_key,
            shape_plot[1],
        )
        print("Applying %s from %s" % (flags.layer_weights_key, flags.layer_weights_file))
        print("Layer weight range: %.3f to %.3f" % (np.amin(layer_weights), np.amax(layer_weights)))

    geom_conv = None



    def LoadFile(fname, EMin = -1.0, nevts = -1):
        print("Load %s" % fname)
        end = None if nevts < 0 else nevts
        with h5py.File(fname,"r") as h5f:
            if(hgcal):
                generated = h5f['showers'][:end,:,:dataset_config['MAX_CELLS']]
                energies = h5f['gen_info'][:end,0]
            else:
                scale_fac = 1000.
                generated = h5f['showers'][:end] * scale_fac
                energies = h5f['incident_energies'][:end] * scale_fac

        if generated.shape[0] != energies.shape[0]:
            # Source file itself has mismatched dataset lengths (seen e.g. in one
            # GraphCNF Photon_LogUniform file). Truncating would silently pair
            # showers with the wrong energies if the dropped entry isn't at the
            # end, so skip the whole file instead (caught by the per-file
            # exception handler in the caller, same as other bad files).
            raise ValueError("%s has mismatched showers/energies lengths (%d vs %d)"
                              % (fname, generated.shape[0], energies.shape[0]))

        energies = np.reshape(energies,(-1,1))
        generated = np.reshape(generated,shape_plot)
        if(EMin > 0.): # apply min energy cut on the unreweighted showers, mimicking the fact that we are cutting off the noise and then correct the layers' relative importance a la the weights!
            mask = generated < EMin
            generated[mask] = 0.

        if layer_weights is not None:
            # Apply proper per-layer sampling fraction weights
            generated *= layer_weights.reshape((1, -1, 1))
        elif hgcal:
            # Legacy: uniform x1000 as crude sampling fraction approximation
            generated *= 1000.


        return generated,energies

    def LoadSample(fname, EMin = -1.0, nevts = -1, reprocess=False ):
        suffix = ".feat.v4"  # v4: removed per-layer log energy from feats (redundant with profiles)
        if layer_weights is not None:
            suffix += ".%s" % flags.layer_weights_key
        feat_basename = os.path.basename(fname) + suffix + ".npz"

        # Try cache next to source file first; fall back to plot_folder for
        # read-only source locations (e.g. /eos).
        feat_file_src = fname + suffix + ".npz"
        feat_file_local = os.path.join(flags.plot_folder, feat_basename)
        feat_file = feat_file_src if os.path.exists(feat_file_src) else feat_file_local

        if(os.path.exists(feat_file) and not reprocess):
            print("Load %s" % feat_file)
            data = np.load(feat_file)
            return data['feats'], data['long_profile'], data['trans_profile']
        else:
            showers, energies = LoadFile(fname, EMin, flags.nevts)
            # Drop showers with any non-finite voxels so feats and profiles are
            # computed on an identical set of showers. compute_profiles used to
            # filter internally while compute_feats did not, which left their
            # row counts mismatched and broke the downstream concatenation.
            finite_mask = np.all(np.isfinite(showers.reshape(showers.shape[0], -1)), axis=1)
            if not np.all(finite_mask):
                n_bad = int(np.sum(~finite_mask))
                print("LoadSample: dropping %d non-finite showers out of %d in %s"
                      % (n_bad, len(finite_mask), fname))
                showers = showers[finite_mask]
                energies = energies[finite_mask]
            feats = compute_feats(showers, energies, geom)
            long_profile, trans_profile = compute_profiles(showers, geom, nRings)
            try:
                np.savez(feat_file_src, feats=feats, long_profile=long_profile, trans_profile=trans_profile)
            except OSError:
                np.savez(feat_file_local, feats=feats, long_profile=long_profile, trans_profile=trans_profile)
            return feats, long_profile, trans_profile


    geant_energies = None
    geant_showers = None
    data_dict = {}

    feats_gen = feats_geant = None
    long_gen = long_geant = None
    trans_gen = trans_geant = None

    if(not flags.geant_only):
        if(flags.generated == ""):
            print("Missing data file to plot!")
            exit(1)
        f_sample_list = utils.get_files(flags.generated)
        nevts_sample = flags.nevts_sample if flags.nevts_sample > 0 else flags.nevts

        for f_sample in f_sample_list:
            try:
                feats, lp, tp = LoadSample( f_sample, flags.EMin, nevts_sample, reprocess=flags.reprocess)
                if(feats_gen is None):
                    feats_gen, long_gen, trans_gen = feats, lp, tp
                else:
                    feats_gen = np.concatenate((feats_gen, feats), axis=0)
                    long_gen = np.concatenate((long_gen, lp), axis=0)
                    trans_gen = np.concatenate((trans_gen, tp), axis=0)

                total_evts = feats_gen.shape[0]
                if(nevts_sample > 0 and total_evts >= nevts_sample): 
                    if(total_evts > nevts_sample): 
                        # trim to exact number if we went over due to concatenation
                        feats_gen = feats_gen[:nevts_sample]
                        long_gen = long_gen[:nevts_sample]
                        trans_gen = trans_gen[:nevts_sample]
                        total_evts = nevts_sample
                    break
            except (OSError, KeyError, ValueError):
                print("Bad file, skipping")

        print("Loaded %i generated showers" % total_evts)


    f_geant_list = utils.get_files(dataset_config['EVAL'], folder=flags.data_folder)
    for f_sample in f_geant_list:
        try:
            feats, lp, tp = LoadSample( f_sample, flags.EMin, flags.nevts, reprocess=flags.reprocess )
        except (OSError, KeyError, ValueError):
            print("Bad Geant file, skipping")
            continue

        if(feats_geant is None):
            feats_geant, long_geant, trans_geant = feats, lp, tp
        else:
            feats_geant = np.concatenate((feats_geant, feats), axis=0)
            long_geant = np.concatenate((long_geant, lp), axis=0)
            trans_geant = np.concatenate((trans_geant, tp), axis=0)

        total_evts = feats_geant.shape[0]
        if(flags.nevts > 0 and total_evts >= flags.nevts): 
            if(total_evts > flags.nevts): 
                feats_geant = feats_geant[:flags.nevts]
                long_geant = long_geant[:flags.nevts]
                trans_geant = trans_geant[:flags.nevts]
            break

    if(feats_geant is None):
        raise RuntimeError("No valid Geant files were loaded from EVAL list.")

    # sanity checks on the calculated features
    inf_gen = np.isinf(feats_gen)
    nan_gen = np.isnan(feats_gen)
    inf_geant = np.isinf(feats_geant)
    nan_geant = np.isnan(feats_geant)
    print(f"Number of Infs: Geant4 {np.sum(inf_geant)}, Model {np.sum(inf_gen)}")
    print(f"Number of Nans: Geant4 {np.sum(nan_geant)}, Model {np.sum(nan_gen)}")
    print("Number of showers: Geant4 %i, Model %i" % (feats_geant.shape[0], feats_gen.shape[0]))

    nLayers = shape_plot[1]
    feat_names = get_feat_names(nLayers)

    if(flags.single_energy):
        # remove incident energy feature
        feats_geant = feats_geant[:, 1:]
        feats_gen = feats_gen[:, 1:]
        feat_names = feat_names[1:]

    if(flags.no_occupancy):
        #don't include occupancy feature
        feats_no_occupancy = [idx for idx,feat_name in enumerate(feat_names) if 'Occupancy' not in feat_name] 
        feats_geant = feats_geant[:, feats_no_occupancy]
        feats_gen = feats_gen[:, feats_no_occupancy]
        feat_names = [feat_names[idx] for idx in feats_no_occupancy]


    #set seed for reproducibility 
    set_seed(flags.seed)


    if(flags.shuffle_labels):
        #randomly shuffle labels, for diff geant-geant bootstraps
        feats_all = np.concatenate([feats_gen, feats_geant], axis=0)
        n_gen = feats_gen.shape[0]
        np.random.shuffle(feats_all)
        feats_gen = feats_all[:n_gen]
        feats_geant = feats_all[n_gen:]


    do_hists = do_classifier = do_fpd = False

    print("mode", flags.mode)
    if(flags.mode == "all"):
        do_hists = do_classifier = do_fpd = True
    elif(flags.mode == "hist"):
        do_hists = True
    elif(flags.mode == "classifier" or flags.mode == "cls"):
        do_classifier = True
    elif(flags.mode == "fpd" or flags.mode == "kpd"):
        do_fpd = True



    #Separation power
    if(do_hists):
        fname = ""
        sep_power_result_str = ""
        sep_power_sums = {
            "Energy": 0.,
            "Transverse":0.,
            "Center": 0.,
            "Width": 0.,
            "Occupancy": 0.,
            "all": 0.,
        }
        sep_power_counts = {
            "Energy": 0,
            "Transverse": 0,
            "Center": 0,
            "Width": 0,
            "Occupancy": 0,
            "all": 0,
        }
        ks_sums = copy.copy(sep_power_sums)

        # Per-feature histograms
        for i, feat_name in enumerate(feat_names):
            if flags.plot:
                fname = os.path.join(flags.plot_folder, feat_name.replace(" ", ""))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                sep_power = make_hist(feats_geant[:,i], feats_gen[:,i], xlabel = feat_name, model_name=flags.name, fname=fname)
                test = kstest(feats_geant[:,i], feats_gen[:,i])
                ks_metric, ks_pval = test.statistic, test.pvalue

            sep_power_result_str += "%i %s: %.3e / %.3e \n" % (i, feat_name, sep_power, ks_metric)

            #ignore incident E
            if("Incident" in feat_name): continue

            if("Energy" in feat_name):
                sum_key = "Energy"
            elif("Center" in feat_name):
                sum_key = "Center"
            elif("Width" in feat_name):
                sum_key = "Width"
            elif("Occupancy" in feat_name):
                sum_key = "Occupancy"
            else:
                print("unmatched feat %s" % feat_name)
                continue

            if np.isnan(sep_power) or np.isnan(ks_metric):
                print("WARNING: NaN sep_power or KS for feature %s, skipping" % feat_name)
                continue
            if max_layer_eval is not None and " Layer " in feat_name:
                layer_num = int(feat_name.split(" Layer ")[-1])
                if layer_num >= max_layer_eval:
                    continue
            sep_power_sums[sum_key] += sep_power
            ks_sums[sum_key] += ks_metric
            sep_power_counts[sum_key] += 1
            sep_power_sums['all'] += sep_power
            sep_power_sums['all'] += sep_power
            ks_sums['all'] += ks_metric
            sep_power_counts['all'] += 1

        # Convert fraction profiles to absolute energy if requested
        if flags.use_absolute:
            E_total_geant = (feats_geant[:, 0] * feats_geant[:, 1]).reshape(-1, 1)
            E_total_gen = (feats_gen[:, 0] * feats_gen[:, 1]).reshape(-1, 1)
            plot_long_geant = long_geant * E_total_geant
            plot_long_gen = long_gen * E_total_gen
            plot_trans_geant = trans_geant * E_total_geant
            plot_trans_gen = trans_gen * E_total_gen
            profile_label = "Energy"
            profile_ylabel = "Energy [MeV]"
        else:
            plot_long_geant = long_geant
            plot_long_gen = long_gen
            plot_trans_geant = trans_geant
            plot_trans_gen = trans_gen
            profile_label = "Energy fraction"
            profile_ylabel = "Energy fraction"

        # Per-layer longitudinal profile histograms
        for i in range(nLayers):
            feat_name = "%s layer %i" % (profile_label, i)
            if flags.plot:
                fname = os.path.join(flags.plot_folder, feat_name.replace(" ", ""))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                sep_power = make_hist(plot_long_geant[:,i], plot_long_gen[:,i], xlabel=feat_name, model_name=flags.name, fname=fname)
                ks_metric = kstest(plot_long_geant[:,i], plot_long_gen[:,i]).statistic

            sep_power_result_str += "%s: %.3e / %.3e \n" % (feat_name, sep_power, ks_metric)
            if np.isnan(sep_power) or np.isnan(ks_metric):
                print("WARNING: NaN sep_power or KS for %s, skipping" % feat_name)
                continue
            if max_layer_eval is not None and i >= max_layer_eval:
                continue
            sep_power_sums["Energy"] += sep_power
            ks_sums["Energy"] += ks_metric
            sep_power_counts["Energy"] += 1

            sep_power_sums['all'] += sep_power
            ks_sums['all'] += ks_metric
            sep_power_counts['all'] += 1

        # Per-ring transverse profile histograms
        for i in range(nRings):
            feat_name = "%s ring %i" % (profile_label, i)
            if flags.plot:
                fname = os.path.join(flags.plot_folder, feat_name.replace(" ", ""))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                sep_power = make_hist(plot_trans_geant[:,i], plot_trans_gen[:,i], xlabel=feat_name, model_name=flags.name, fname=fname)
                ks_metric = kstest(plot_trans_geant[:,i], plot_trans_gen[:,i]).statistic

            sep_power_result_str += "%s: %.3e / %.3e \n" % (feat_name, sep_power, ks_metric)
            if np.isnan(sep_power) or np.isnan(ks_metric):
                print("WARNING: NaN sep_power or KS for %s, skipping" % feat_name)
                continue
            sep_power_sums["Transverse"] += sep_power
            ks_sums["Transverse"] += ks_metric
            sep_power_counts["Transverse"] += 1

            sep_power_sums['all'] += sep_power
            ks_sums['all'] += ks_metric
            sep_power_counts['all'] += 1

        #detailed breakdown in sep file
        with open(os.path.join(flags.plot_folder, 'sep_power.txt'), 'w') as f:
            f.write(sep_power_result_str)

        print("Saved all sep. power metrics in %s" %  os.path.join(flags.plot_folder, "sep_power.txt"))

        #group by categories for metrics file
        sep_power_metrics_str = ""
        for key in sep_power_sums.keys():
            norm = sep_power_counts[key]
            if norm <= 0:
                continue
            avg_sep = sep_power_sums[key] / norm
            avg_ks = ks_sums[key] / norm
            sep_power_metrics_str += "Avg separation power / KS of %s features: %.3e / %.3e \n" % (key, avg_sep, avg_ks)

        print(sep_power_metrics_str)

        with open(os.path.join(flags.plot_folder, 'metrics.txt'), 'w') as f:
            f.write("Num generated showers: %i\n" % feats_gen.shape[0])
            f.write(sep_power_metrics_str)

        # Summary profile plots (average ± std across showers)
        if flags.plot:
            make_profile(
                plot_long_geant, plot_long_gen,
                xlabel="Layer", ylabel=profile_ylabel,
                model_name=flags.name,
                fname=os.path.join(flags.plot_folder, "LongitudinalProfile"),
            )
            make_profile(
                plot_trans_geant, plot_trans_gen,
                xlabel="Ring", ylabel=profile_ylabel,
                model_name=flags.name,
                fname=os.path.join(flags.plot_folder, "Transverse"),
            )
            print("Saved profile summary plots")


    # Combine scalar features with per-shower profile fractions for classifier/FPD
    feats_cls_gen = np.concatenate((feats_gen, long_gen, trans_gen), axis=1)
    feats_cls_geant = np.concatenate((feats_geant, long_geant, trans_geant), axis=1)

    # Remove showers with non-finite (NaN or inf) or absurdly-large features before
    # classifier/FPD. inf arises from overflow in feature computation (e.g.
    # energies*coord^power in utils.py) on finite-but-huge voxels, which np.isnan
    # would miss but the downstream StandardScaler rejects. Some generative models
    # occasionally emit a pathological/diverged shower whose features are still
    # finite (e.g. ~1e6-1e36) but swamp the FPD covariance estimate, producing NaNs
    # inside jetnet's curve_fit; a generous magnitude cutoff catches these too since
    # no legitimate feature value approaches it.
    FEATURE_ABS_MAX = 1e4
    bad_mask_gen   = np.any(~np.isfinite(feats_cls_gen)   | (np.abs(feats_cls_gen)   > FEATURE_ABS_MAX), axis=1)
    bad_mask_geant = np.any(~np.isfinite(feats_cls_geant) | (np.abs(feats_cls_geant) > FEATURE_ABS_MAX), axis=1)
    if bad_mask_gen.any() or bad_mask_geant.any():
        print(f"Removing {bad_mask_gen.sum()} generated and {bad_mask_geant.sum()} Geant4 showers with non-finite/extreme features before classifier/FPD")
        feats_cls_gen   = feats_cls_gen[~bad_mask_gen]
        feats_cls_geant = feats_cls_geant[~bad_mask_geant]

    # Optionally truncate per-layer features at MAX_LAYER_EVAL (all metrics)
    if max_layer_eval is not None:
        n_layers = long_gen.shape[1]
        feats_cls_gen   = truncate_layer_features(feats_cls_gen,   n_layers, max_layer=max_layer_eval)
        feats_cls_geant = truncate_layer_features(feats_cls_geant, n_layers, max_layer=max_layer_eval)

    if(do_classifier):
        # Rerunning classifier-only (mode=cls) against an existing metrics.txt
        # (written by a prior mode=all/hist run) would otherwise just append
        # another set of classifier lines below the stale ones. Strip any
        # previous classifier-derived lines first so metrics.txt stays
        # single-valued per metric; sep-power/FPD lines are untouched.
        metrics_path = os.path.join(flags.plot_folder, 'metrics.txt')
        if os.path.exists(metrics_path):
            stale_prefixes = ("Result of classifier test", "OOD fraction", "OOD test set size", "Reweighting efficiency")
            with open(metrics_path) as f:
                kept_lines = [line for line in f if not line.startswith(stale_prefixes)]
            with open(metrics_path, 'w') as f:
                f.writelines(kept_lines)

        labels_diffu = np.ones((feats_cls_gen.shape[0], 1), dtype=np.float32)
        labels_geant = np.zeros((feats_cls_geant.shape[0], 1), dtype=np.float32)

        labels_all = np.concatenate((labels_diffu, labels_geant), axis = 0)
        feats_all = np.concatenate((feats_cls_gen, feats_cls_geant), axis = 0)


        scaler = StandardScaler()
        feats_all = scaler.fit_transform(feats_all)
        print(feats_all.shape, labels_all.shape)
        inputs_all = np.concatenate((feats_all, labels_all), axis = 1)

        ttv_fracs = np.array([0.7, 0.1, 0.2])
        train_data, test_data, val_data = ttv_split(inputs_all, ttv_fracs)


        input_dim = feats_all.shape[1]
        cls_num_layer = 2
        cls_num_hidden = 2024
        dropout = 0.2
        cls_lr = 1e-4
        classifier = DNN(input_dim = input_dim, num_layer= cls_num_layer, num_hidden = cls_num_hidden, dropout_probability = dropout)
        classifier.to(device)
        print(classifier)
        total_parameters = sum(p.numel() for p in classifier.parameters() if p.requires_grad)

        print("Classifier has {} parameters".format( int(total_parameters)))

        optimizer = torch.optim.Adam(classifier.parameters(), lr= cls_lr)


        if flags.save_mem:
            train_data = torchdata.TensorDataset(torch.tensor(train_data))
            test_data = torchdata.TensorDataset(torch.tensor(test_data))
            val_data = torchdata.TensorDataset(torch.tensor(val_data))
        else:
            train_data = torchdata.TensorDataset(torch.tensor(train_data).to(device))
            test_data = torchdata.TensorDataset(torch.tensor(test_data).to(device))
            val_data = torchdata.TensorDataset(torch.tensor(val_data).to(device))

        train_dataloader = torchdata.DataLoader(train_data, batch_size=flags.cls_batch_size, shuffle=True)
        test_dataloader = torchdata.DataLoader(test_data, batch_size=flags.cls_batch_size, shuffle=False)
        val_dataloader = torchdata.DataLoader(val_data, batch_size=flags.cls_batch_size, shuffle=False)

        for i in range(flags.cls_n_iters):
            classifier = train_and_evaluate_cls(classifier, train_dataloader, val_dataloader, optimizer, flags)
            #classifier = load_classifier(classifier, flags)

            with torch.no_grad():
                print("Now looking at independent dataset:")
                eval_acc, eval_auc, test_scores, test_labels = evaluate_cls(
                    classifier, test_dataloader, flags,
                    final_eval=True,
                    calibration_data=val_dataloader)

            # OOD fraction: events the classifier separates with near-certainty
            # (predicted prob very close to 0 or 1), combined and per true class.
            OOD_LO, OOD_HI = 0.01, 0.99
            is_ood = (test_scores < OOD_LO) | (test_scores > OOD_HI)
            is_gen = test_labels == 1
            is_geant = test_labels == 0
            ood_frac_all = is_ood.mean()
            ood_frac_gen = is_ood[is_gen].mean() if is_gen.any() else float('nan')
            ood_frac_geant = is_ood[is_geant].mean() if is_geant.any() else float('nan')
            # Test-set sizes behind each fraction above, so a reported 0.0
            # can be shown as "< 1/n_test" downstream (print_summary.py) --
            # with a finite test set we have no power to detect an OOD rate
            # below one event, so an observed zero isn't really "zero".
            n_test_all, n_test_gen, n_test_geant = len(test_scores), int(is_gen.sum()), int(is_geant.sum())

            # Reweighting efficiency: Kish's effective sample size for
            # importance weights w = (1-p)/p that would reweight generated
            # events to match Geant4. Probabilities are clipped away from
            # 0/1 to keep weights finite.
            REWEIGHT_EPS = 1e-3
            # Above this OOD fraction the classifier separates gen events
            # almost perfectly, so the density-overlap premise the
            # reweighting trick relies on breaks down. Kish's ESS only
            # measures relative weight variance: if calibration collapses
            # to a constant near the eps floor (common at near-total
            # separation), every weight is identical and n_eff/n reports
            # ~1.0 even though the weights themselves are meaningless.
            REWEIGHT_OOD_THRESHOLD = 0.98
            if ood_frac_gen > REWEIGHT_OOD_THRESHOLD:
                reweight_efficiency = float('nan')
            else:
                p_gen = np.clip(test_scores[is_gen], REWEIGHT_EPS, 1. - REWEIGHT_EPS)
                weights = (1. - p_gen) / p_gen
                n_eff = weights.sum()**2 / (weights**2).sum()
                reweight_efficiency = n_eff / len(weights)

            cls_string = "Result of classifier test (AUC): %.4f \n" % eval_auc
            cls_string += "OOD fraction (combined / gen / geant): %.4f / %.4f / %.4f \n" % (
                ood_frac_all, ood_frac_gen, ood_frac_geant)
            cls_string += "OOD test set size (combined / gen / geant): %d / %d / %d \n" % (
                n_test_all, n_test_gen, n_test_geant)
            cls_string += "Reweighting efficiency (n_eff/n, gen->geant): %.4f \n" % reweight_efficiency
            print(cls_string)
            with open(os.path.join(flags.plot_folder, 'metrics.txt'), 'a') as f:
                f.write(cls_string)

            if flags.save_classifier:
                torch.save({
                    'model_state_dict': classifier.state_dict(),
                    'scaler': scaler,
                    'input_dim': input_dim,
                    'num_layer': cls_num_layer,
                    'num_hidden': cls_num_hidden,
                    'dropout': dropout,
                }, os.path.join(flags.plot_folder, 'classifier.pt'))
                np.savez(os.path.join(flags.plot_folder, 'classifier_test_scores.npz'),
                         scores=test_scores, labels=test_labels)
                print("Saved classifier checkpoint and test scores in %s" % flags.plot_folder)

    if(do_fpd):
        # Rerunning fpd-only (mode=fpd) against an existing metrics.txt would
        # otherwise just append another FPD/KPD pair below the stale ones;
        # strip any previous fpd-derived lines first, mirroring the
        # classifier block above.
        metrics_path = os.path.join(flags.plot_folder, 'metrics.txt')
        if os.path.exists(metrics_path):
            stale_prefixes = ("FPD", "KPD")
            with open(metrics_path) as f:
                kept_lines = [line for line in f if not line.startswith(stale_prefixes)]
            with open(metrics_path, 'w') as f:
                f.writelines(kept_lines)

        #Larger number of samples than default (50k) found to be more stable, reduced unc
        max_samples = 200000
        if(flags.fpd_trim > 0):
            # Optional NLL-trim, robust to the heavy-tailed-feature
            # instability of vanilla FPD on HGCal showers (see fpd_trim.py).
            from fpd_trim import fpd_trimmed_intercept
            fpd_val, fpd_err = fpd_trimmed_intercept(feats_cls_geant, feats_cls_gen, trim_pct=flags.fpd_trim)
            fpd_tag = " [NLL-trim %g%%]" % (flags.fpd_trim * 100)
        else:
            fpd_val, fpd_err = jetnet.evaluation.fpd(feats_cls_geant, feats_cls_gen, max_samples=max_samples)
            fpd_tag = ""

        kpd_val, kpd_err = jetnet.evaluation.kpd(feats_cls_geant, feats_cls_gen)

        fpd_result_str = (
                f"FPD{fpd_tag}: {fpd_val*1e3:.4f} ± {fpd_err*1e3:.4f} x 10^-3\n" 
                f"KPD: {kpd_val*1e3:.4f} ± {kpd_err*1e3:.4f} x 10^-3\n"
            )
        print(fpd_result_str)

        with open(os.path.join(flags.plot_folder, 'metrics.txt'), 'a') as f:
            f.write(fpd_result_str)

    print("Final metrics saved in %s" % (os.path.join(flags.plot_folder, "metrics.txt")))


if(__name__ == "__main__"):

    parser = argparse.ArgumentParser()

    parser.add_argument('-p', '--plot_folder', default='plots/eval/', help='Folder to save results')
    parser.add_argument('-d', '--data_folder', default='data/', help='Folder with Geant dataset')
    parser.add_argument('--generated', '-g', default='', help='Generated showers')
    parser.add_argument('--config', '-c', default='config_dataset2.json', help='Training parameters')
    parser.add_argument('-n', '--nevts', type=int,default=-1, help='Number of events to load')
    parser.add_argument('--nevts_sample', type=int,default=-1, help='Number of events to load from generated sample (if different from nevts)')
    parser.add_argument('--EMin', type = float, default=0.00001, help='Voxel min energy (GeV)')
    parser.add_argument('--name', default='Model', help='Model name (for plot labels)')

    parser.add_argument('--plot', default=False, action='store_true', help='Save 1D feature plots')
    parser.add_argument('--seed', type=int, default=123, help='Set random seed for classifier')

    parser.add_argument('--cls_n_iters', default=1, type=int, help='Num classifiers to train')
    parser.add_argument('--cls_n_epochs', default=50, type=int, help='Num classifier epochs')
    parser.add_argument('--cls_batch_size', default=256, type=int, help='classifier batch size')
    parser.add_argument('--save_mem', action='store_true', default=False,help='Limit GPU memory for classifier')
    parser.add_argument('--no_save_classifier', action='store_false', dest='save_classifier', default=True,
                        help='Skip saving classifier.pt / classifier_test_scores.npz')

    parser.add_argument('--shuffle_labels', default=False, action='store_true', help='Randomly permute labels (for geant-geant bootstraps)')

    parser.add_argument('--geant_only', action='store_true', default=False,help='Plots with just geant')
    parser.add_argument('--single_energy', action='store_true', default=False,help='Flag for the evaluation at fixed incident energy')
    parser.add_argument('--reprocess', action='store_true', default=False,help='Recompute features for eval')
    parser.add_argument('--no_occupancy', action='store_true', default=False,help='Dont include occupancy feature')
    parser.add_argument('-m', '--mode', default='all', help='Which eval metrics to run. Options : hist, cls, fpd, all (default)')
    parser.add_argument('--fpd_trim', type=float, default=0.0,
                        help="Optional NLL-trim fraction for FPD: drop the top-Mahalanobis fraction of events per batch "
                             "under each sample's own MLE Gaussian, then run jetnet's Chong-Forsyth 1/N extrapolation "
                             "(see fpd_trim.py). Default 0 = vanilla jetnet FPD; e.g. 0.01 trims 1%%.")
    parser.add_argument('--apply_layer_weights', action='store_true', default=True,
                        help='Apply per-layer weights loaded from JSON before feature extraction (default key: weightsPerLayer_V16). On by default.')
    parser.add_argument('--no_layer_weights', action='store_false', dest='apply_layer_weights',
                        help='Disable per-layer weights (use legacy uniform x1000 scaling)')
    parser.add_argument('--layer_weights_file', default='HGCalRecHit_layer_weights.json',
                        help='JSON file with HGCal layer weights')
    parser.add_argument('--layer_weights_key', default='weightsPerLayer_V16',
                        help='Key to read from --layer_weights_file (default: weightsPerLayer_V16)')
    parser.add_argument('--use_absolute', action='store_true', default=False,
                        help='Plot per-layer/ring absolute energy instead of energy fraction')

    flags = parser.parse_args()
    compute_metrics(flags)
