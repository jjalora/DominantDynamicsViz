import marimo

__generated_with = "0.3.3"
app = marimo.App()


@app.cell
def __():
    import marimo as mo
    mo.md("# Open Loop, Finite Horizon Prediction Resuls! 🧠🤖")
    return mo,


@app.cell
def __():
    from os.path import join, split, dirname, abspath
    import copy
    import pickle
    import numpy as np
    from numpy.random import randint
    from tqdm.auto import tqdm
    import time
    np.set_printoptions(linewidth=100)
    import sys
    sys.path.append('..')
    from collections import defaultdict

    import utils as utils
    import plot_utils as plot
    import matplotlib.pyplot as plt
    return (
        abspath,
        copy,
        defaultdict,
        dirname,
        join,
        np,
        pickle,
        plot,
        plt,
        randint,
        split,
        sys,
        time,
        tqdm,
        utils,
    )


@app.cell
def __(__file__, abspath, defaultdict, dirname, sys):
    path = dirname(abspath(__file__))
    root = dirname(path)
    sys.path.append(root)

    def nested_dict():
        return defaultdict(nested_dict)
    return nested_dict, path, root


@app.cell
def __(np):
    # Default settings
    np.random.seed(seed=1) # Seed for random samples
    rDOF = 3
    oDOF = 3
    SSMDim = 6
    DT = 0.01
    return DT, SSMDim, oDOF, rDOF


@app.cell
def __(mo):
    ROBOT = mo.ui.dropdown(["diamond", "trunk"], value="diamond", label=f"Choose Type of Robot.").form()
    ROBOT
    return ROBOT,


@app.cell
def __(mo):
    modelOptions = ["posvel", "linear", "singleDelay", "delays"]
    multiselect = mo.ui.multiselect(options=modelOptions, label=f"Choose SSM Models to Simulate.").form()
    multiselect
    return modelOptions, multiselect


@app.cell
def __(multiselect):
    modelType = multiselect.value
    return modelType,


@app.cell
def __(ROBOT, join, path):
    pathToModel = join(path, ROBOT.value, "SSMmodels")
    return pathToModel,


@app.cell
def __(ROBOT, join, nested_dict, path, pickle):
    # Parameters for robot and SSM models
    if ROBOT.value == "diamond":
        TIP_NODE = 1354
        N_NODES = 1628
        INPUT_DIM = 4
    else:
        TIP_NODE = 51
        N_NODES = 709
        INPUT_DIM = 8

    robot_dir = join(path, ROBOT.value)
    rest_file = join(robot_dir, 'rest_qv.pkl')

    # load rest position
    with open(rest_file, 'rb') as f:
        rest_data = pickle.load(f)
        rest_q = rest_data['q'][0]

    display_names = nested_dict()

    # Legend for figures
    display_names["diamond"] = {
        "SSMR_linear": "SSSR (6D)",
        "SSMR_singleDelay": "SSMR (6D)",
        "SSMR_delays": "SSMR (4 Delays)",
        "SSMR_posvel": "SSMR (Pos-Vel)",
        "DMD": "DMD (11D)",
        "koopman": "EDMD (66D)",
        "tpwl": "TPWL (42D)"
    }

    display_names["trunk"] = {
        "SSMR_linear": "SSSR (6D)",
        "SSMR_singleDelay": "SSMR (6D)",
        "SSMR_delays": "SSMR (4 Delays)",
        "SSMR_posvel": "SSMR (Pos-Vel)",
        "DMD": "DMD (15D)",
        "koopman": "EDMD (120D)",
        "tpwl": "TPWL (28D)"
    }
    return (
        INPUT_DIM,
        N_NODES,
        TIP_NODE,
        display_names,
        f,
        rest_data,
        rest_file,
        rest_q,
        robot_dir,
    )


@app.cell
def __(
    DT,
    SSMDim,
    TIP_NODE,
    join,
    modelType,
    np,
    pathToModel,
    pickle,
    rest_file,
    rest_q,
    robot_dir,
    utils,
):
    models = {}

    def make_assemble_observables(delays):
        def assemble_observables(oData):
            return utils.delayEmbedding(oData, up_to_delay=delays)
        return assemble_observables

    for _m in modelType:
        with open(join(pathToModel, f"SSM_model_{_m}.pkl"), "rb") as _f:
            model = pickle.load(_f)

        V = model['model']['V']
        r_coeff = model['model']['r_coeff']
        w_coeff = model['model']['w_coeff']
        v_coeff = model['model']['v_coeff']
        B_r = model['model']['B']
        q_bar = (model['model']['q_eq'] - rest_q)[TIP_NODE*3:TIP_NODE*3+3]
        if _m == "posvel":
            q_bar = np.hstack((q_bar, np.zeros(3)))
        else:
            # For delays, repear q_bar model['delays'] times
            q_bar = np.repeat(q_bar, model['params']['delays']+1)


        u_bar = model['model']['u_eq']
        ROMOrder = model['params']['ROM_order']
        SSMOrder = model['params']['SSM_order']

        # check that the model is the correct type
        assert model['params']['ROM_order'] == ROMOrder
        assert model['params']['SSM_order'] == SSMOrder

        if model['params']['delay_embedding']:
            delays = model['params']['delays']
            assemble_observables = make_assemble_observables(delays)
        else:
            def assemble_observables(oData):
                if oData.shape[0] > 6:
                    tip_node_slice = np.s_[3*TIP_NODE:3*TIP_NODE+3]
                else:
                    tip_node_slice = np.s_[:3]
                return np.vstack((oData[tip_node_slice, :], np.gradient(oData[tip_node_slice, :], DT, axis=1)))

        models[_m] = {
            'V': V,
            'r_coeff': r_coeff,
            'w_coeff': w_coeff,
            'v_coeff': v_coeff,
            'B_r': B_r,
            'q_bar': q_bar,
            'u_bar': u_bar,
            'ROMOrder': ROMOrder,
            'SSMOrder': SSMOrder,
            'SSMDim': SSMDim,
            'assemble_observables': assemble_observables,
            'dt' : DT
        }

    # Load and construct test trajectory for each SSM model
    test_trajectories = {}
    traj_dir = join(robot_dir, "dataCollection", "open-loop_500")
    (t, z), u = utils.import_pos_data(data_dir=traj_dir,
                                      rest_file=rest_file,
                                      output_node=TIP_NODE, return_inputs=True, traj_index=0)

    t, z, u = utils.interpolate_time_series(t, z, u, DT)

    for _m in modelType:
        y = models[_m]['assemble_observables'](z)
        test_trajectories[_m] = dict({
            'name': _m,
            't': t,
            'z': z,
            'u': u,
            'y': y
        })

    # Store into dictionary for access later
    z_tot = {}
    y_tot = {}
    u_tot = {}
    t_tot = {}

    for _m in modelType:
        z_tot[_m] = test_trajectories[_m]['z']
        y_tot[_m] = test_trajectories[_m]['y']
        u_tot[_m] = test_trajectories[_m]['u']
        t_tot[_m] = np.arange(z_tot[_m].shape[1]) * DT
    return (
        B_r,
        ROMOrder,
        SSMOrder,
        V,
        assemble_observables,
        delays,
        make_assemble_observables,
        model,
        models,
        q_bar,
        r_coeff,
        t,
        t_tot,
        test_trajectories,
        traj_dir,
        u,
        u_bar,
        u_tot,
        v_coeff,
        w_coeff,
        y,
        y_tot,
        z,
        z_tot,
    )


@app.cell
def __(mo):
    horizon = mo.ui.slider(1, 10, label=f"Prediction Horizon Length").form()
    horizon
    return horizon,


@app.cell
def __(mo):
    samples = mo.ui.slider(50, 1000, label=f"Number of Sampled Initial Conditions along Trajectory").form()
    samples
    return samples,


@app.cell
def __(horizon, modelType, randint, samples, z_tot):
    N_horizon, N_samples = horizon.value, samples.value
    sample_indices = randint(0, z_tot[modelType[0]].shape[1], N_samples)
    return N_horizon, N_samples, sample_indices


@app.cell
def __(
    N_horizon,
    N_samples,
    join,
    mo,
    modelType,
    models,
    np,
    pickle,
    sample_indices,
    t_tot,
    time,
    traj_dir,
    u_tot,
    utils,
    y_tot,
    z_tot,
):
    import warnings
    warnings.filterwarnings('ignore')

    z_preds = {}
    rmse_samples = {}
    xdots = {}

    for m in mo.status.progress_bar(modelType):
        print(f"==================== {m} ====================")
        q_samples = []
        rmse_samples[m] = []
        xdots[m] = []
        advect_times = []
        z_preds[m] = []
        z_trues = []

        # Grab current model
        model_ssm = models[m]

        for i in range(N_samples):
            try:
                start_idx = sample_indices[i]
                end_idx = start_idx + N_horizon
                q_samples.append(z_tot[m][:3, start_idx])
                t0 = time.time()

                t_horiz = t_tot[m][start_idx:end_idx]
                y0 = y_tot[m][:, start_idx]
                dt = model_ssm['dt']
                Nhoriz = len(t_horiz) - 1
                xhoriz = np.full((model_ssm['SSMDim'], Nhoriz+1), np.nan)
                y_pred = np.full((len(y0), Nhoriz+1), np.nan)
                y_pred[:, 0] = y0
                uhoriz = u_tot[m][:, start_idx:end_idx] 

                # Setup the model
                Rauton = lambda xr: model_ssm['r_coeff'] @ utils.phi(xr, model_ssm['ROMOrder'])
                R = lambda xr, u: Rauton(np.atleast_2d(xr)) + model_ssm['B_r'] @ utils.phi(u, order=1)
                # chart map
                if m == "delays" or model_ssm['v_coeff'] is None:
                    Vauton = lambda y : model_ssm['V'].T @ y
                else:
                    Vauton = lambda y: model_ssm['v_coeff'] @ utils.phi(y.reshape(-1,1), model_ssm['SSMOrder'])
                Wauton = lambda xr: model_ssm['w_coeff'] @ utils.phi(xr, model_ssm['SSMOrder']) # parameterization map

                # Setup the initial condition in reduced coordinates
                x0 = Vauton(y0)

                y_pred = utils.predict_open_loop(R, Wauton, t_horiz, uhoriz, x0=x0.flatten(), method='Radau')

                t1 = time.time()

                # compute RMSE
                if m == "posvel":
                    z_pred, z_true = y_pred[:3, :], y_tot[m][:3, start_idx:end_idx]
                else:
                    z_pred, z_true = y_pred[-3:, :], y_tot[m][-3:, start_idx:end_idx]

                rmse = np.sqrt(np.mean(np.linalg.norm(z_pred - z_true, axis=0)**2))
                rmse_samples[m].append(rmse)
                advect_times.append(t1 - t0)
                z_preds[m].append(z_pred)
                z_trues.append(z_true)
            except:
                # RMSE is nan
                rmse_samples[m].append(np.nan)
                advect_times.append(np.nan)
                z_preds[m].append(np.full((3, N_horizon), np.nan))
                z_trues.append(np.full((3, N_horizon), np.nan))

        print("avg RMSE:", np.nanmean(rmse_samples[m]))

        with open(join(traj_dir, f"SSMR_{m}_rmse_samples.pkl"), "wb") as f_ssm:
            pickle.dump(rmse_samples[m], f_ssm)
        with open(join(traj_dir, f"SSMR_{m}_q_samples.pkl"), "wb") as f_ssm:
            pickle.dump(q_samples, f_ssm)
        with open(join(traj_dir, f"SSMR_{m}_advect_times.pkl"), "wb") as f_ssm:
            pickle.dump(advect_times, f_ssm)
    return (
        Nhoriz,
        R,
        Rauton,
        Vauton,
        Wauton,
        advect_times,
        dt,
        end_idx,
        f_ssm,
        i,
        m,
        model_ssm,
        q_samples,
        rmse,
        rmse_samples,
        start_idx,
        t0,
        t1,
        t_horiz,
        uhoriz,
        warnings,
        x0,
        xdots,
        xhoriz,
        y0,
        y_pred,
        z_pred,
        z_preds,
        z_true,
        z_trues,
    )


@app.cell
def __():
    # DMD Baseline
    from scipy.io import loadmat
    from sofacontrol.baselines.koopman import koopman_utils
    return koopman_utils, loadmat


@app.cell
def __(ROBOT, join, koopman_utils, loadmat, path):
    DMD_model_file = join(path, ROBOT.value, "DMD.mat")
    DMD_data = loadmat(DMD_model_file)['py_data'][0, 0]
    raw_model = DMD_data['model']
    raw_params = DMD_data['params']
    model_DMD = koopman_utils.KoopmanModel(raw_model, raw_params)
    scaling_DMD = koopman_utils.KoopmanScaling(scale=model_DMD.scale)
    return (
        DMD_data,
        DMD_model_file,
        model_DMD,
        raw_model,
        raw_params,
        scaling_DMD,
    )


@app.cell
def __(
    INPUT_DIM,
    N_horizon,
    N_samples,
    TIP_NODE,
    join,
    model_DMD,
    np,
    pickle,
    rest_q,
    sample_indices,
    scaling_DMD,
    t_tot,
    time,
    tqdm,
    traj_dir,
    u_tot,
    utils,
    z_tot,
):
    z_tot_DMD = (z_tot['posvel'].T + rest_q[3*TIP_NODE:3*(TIP_NODE+1)]).T
    z_tot_DMD = scaling_DMD.scale_down(y=z_tot_DMD.T).T
    u_tot_DMD = scaling_DMD.scale_down(u=utils.delayEmbedding(u_tot['posvel'], up_to_delay=1, embed_coords=list(range(INPUT_DIM)))[:INPUT_DIM, :].T).T
    y_tot_DMD = np.vstack([z_tot_DMD, utils.delayEmbedding(z_tot_DMD, up_to_delay=1)[:3, :], u_tot_DMD])

    if model_DMD.inputInFeatures:
        y_tot_DMD = np.vstack([z_tot_DMD, utils.delayEmbedding(z_tot_DMD, up_to_delay=1)[:3, :], u_tot_DMD])
    else:
        y_tot_DMD = np.vstack([z_tot_DMD, utils.delayEmbedding(z_tot_DMD, up_to_delay=1)[:3, :]])

    rmse_samples_DMD = []
    xdots_DMD = []
    advect_times_DMD = []
    z_preds_DMD = []
    z_trues_DMD = []
    q_samples_DMD = []

    for i_DMD in tqdm(range(N_samples), position=1, leave=True):
        start_idx_DMD = sample_indices[i_DMD]
        end_idx_DMD = start_idx_DMD + N_horizon
        q_samples_DMD.append(z_tot['posvel'][:3, start_idx_DMD])

        t0_DMD = time.time()
        # advect Koopman to obtain finite-horizon prediction
        t_DMD, y_DMD, u_DMD = t_tot['posvel'][start_idx_DMD:end_idx_DMD], y_tot_DMD[:, start_idx_DMD:end_idx_DMD], u_tot_DMD[:, start_idx_DMD:end_idx_DMD]
        y0_DMD = y_DMD[:, 0]
        N = len(t_DMD)-1
        x_DMD = np.full((model_DMD.A_d.shape[0], N+1), np.nan)
        xdot = np.full((model_DMD.A_d.shape[0], N), np.nan)
        y_pred_DMD = np.full((len(y0_DMD), N+1), np.nan)
        y_pred_DMD[:, 0] = y0_DMD
        # advect for horizon N
        for j_DMD in range(N):
            # lift the observables
            if j_DMD == 0:
                x_DMD[:, j_DMD] = model_DMD.lift_data(*y_pred_DMD[:, j_DMD])

            x_DMD[:, j_DMD+1] = model_DMD.A_d @ x_DMD[:, j_DMD] + model_DMD.B_d @ u_DMD[:, j_DMD]

            if model_DMD.inputInFeatures:
                y_pred_DMD[:, j_DMD+1] = np.concatenate([model_DMD.C @ x_DMD[:, j_DMD+1], y_pred_DMD[:3, j_DMD], u_DMD[:, j_DMD]])
            else:
                y_pred_DMD[:, j_DMD+1] = np.concatenate([model_DMD.C @ x_DMD[:, j_DMD+1], y_pred_DMD[:3, j_DMD]])
        t1_DMD = time.time()
        # compute RMSE
        z_pred_DMD = (scaling_DMD.scale_up(y=y_pred_DMD[:3, :].T) - rest_q[3*TIP_NODE:3*(TIP_NODE+1)]).T
        z_true_DMD = z_tot['posvel'][:, start_idx_DMD:end_idx_DMD]
        rmse_DMD = np.sqrt(np.mean(np.linalg.norm(z_pred_DMD - z_true_DMD, axis=0)**2))
        rmse_samples_DMD.append(rmse_DMD)
        advect_times_DMD.append(t1_DMD - t0_DMD)
        z_preds_DMD.append(z_pred_DMD)
        z_trues_DMD.append(z_true_DMD)
        xdots_DMD.append(xdot)

    print("avg RMSE:", np.nanmean(rmse_samples_DMD))
    # print("max RMSE sample idx:", sample_indices[max_rmse_index])
    with open(join(traj_dir, f"DMD_rmse_samples.pkl"), "wb") as f_DMD:
        pickle.dump(rmse_samples_DMD, f_DMD)
    with open(join(traj_dir, f"DMD_q_samples.pkl"), "wb") as f_DMD:
        pickle.dump(q_samples_DMD, f_DMD)
    with open(join(traj_dir, f"DMD_advect_times.pkl"), "wb") as f_DMD:
        pickle.dump(advect_times_DMD, f_DMD)

    print(rmse_samples_DMD)
    return (
        N,
        advect_times_DMD,
        end_idx_DMD,
        f_DMD,
        i_DMD,
        j_DMD,
        q_samples_DMD,
        rmse_DMD,
        rmse_samples_DMD,
        start_idx_DMD,
        t0_DMD,
        t1_DMD,
        t_DMD,
        u_DMD,
        u_tot_DMD,
        x_DMD,
        xdot,
        xdots_DMD,
        y0_DMD,
        y_DMD,
        y_pred_DMD,
        y_tot_DMD,
        z_pred_DMD,
        z_preds_DMD,
        z_tot_DMD,
        z_true_DMD,
        z_trues_DMD,
    )


@app.cell
def __(ROBOT, join, koopman_utils, loadmat, path):
    # Koopman Baseline
    koopman_model_file = join(path, ROBOT.value, "koopman_model.mat")
    koopman_data = loadmat(koopman_model_file)['py_data'][0, 0]

    raw_model_koop = koopman_data['model']
    raw_params_koop = koopman_data['params']
    model_koop = koopman_utils.KoopmanModel(raw_model_koop, raw_params_koop)
    scaling_koop = koopman_utils.KoopmanScaling(scale=model_koop.scale)
    return (
        koopman_data,
        koopman_model_file,
        model_koop,
        raw_model_koop,
        raw_params_koop,
        scaling_koop,
    )


@app.cell
def __(
    INPUT_DIM,
    N_horizon,
    N_samples,
    TIP_NODE,
    join,
    model_koop,
    np,
    pickle,
    rest_q,
    sample_indices,
    scaling_koop,
    t_tot,
    time,
    tqdm,
    traj_dir,
    u_tot,
    utils,
    z_tot,
):
    # Transformations and initializations for Koopman analysis
    z_tot_koop = (z_tot['posvel'].T + rest_q[3*TIP_NODE:3*(TIP_NODE+1)]).T
    z_tot_koop = scaling_koop.scale_down(y=z_tot_koop.T).T
    u_tot_koop = scaling_koop.scale_down(u=utils.delayEmbedding(u_tot['posvel'], up_to_delay=1, embed_coords=list(range(INPUT_DIM)))[:INPUT_DIM, :].T).T

    if model_koop.inputInFeatures:
        y_tot_koop = np.vstack([z_tot_koop, utils.delayEmbedding(z_tot_koop, up_to_delay=1)[:3, :], u_tot_koop])
    else:
        y_tot_koop = np.vstack([z_tot_koop, utils.delayEmbedding(z_tot_koop, up_to_delay=1)[:3, :]])

    rmse_samples_koop = []
    xdots_koop = []
    advect_times_koop = []
    z_preds_koop = []
    z_trues_koop = []
    q_samples_koop = []

    for i_koop in tqdm(range(N_samples), position=1, leave=True):
        start_idx_koop = sample_indices[i_koop]
        end_idx_koop = start_idx_koop + N_horizon
        q_samples_koop.append(z_tot['posvel'][:3, start_idx_koop])
        t0_koop = time.time()

        t_koop, y_koop, u_koop = t_tot['posvel'][start_idx_koop:end_idx_koop], y_tot_koop[:, start_idx_koop:end_idx_koop], u_tot_koop[:, start_idx_koop:end_idx_koop]
        y0_koop = y_koop[:, 0]
        N_koop = len(t_koop)-1
        x_koop = np.full((model_koop.A_d.shape[0], N_koop+1), np.nan)
        xdot_koop = np.full((model_koop.A_d.shape[0], N_koop), np.nan)
        y_pred_koop = np.full((len(y0_koop), N_koop+1), np.nan)
        y_pred_koop[:, 0] = y0_koop

        for j_koop in range(N_koop):
            if j_koop == 0:
                x_koop[:, j_koop] = model_koop.lift_data(*y_pred_koop[:, j_koop])
            x_koop[:, j_koop+1] = model_koop.A_d @ x_koop[:, j_koop] + model_koop.B_d @ u_koop[:, j_koop]

            if model_koop.inputInFeatures:
                y_pred_koop[:, j_koop+1] = np.concatenate([model_koop.C @ x_koop[:, j_koop+1], y_pred_koop[:3, j_koop], u_koop[:, j_koop]])
            else:
                y_pred_koop[:, j_koop+1] = np.concatenate([model_koop.C @ x_koop[:, j_koop+1], y_pred_koop[:3, j_koop]])

        t1_koop = time.time()

        z_pred_koop = (scaling_koop.scale_up(y=y_pred_koop[:3, :].T) - rest_q[3*TIP_NODE:3*(TIP_NODE+1)]).T
        z_true_koop = z_tot['posvel'][:, start_idx_koop:end_idx_koop]
        rmse_koop = np.sqrt(np.mean(np.linalg.norm(z_pred_koop - z_true_koop, axis=0)**2))
        rmse_samples_koop.append(rmse_koop)
        advect_times_koop.append(t1_koop - t0_koop)
        z_preds_koop.append(z_pred_koop)
        z_trues_koop.append(z_true_koop)
        xdots_koop.append(xdot_koop)

    print("Average RMSE:", np.nanmean(rmse_samples_koop))

    # Saving results
    with open(join(traj_dir, "koopman_rmse_samples.pkl"), "wb") as f_koop:
        pickle.dump(rmse_samples_koop, f_koop)
    with open(join(traj_dir, "koopman_q_samples.pkl"), "wb") as f_koop:
        pickle.dump(q_samples_koop, f_koop)
    with open(join(traj_dir, "koopman_advect_times.pkl"), "wb") as f_koop:
        pickle.dump(advect_times_koop, f_koop)

    print(rmse_samples_koop)
    return (
        N_koop,
        advect_times_koop,
        end_idx_koop,
        f_koop,
        i_koop,
        j_koop,
        q_samples_koop,
        rmse_koop,
        rmse_samples_koop,
        start_idx_koop,
        t0_koop,
        t1_koop,
        t_koop,
        u_koop,
        u_tot_koop,
        x_koop,
        xdot_koop,
        xdots_koop,
        y0_koop,
        y_koop,
        y_pred_koop,
        y_tot_koop,
        z_pred_koop,
        z_preds_koop,
        z_tot_koop,
        z_true_koop,
        z_trues_koop,
    )


@app.cell
def __(DT, N_NODES, ROBOT, TIP_NODE, join, np, path):
    # TPWL Baseline
    from sofacontrol.tpwl import tpwl, tpwl_config
    from sofacontrol.measurement_models import linearModel, MeasurementModel

    # Specify a measurement and output model
    cov_q = 0.0 * np.eye(3 * len([TIP_NODE]))
    cov_v = 0.0 * np.eye(3 * len([TIP_NODE]))
    measurement_model = linearModel(nodes=[TIP_NODE], num_nodes=N_NODES)
    output_model = MeasurementModel(nodes=[TIP_NODE], num_nodes=N_NODES, pos=True, vel=True, S_q=cov_q, S_v=cov_v)
    # Load and configure the TPWL model from data saved
    tpwl_model_file = join(path, ROBOT.value, "tpwl_model_snapshots.pkl")
    config = tpwl_config.tpwl_dynamics_config()
    model_tpwl = tpwl.TPWLATV(data=tpwl_model_file, params=config.constants_sim, Hf=output_model.C,
                         Cf=measurement_model.C)
    model_tpwl.pre_discretize(dt=DT)
    return (
        MeasurementModel,
        config,
        cov_q,
        cov_v,
        linearModel,
        measurement_model,
        model_tpwl,
        output_model,
        tpwl,
        tpwl_config,
        tpwl_model_file,
    )


@app.cell
def __(
    DT,
    N_NODES,
    join,
    np,
    rest_file,
    robot_dir,
    split,
    traj_dir,
    utils,
    z_tot,
):
    test_trajectories_tpwl = []
    traj_dir_tpwl = join(robot_dir, "dataCollection", "open-loop_500")
    (t_tpwl_test, qv_tpwl_test), u_tpwl_test = utils.import_pos_data(data_dir=traj_dir_tpwl,
                                      rest_file=rest_file,
                                      output_node="all", return_inputs=True, return_velocity=True, traj_index=0)

    _, q_tip, _ = utils.interpolate_time_series(t_tpwl_test, qv_tpwl_test[:3*N_NODES, :], u_tpwl_test, DT)
    t_tpwl_test, v_tip, u_tpwl_test = utils.interpolate_time_series(t_tpwl_test, qv_tpwl_test[3*N_NODES:, :], u_tpwl_test, DT)

    test_trajectories_tpwl.append({
            'name': split(traj_dir)[-1],
            't': t_tpwl_test,
            'q': q_tip,
            'v': v_tip,
            'u': u_tpwl_test,
        })

    q_tot_tpwl = np.hstack([traj['q'] for traj in test_trajectories_tpwl])
    v_tot_tpwl = np.hstack([traj['v'] for traj in test_trajectories_tpwl])
    u_tot_tpwl = np.hstack([traj['u'] for traj in test_trajectories_tpwl])
    t_tot_tpwl = np.arange(z_tot['posvel'].shape[1]) * DT
    return (
        q_tip,
        q_tot_tpwl,
        qv_tpwl_test,
        t_tot_tpwl,
        t_tpwl_test,
        test_trajectories_tpwl,
        traj_dir_tpwl,
        u_tot_tpwl,
        u_tpwl_test,
        v_tip,
        v_tot_tpwl,
    )


@app.cell
def __(
    DT,
    N_horizon,
    N_samples,
    join,
    model_tpwl,
    np,
    pickle,
    q_tot_tpwl,
    sample_indices,
    t_tot_tpwl,
    time,
    tqdm,
    traj_dir_tpwl,
    u_tot_tpwl,
    v_tot_tpwl,
    z_tot,
):
    rmse_samples_tpwl = []
    xdots_tpwl = []
    advect_times_tpwl = []
    z_preds_tpwl = []
    z_trues_tpwl = []
    q_samples_tpwl = []

    for j_tpwl in tqdm(range(N_samples), position=1, leave=True):
        start_idx_tpwl = sample_indices[j_tpwl]
        end_idx_tpwl = start_idx_tpwl + N_horizon
        q_samples_tpwl.append(z_tot['posvel'][:3, start_idx_tpwl])
        t0_tpwl = time.time()
        t_tpwl, q_tpwl, v_tpwl, u_tpwl = t_tot_tpwl[start_idx_tpwl:end_idx_tpwl], q_tot_tpwl[:, start_idx_tpwl:end_idx_tpwl], v_tot_tpwl[:, start_idx_tpwl:end_idx_tpwl], u_tot_tpwl[:, start_idx_tpwl:end_idx_tpwl]
        N_tpwl = len(t_tpwl)-1
        xf_tpwl = np.vstack([v_tpwl, q_tpwl]) + model_tpwl.rom.x_ref[:, None]
        x0_tpwl = model_tpwl.rom.compute_RO_state(xf=xf_tpwl[:, 0])
        x_tpwl = np.full((model_tpwl.state_dim, N_tpwl+1), np.nan)
        xdot_tpwl = np.full((model_tpwl.state_dim, N_tpwl), np.nan)
        x_tpwl[:, 0] = x0_tpwl
        z0_tpwl = model_tpwl.x_to_zy(x0_tpwl, z=True)

        for i_tpwl in range(N_tpwl):
            A_tpwl, B_tpwl, d_tpwl = model_tpwl.get_jacobians(x=x_tpwl[:, i_tpwl], u=u_tpwl[:, i_tpwl], dt=DT)
            x_tpwl[:, i_tpwl+1] = A_tpwl @ x_tpwl[:, i_tpwl] + B_tpwl @ u_tpwl[:, i_tpwl] + d_tpwl

        t1_tpwl = time.time()
        z_pred_tpwl = model_tpwl.x_to_zy(x_tpwl.T, z=True).T[3:, :]
        z_true_tpwl = z_tot['posvel'][:, start_idx_tpwl:end_idx_tpwl]
        rmse_tpwl = np.sqrt(np.mean(np.linalg.norm(z_pred_tpwl - z_true_tpwl, axis=0)**2))
        rmse_samples_tpwl.append(rmse_tpwl)
        advect_times_tpwl.append(t1_tpwl - t0_tpwl)
        z_preds_tpwl.append(z_pred_tpwl)
        z_trues_tpwl.append(z_true_tpwl)
        xdots_tpwl.append(xdot_tpwl)

    print("Average RMSE:", np.nanmean(rmse_samples_tpwl))
    print("Minimum RMSE:", np.nanmin(rmse_samples_tpwl), "Maximum RMSE:", np.nanmax(rmse_samples_tpwl))

    with open(join(traj_dir_tpwl, "tpwl_rmse_samples.pkl"), "wb") as f_tpwl:
        pickle.dump(rmse_samples_tpwl, f_tpwl)
    with open(join(traj_dir_tpwl, "tpwl_q_samples.pkl"), "wb") as f_tpwl:
        pickle.dump(q_samples_tpwl, f_tpwl)
    with open(join(traj_dir_tpwl, "tpwl_advect_times.pkl"), "wb") as f_tpwl:
        pickle.dump(advect_times_tpwl, f_tpwl)
    return (
        A_tpwl,
        B_tpwl,
        N_tpwl,
        advect_times_tpwl,
        d_tpwl,
        end_idx_tpwl,
        f_tpwl,
        i_tpwl,
        j_tpwl,
        q_samples_tpwl,
        q_tpwl,
        rmse_samples_tpwl,
        rmse_tpwl,
        start_idx_tpwl,
        t0_tpwl,
        t1_tpwl,
        t_tpwl,
        u_tpwl,
        v_tpwl,
        x0_tpwl,
        x_tpwl,
        xdot_tpwl,
        xdots_tpwl,
        xf_tpwl,
        z0_tpwl,
        z_pred_tpwl,
        z_preds_tpwl,
        z_true_tpwl,
        z_trues_tpwl,
    )


@app.cell
def __(ROBOT, mo):
    mo.md(f"**Open Loop Results for {ROBOT.value} Robot**")
    return


@app.cell
def __(ROBOT, copy, display_names, join, np, pickle, plot, plt, traj_dir):
    baselines = ["DMD", "koopman", "tpwl"]
    # methods = ["SSMR_" + model_name for model_name in modelType] + baselines
    methods = ["SSMR_singleDelay", "SSMR_linear"] + baselines


    display_names_all_adiabatic = copy.copy(display_names[ROBOT.value])

    show_advect_times = False
    max_value_for_legand = 5.

    fig, axs = plt.subplots(3 + show_advect_times, len(methods),
                            figsize=(2.6*len(methods), (9 if show_advect_times else 7)),
                            height_ratios=([4, 3, 2, 2] if show_advect_times else [5, 2, 1]),
                            sharey='row', sharex='row', layout="compressed")
    for i_sim, method in enumerate(methods):
        with open(join(traj_dir, f"{method}_rmse_samples.pkl"), "rb") as f_method:
            rmse_samples_sim = np.array(pickle.load(f_method))
        with open(join(traj_dir, f"{method}_q_samples.pkl"), "rb") as f_method:
            q_samples_sim = np.stack(pickle.load(f_method))
        with open(join(traj_dir, f"{method}_advect_times.pkl"), "rb") as f_method:
            advect_times_sim = np.array(pickle.load(f_method))
        colorbar = (i_sim == len(methods) - 1)
        if i_sim == 0:
            ylabels = [r"$y$ [mm]", r"$z$ [mm]"]
        else:
            ylabels = ["", ""]

        z_sign = -1 if ROBOT.value == "trunk" else 1
        colorbar = (i_sim == len(methods) - 1)
        plot.prediction_accuracy_map(q_samples_sim[:, [0, 1]], rmse_samples_sim, vmin=0., vmax=max_value_for_legand, ax=axs[0, i_sim], ylabel=ylabels[0], cax=axs[:, :], show=False, colorbar=colorbar)
        plot.prediction_accuracy_map(np.array([q_samples_sim[:, 0], z_sign*q_samples_sim[:, 2]]).T, rmse_samples_sim, vmin=0., vmax=max_value_for_legand, ax=axs[1, i_sim], colorbar=False, ylabel=ylabels[1], cax=axs[:, :], show=False)
        plot.boxplot(rmse_samples_sim, ax=axs[2, i_sim], show=False, xlabel="RMSE [mm]")
        if show_advect_times:
            plot.boxplot(advect_times_sim, ax=axs[2, i_sim], show=False, xlabel="Advect time [s]")
        axs[0, i_sim].set_title(display_names_all_adiabatic[method])

    fig.savefig(join(traj_dir, f"prediction_accuracy_{ROBOT.value}.png"), bbox_inches='tight', dpi=300)
    fig.savefig(join(traj_dir, f"prediction_accuracy_{ROBOT.value}.eps"), bbox_inches='tight', dpi=300)

    fig
    return (
        advect_times_sim,
        axs,
        baselines,
        colorbar,
        display_names_all_adiabatic,
        f_method,
        fig,
        i_sim,
        max_value_for_legand,
        method,
        methods,
        q_samples_sim,
        rmse_samples_sim,
        show_advect_times,
        ylabels,
        z_sign,
    )


@app.cell
def __():
    return


if __name__ == "__main__":
    app.run()
