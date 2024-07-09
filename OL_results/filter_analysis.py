import marimo

__generated_with = "0.3.3"
app = marimo.App()


@app.cell
def __():
    import marimo as mo
    mo.md("# Effect of Time-Delays on Open-loop Prediction")
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

    import utils as utils
    import plot_utils as plot
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.ticker import MaxNLocator, LogLocator, LogFormatter
    return (
        LogFormatter,
        LogLocator,
        MaxNLocator,
        abspath,
        copy,
        dirname,
        gridspec,
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
def __(__file__, abspath, dirname, sys):
    path = dirname(abspath(__file__))
    root = dirname(path)
    sys.path.append(root)
    return path, root


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
def __(ROBOT, join, path):
    pathToModel = join(path, ROBOT.value, "SSMmodels", "filter_analysis")
    return pathToModel,


@app.cell
def __(ROBOT, join, np, path, pickle):
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

    # Legend for figures
    display_names = {
        "SSMR_linear": "SSSR (1 Delay)",
        "SSMR_singleDelay": "SSMR (1 Delay)",
        "SSMR_4delays": "SSMR (4 Delays)",
        "SSMR_10delays": "SSMR (10 Delays)",
        "SSMR_15delays": "SSMR (15 Delays)",
        "SSMR_20delays": "SSMR (20 Delays)"
    }

    # Test Trajectory
    test_file = join(robot_dir, "dataCollection", "test_traj.pkl")
    with open(test_file, 'rb') as f:
        test_data = pickle.load(f)
    test_traj_z = test_data['z'].T[-3:, :] - np.atleast_2d(rest_q[3*TIP_NODE:3*TIP_NODE+3]).T
    test_traj_t = test_data['t']
    # plt.plot(test_traj_z[:, 0], test_traj_z[:, 1])
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
        test_data,
        test_file,
        test_traj_t,
        test_traj_z,
    )


@app.cell
def __(DT, SSMDim, TIP_NODE, join, np, pathToModel, pickle, rest_q, utils):
    models = {}

    def make_assemble_observables(delays):
        def assemble_observables(oData):
            return utils.delayEmbedding(oData, up_to_delay=delays)
        return assemble_observables

    modelType = ["4delays", "10delays", "15delays", "20delays",
                 "singleDelayFrom4", "singleDelayFrom10", "singleDelayFrom15", "singleDelayFrom20"]

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

    # TODO: Debugging
    # test_traj_z_transform = test_traj_phi[-3:, :]
    # print(test_traj_phi.shape)
    return (
        B_r,
        ROMOrder,
        SSMOrder,
        V,
        assemble_observables,
        delays,
        make_assemble_observables,
        model,
        modelType,
        models,
        q_bar,
        r_coeff,
        u_bar,
        v_coeff,
        w_coeff,
    )


@app.cell
def __(mo):
    mo.md("### Visualize effect of time delays")
    return


@app.cell
def __(
    MaxNLocator,
    ROBOT,
    gridspec,
    join,
    models,
    np,
    path,
    plt,
    test_traj_z,
    utils,
):
    # if models['singleDelayFrom20']['v_coeff'] is not None:
    #     eta = models["singleDelayFrom20"]['v_coeff'] @ utils.phi(test_traj_phi, order=3)
    #     z_SSM = (models["singleDelayFrom20"]['w_coeff'] @ utils.phi(eta, order=3))[-3:, :]
    # else:
    #     eta = utils.phi(models["singleDelayFrom20"]['V'].T @ test_traj_phi, order=3)
    #     z_SSM = (models["singleDelayFrom20"]['w_coeff'] @ eta)[-3:, :]

    # # z_SSM = (models["singleDelayFrom15"]['w_coeff'] @ phi)[-3:, :]
    # plt.plot(test_traj_z[0, :], test_traj_z[1, :])
    # plt.plot(z_SSM[0, :], z_SSM[1, :])

    label_counter = 0
    label_list = [chr(i) for i in range(ord('a'), ord('z')+1)]

    SUBPLOT_MAPPING = {
            (0, 0): ["4delays", "singleDelayFrom4"],
            (0, 1): ["10delays", "singleDelayFrom10"],
            (1, 0): ["15delays", "singleDelayFrom15"],
            (1, 1): ["20delays", "singleDelayFrom20"]
            }

    # Reconstruction error for delayed observable and swapped coordinates (delay, singleDelay)
    reconstruction_ratio = {4: [],
                           10: [],
                           15: [],
                           20: []}
    DELAY_MAPPING = {(0,0): 4,
                     (0,1): 10,
                     (1,0): 15,
                     (1,1): 20}

    top_row_axes = []  # List to store all axes of the top row
    dark_orange = (0.8, 0.4, 0)

    fig = plt.figure(figsize=(8, 3))
    gs = gridspec.GridSpec(1, 2, figure=fig)

    # Top row: Each plot is further divided into 2x2 grid
    gs_sub = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=gs[0, 1], wspace=0.25, hspace=0.85)
    for k in range(2):
        for l in range(2):
            ax = fig.add_subplot(gs_sub[k, l])
            ax.yaxis.set_major_locator(MaxNLocator(3))  # Set the maximum number of y-axis ticks to 3
            ax.xaxis.set_major_locator(MaxNLocator(3))  # Set the maximum number of y-axis ticks to 3
            top_row_axes.append(ax)  # Add the axis to our list
            current_model_labels = SUBPLOT_MAPPING[(k, l)]

            # Hide y-axis for plots that are not left-most
            if l > 0:
                ax.tick_params(labelleft=False)

            for model_label in current_model_labels:
                # Get desired trajectory and controlled trajectory
                test_traj_phi = models[model_label]['assemble_observables'](test_traj_z)

                if models[model_label]['v_coeff'] is not None:
                    eta = models[model_label]['v_coeff'] @ utils.phi(test_traj_phi, order=3)
                    z_SSM = (models[model_label]['w_coeff'] @ utils.phi(eta, order=3))[-3:, :]
                else:
                    eta = utils.phi(models[model_label]['V'].T @ test_traj_phi, order=3)
                    z_SSM = (models[model_label]['w_coeff'] @ eta)[-3:, :]

                ax.plot(test_traj_z[0, :], test_traj_z[1, :], color="k", 
                        ls='--', alpha=1.0, linewidth=2, label='Target', zorder=1)

                color = "green" if "single" in model_label else dark_orange
                line, = ax.plot(z_SSM[0, :], z_SSM[1, :], alpha=0.8, color=color)

                # Store reconstruction error. Delays appended first, then singleDelays.
                current_delay = DELAY_MAPPING[(k, l)]
                ax.set_title(f"{current_delay} Delays", fontsize=14)

                reconstruction_ratio[current_delay].append(np.linalg.norm(z_SSM[:, :])/np.linalg.norm(test_traj_z[:, :]))
                # print((current_delay, reconstruction_ratio[current_delay]))
    ax.text(-1.65, 3.1, f"({label_list[1]})", transform=ax.transAxes, fontsize=12, va='top', ha='left')

    ratio_delays = {}
    ratio_singleDelay = {}
    for key in reconstruction_ratio.keys():
        ratio_delays[key] = 100*reconstruction_ratio[key][0]
        ratio_singleDelay[key] = 100*reconstruction_ratio[key][1]

    ax = fig.add_subplot(gs[0, 0])
    ax.yaxis.set_major_locator(MaxNLocator(3))
    ax.plot(reconstruction_ratio.keys(), ratio_delays.values(), '-o', color=dark_orange, label="Original Delays")
    ax.plot(reconstruction_ratio.keys(), ratio_singleDelay.values(), '-D', color='green', label="Reparameterized to 1 Delay")
    ax.set_ylabel("Reconstruction Ratio [%]", fontsize=14)
    ax.set_xlabel("Number of Delays", fontsize=14)
    ax.set_ylim([0, 105])
    ax.set_yticks(range(0, 106, 20))
    ax.legend(loc='lower left', fontsize=9)
    ax.grid(True)
    ax.text(-0.32, 1.1, f"({label_list[0]})", transform=ax.transAxes, fontsize=12, va='top', ha='left')

    SAVE_DIR = join(path, "plots")
    plt.savefig(join(SAVE_DIR, f"{ROBOT.value}_filter_analysis.png"), bbox_inches='tight', dpi=200)

    plt.show()
    fig
    return (
        DELAY_MAPPING,
        SAVE_DIR,
        SUBPLOT_MAPPING,
        ax,
        color,
        current_delay,
        current_model_labels,
        dark_orange,
        eta,
        fig,
        gs,
        gs_sub,
        k,
        key,
        l,
        label_counter,
        label_list,
        line,
        model_label,
        ratio_delays,
        ratio_singleDelay,
        reconstruction_ratio,
        test_traj_phi,
        top_row_axes,
        z_SSM,
    )


@app.cell
def __():
    return


if __name__ == "__main__":
    app.run()
