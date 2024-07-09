import marimo

__generated_with = "0.3.3"
app = marimo.App()


@app.cell
def __(__file__):
    import marimo as mo
    from os.path import join, split, dirname, abspath, exists, isdir
    import copy
    import pickle
    import numpy as np
    from numpy.random import randint
    from tqdm.auto import tqdm
    import time
    np.set_printoptions(linewidth=100)
    import sys
    sys.path.append('..')

    from utils import generate_obstacle_constraint, nested_dict, is_dominated, get_pareto_front, get_metric_value, get_pareto_data, plot_bar_chart
    import plot_utils as plot
    import matplotlib.pyplot as plt
    from collections import defaultdict
    from sofacontrol.utils import load_data, remove_decimal, CircleObstacle
    from sofacontrol.measurement_models import linearModel
    from scipy.interpolate import interp1d
    import matplotlib.gridspec as gridspec
    from matplotlib.ticker import MaxNLocator, LogLocator, LogFormatter
    import matplotlib
    import matplotlib.patches as mpatches
    from matplotlib import patches
    from os import listdir

    path = dirname(abspath(__file__))
    root = dirname(path)
    sys.path.append(root)
    return (
        CircleObstacle,
        LogFormatter,
        LogLocator,
        MaxNLocator,
        abspath,
        copy,
        defaultdict,
        dirname,
        exists,
        generate_obstacle_constraint,
        get_metric_value,
        get_pareto_data,
        get_pareto_front,
        gridspec,
        interp1d,
        is_dominated,
        isdir,
        join,
        linearModel,
        listdir,
        load_data,
        matplotlib,
        mo,
        mpatches,
        nested_dict,
        np,
        patches,
        path,
        pickle,
        plot,
        plot_bar_chart,
        plt,
        randint,
        remove_decimal,
        root,
        split,
        sys,
        time,
        tqdm,
    )


@app.cell
def __(join, path):
    # Helper options

    metric_legend = {
        "rmse": r"Relative RMSE [%]",
        "ITAE": r"Relative ITAE [%]",
        "IAE": r"Relative IAE [%]",
        "ISE": r"Relative ISE [%]"
    }

    task_file = {
        "trunk":
            {
                "ASL": join(path, "trunk_results", "control_tasks", "ASL.pkl"),
                "Pacman": join(path, "trunk_results", "control_tasks", "pacman.pkl"),
                "Stanford": join(path, "trunk_results", "control_tasks", "stanford.pkl")
            },
        "diamond":
            {
                "Figure 8": join(path, "diamond_results", "control_tasks", "figure8.pkl"),
                "Fast Figure 8": join(path, "diamond_results", "control_tasks", "figure8_fast.pkl"),
                "Circle": join(path, "diamond_results", "control_tasks", "circle.pkl")
            },
        "diamond hardware":
            {
                "Figure 8": join(path, "hardware_results", "control_tasks", "figure8.pkl"),
                "Circle": join(path, "hardware_results", "control_tasks", "circle.pkl")
            }
    }

    result_loc = {
        "trunk":
            {
                "ASL": join(path, "trunk_results", "ASL"),
                "Pacman": join(path, "trunk_results", "pacman"),
                "Stanford": join(path, "trunk_results", "stanford")
            },
        "diamond":
            {
                "Figure 8": join(path, "diamond_results", "figure8"),
                "Fast Figure 8": join(path, "diamond_results", "figure8_fast"),
                "Circle": join(path, "diamond_results", "circle")
            },
        "diamond hardware":
            {
                "Figure 8": join(path, "hardware_results", "figure8_runs"),
                "Circle": join(path, "hardware_results", "circle_runs")
            }
    }

    simulation_name = {
        "trunk":
        {
            "ASL": "Simulated Closed-Loop for ASL Trajectory (Trunk)",
            "Pacman": "Simulated Closed-Loop for Pacman Trajectory (Trunk)",
            "Stanford": "Simulated Closed-Loop for Stanford Trajectory (Trunk)"
        },
        "diamond":
        {
            "Figure 8": "Simulated Closed-Loop for Slow Figure 8 Trajectory (Diamond)",
            "Fast Figure 8": "Simulated Closed-Loop for Fast Figure 8 Trajectory (Diamond)",
            "Circle": "Simulated Closed-Loop for Circle Trajectory (Diamond)"
        },
        "diamond hardware":
        {
            "Figure 8": "Hardware Closed-Loop for Slow Figure 8 Trajectory (Diamond)",
            "Circle": "Hardware Closed-Loop for Circle Trajectory (Diamond)"
        }
    }

    sim_prefix = {
        "trunk":
        {
            "ssm": "ssmr_singleDelay_sim", 
            "linear": "ssmr_linear_sim", 
            "koopman": "koopman_sim",
            "DMD": "DMD_sim",
            "tpwl": "tpwl_sim"
        },
        "diamond":
        {
            "ssm": "ssmr_singleDelay_sim", 
            "linear": "ssmr_linear_sim", 
            "koopman": "koopman_sim",
            "DMD": "DMD_sim",
            "tpwl": "tpwl_sim"
        },
        "diamond hardware":
        {
            "ssm": "SSM", 
            "linear": "linear", 
            "koopman": "koopman",
            "DMD": "koopman_static",
            "tpwl": "tpwl"
        }
    }

    legend_name = {
        'trunk':
        {
            'ssm': "SSMR (6D)",
            'linear': "SSSR (6D)",
            'tpwl': "TPWL (42D)",
            'koopman': "Koopman/EDMD (120D)",
            'DMD': "DMD (15D)"
        },
        'diamond':
        {
            'ssm': "SSMR (6D)",
            'linear': "SSSR (6D)",
            'tpwl': "TPWL (42D)",
            'koopman': "Koopman/EDMD (66D)",
            'DMD': "DMD (11D)"
        },
        'diamond hardware':
        {
            'SSM': "SSMR (6D)",
            'linear': "SSSR (6D)",
            'tpwl': "TPWL (42D)",
            'koopman': "Koopman/EDMD (66D)",
            'koopman_static': "Koopman LQR + Pregain (6D)"
        }
    }

    bar_plot_name = {
        'trunk':
        {
            'ssm': "SSMR\n(6D)",
            'linear': "SSSR\n(6D)",
            'tpwl': "TPWL\n(28D)",
            'koopman': "Koopman/EDMD\n(120D)",
            'DMD': "DMD\n(15D)"
        },
        'diamond':
        {
            'ssm': "SSMR\n(6D)",
            'linear': "SSSR\n(6D)",
            'tpwl': "TPWL\n(42D)",
            'koopman': "Koopman/EDMD\n(66D)",
            'DMD': "DMD\n(11D)"
        },
        'diamond hardware':
        {
            'SSM': "SSMR\n(6D)",
            'linear': "SSSR\n(6D)",
            'tpwl': "TPWL\n(42D)",
            'koopman': "Koopman/EDMD\n(66D)",
            'koopman_static': "Koopman LQR w/\nPregain\n(6D)"
        }
    }

    color_legend = {
        'ssm': 'tab:orange',
        'linear': 'tab:purple',
        'tpwl': 'tab:olive',
        'koopman': 'green',
        'DMD': 'tab:cyan'
    }

    alpha = {
        'ssm': 0.7,
        'linear': 1.0,
        'tpwl': 0.7,
        'koopman': 0.7,
        'DMD': 0.5
    }
    return (
        alpha,
        bar_plot_name,
        color_legend,
        legend_name,
        metric_legend,
        result_loc,
        sim_prefix,
        simulation_name,
        task_file,
    )


@app.cell
def __(mo):
    robot_type_options = ["trunk", "diamond", "diamond hardware"]
    robot_type = mo.ui.dropdown(robot_type_options, value="diamond", label=f"Choose (simulated) robot type.").form()
    robot_type
    return robot_type, robot_type_options


@app.cell
def __(join, linearModel, np, path, pickle, robot_type):
    # Load simulated robot parameters
    if robot_type.value == "trunk":
        tip_node = 51
        num_nodes = 709
        rest_file = join(path, 'rest_qv_trunk.pkl')
    elif robot_type.value == "diamond":
        tip_node = 1354
        num_nodes = 1628
        rest_file = join(path, 'rest_qv_diamond.pkl')

    if robot_type.value == "trunk" or robot_type.value == "diamond":
        with open(rest_file, 'rb') as f:
            rest_data = pickle.load(f)
            rest_q = np.hstack((rest_data['q'][1], rest_data['q'][0]))

        # Load trunk equilibrium point
        outputModel = linearModel([tip_node], num_nodes, vel=False)
        Z_EQ = outputModel.evaluate(rest_q, qv=False)

        if robot_type.value == "trunk":
            Z_EQ[2] *= -1
    else:
        Z_EQ = np.array([0.0, 0.0, 0.0])
    return (
        Z_EQ,
        f,
        num_nodes,
        outputModel,
        rest_data,
        rest_file,
        rest_q,
        tip_node,
    )


@app.cell
def __(mo, robot_type):
    if robot_type.value == "trunk":
        control_task_options = ["ASL", "Pacman", "Stanford"]
    elif robot_type.value == "diamond":
        control_task_options = ["Figure 8", "Fast Figure 8", "Circle"]
    else:
        control_task_options = ["Figure 8", "Circle"]

    control_task = mo.ui.dropdown(control_task_options, label=f"Choose control task.").form()
    control_task
    return control_task, control_task_options


@app.cell
def __(mo, robot_type):
    if robot_type.value == "diamond" or robot_type.value == "trunk":
        dt_param_options = [0.02, 0.05]
        dt_params = mo.ui.dropdown([str(_dt) for _dt in dt_param_options], value="0.02",
                                   label=f"Choose time-discretization of model.").form()
        numexp_or_dt = dt_params
    else:
        num_experiments = mo.ui.slider(1, 10, label=f"Number of experiments for diamond hardware.").form()
        numexp_or_dt = num_experiments

    numexp_or_dt
    return dt_param_options, dt_params, num_experiments, numexp_or_dt


@app.cell
def __(mo):
    error_metric_options = ["rmse", "IAE", "ISE"]
    selected_metric = mo.ui.dropdown(error_metric_options, value="ISE", label=f"Choose error metric (y-axis)").form()
    selected_metric
    return error_metric_options, selected_metric


@app.cell
def __(
    Z_EQ,
    control_task,
    dt_params,
    exists,
    get_metric_value,
    interp1d,
    join,
    load_data,
    np,
    numexp_or_dt,
    remove_decimal,
    result_loc,
    robot_type,
    selected_metric,
    sim_prefix,
    task_file,
):
    # Load all relevant files (control task, simulation, normalizer)

    # Convert dt to the corresponding folder name
    if robot_type.value == "diamond" or robot_type.value == "trunk":
        dt_string = remove_decimal(dt_params.value)
        _dt = float(dt_params.value)
    else:
        dt_string = None

    # Load control task and interpolate
    _task_file = task_file[robot_type.value][control_task.value]
    z_target = load_data(_task_file)
    zf_target = interp1d(z_target['t'], z_target['z'], axis=0)

    # Load simulated results
    if dt_string is not None:
        simulation_dir = join(result_loc[robot_type.value][control_task.value], dt_string)

        # Load normalizer for error metric
        normalizer_file_path = join(simulation_dir, 
                                    sim_prefix[robot_type.value]["ssm"] + ".pkl")
        normalizer_data = load_data(normalizer_file_path)
        idx_normalizer = np.argwhere(normalizer_data['t'] >= 1.0)[0][0]

        t_normalizer = normalizer_data['t'][idx_normalizer:] - normalizer_data['t'][idx_normalizer]
        zf_target_normalizer = zf_target(t_normalizer[:-1])
        z_normalizer_centered = normalizer_data['z'][idx_normalizer:, 3:] - Z_EQ

        # Error metric includes x-y-z for y-z tasks, but only x-y for x-y tasks
        if control_task.value == "Circle":
            error_normalize = z_normalizer_centered[:-1, :] - zf_target_normalizer
        else:
            error_normalize = z_normalizer_centered[:-1, :2] - zf_target_normalizer[:, :2]

        # Now compute error metric
        normalizer_error_metric = get_metric_value(selected_metric.value, error_normalize, normalizer_data['t'][:error_normalize.shape[0]])
    else:
        simulation_dir = join(result_loc[robot_type.value][control_task.value], sim_prefix[robot_type.value]["ssm"])

        # Aggregate all error metric across all runs
        error_metric_normalizer = []

        # Comute error metric average and use as normalizer
        for _j in range(numexp_or_dt.value):
            _run_file_j = join(simulation_dir, sim_prefix[robot_type.value]["ssm"] + f"_{_j}.pkl")
            if exists(_run_file_j):
                normalizer_data_j = load_data(_run_file_j)
            else:
                raise RuntimeError(f"Simulation not found: {_run_file_j}")

            # Now we get into the meat and potatoes of getting the data
            idx_normalizer_j = np.argwhere(normalizer_data_j['t'] >= 1.0)[0][0]
            t_normalizer_j = normalizer_data_j['t'][idx_normalizer_j:] - normalizer_data_j['t'][idx_normalizer_j]
            zf_target_normalizer_j = zf_target(t_normalizer_j[:-1])
            z_normalizer_centered_j = normalizer_data_j['z'].T[idx_normalizer_j:, :] - Z_EQ

            least_idx = np.min([z_normalizer_centered_j.shape[0], zf_target_normalizer_j.shape[0]])
            if control_task.value == "Circle":
                error_j = z_normalizer_centered_j[:least_idx, 1:] - zf_target_normalizer_j[:least_idx, 1:]
            else:
                error_j = z_normalizer_centered_j[:least_idx, :2] - zf_target_normalizer_j[:least_idx, :2]

            error_metric_j = get_metric_value(selected_metric.value, error_j, normalizer_data_j['t'][:error_j.shape[0]])
            error_metric_normalizer.append(error_metric_j)

            # Now we compute error metric
            normalizer_error_metric = np.mean(error_metric_normalizer)
    return (
        dt_string,
        error_j,
        error_metric_j,
        error_metric_normalizer,
        error_normalize,
        idx_normalizer,
        idx_normalizer_j,
        least_idx,
        normalizer_data,
        normalizer_data_j,
        normalizer_error_metric,
        normalizer_file_path,
        simulation_dir,
        t_normalizer,
        t_normalizer_j,
        z_normalizer_centered,
        z_normalizer_centered_j,
        z_target,
        zf_target,
        zf_target_normalizer,
        zf_target_normalizer_j,
    )


@app.cell
def __(robot_type):
    # Plot the closed-loop results. 
    # Qualitative trajectories at the top and error metric in the bottom

    if robot_type.value == "diamond hardware":
        SUBPLOT_MAPPING = {
            (0, 0): ["SSM"],
            (0, 1): ["koopman"],
            (1, 0): ["linear", "koopman_static"],
            (1, 1): ["tpwl"]
        }
        CONTROL_MAPPING = {
            "SSM": "ssm",
            "koopman": "koopman",
            "linear": "linear",
            "koopman_static": "DMD",
            "tpwl": "tpwl"
        }
    else:
        SUBPLOT_MAPPING = {
            (0, 0): ["ssmr_singleDelay"],
            (0, 1): ["koopman"],
            (1, 0): ["ssmr_linear", "DMD"],
            (1, 1): ["tpwl"]
        }
        CONTROL_MAPPING = {
            "ssmr_singleDelay": "ssm",
            "koopman": "koopman",
            "ssmr_linear": "linear",
            "DMD": "DMD",
            "tpwl": "tpwl"
        }
    return CONTROL_MAPPING, SUBPLOT_MAPPING


@app.cell
def __(
    CONTROL_MAPPING,
    MaxNLocator,
    SUBPLOT_MAPPING,
    Z_EQ,
    alpha,
    bar_plot_name,
    color_legend,
    control_task,
    get_metric_value,
    gridspec,
    join,
    legend_name,
    load_data,
    metric_legend,
    nested_dict,
    normalizer_error_metric,
    np,
    plot_bar_chart,
    plt,
    robot_type,
    selected_metric,
    simulation_dir,
    simulation_name,
    zf_target,
):
    # Plotting for simulated robots

    if robot_type.value == "trunk" or robot_type.value == "diamond":
        top_row_axes = []  # List to store all axes of the top row
        simData = nested_dict()
        control_error_metric = nested_dict()
        
        handles, labels = [], []
        fig = plt.figure(figsize=(10, 6))
        gs = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[1.6, 1.], hspace=0.45)
        
        # Top row: Each plot is further divided into 2x2 grid
        gs_sub = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=gs[0, 0], wspace=0.1, hspace=0.3)
        
        # Go through each of the four subplots for the trajectories
        for k in range(2):
            for l in range(2):
                ax = fig.add_subplot(gs_sub[k, l])
                # Set the maximum number of y-axis and x-axis ticks to 3
                ax.yaxis.set_major_locator(MaxNLocator(3))
                ax.xaxis.set_major_locator(MaxNLocator(3))
        
                # Add the axis to our list
                top_row_axes.append(ax)
        
                # Hide y-axis for plots that are not left-most
                if l > 0:
                    ax.tick_params(labelleft=False)
        
                # Load the simulated data commensurate with subplot mapping
                current_model_labels = SUBPLOT_MAPPING[(k, l)]
                for _model in current_model_labels:
                    control = CONTROL_MAPPING[_model]
                    control_data = load_data(join(simulation_dir, _model + "_sim.pkl"))
        
                    # Load current model data
                    idx = np.argwhere(control_data['t'] >= 1.0)[0][0]
                    simData[control]['t'] = control_data['t'][idx:] - control_data['t'][idx]
                    simData[control]['z'] = control_data['z'][idx:, 3:]
                    simData[control]['u'] = control_data['u'][idx:, :]
        
        
                    # Compute error for current model in subplot
                    zf_target_control = zf_target(simData[control]['t']) if robot_type.value == "trunk" else zf_target(simData[control]['t'][:-1])
                    z_centered = simData[control]['z'] - Z_EQ if robot_type.value == "trunk" else simData[control]['z'][:-1, :] - Z_EQ
                    if control_task.value == "Circle":
                        _error = (z_centered - zf_target_control)
                    else:
                        _error = (z_centered[:, :2] - zf_target_control[:, :2])
        
                    # Now compute error metric for current model
                    control_error_metric[control] = get_metric_value(selected_metric.value, _error, simData[control]['t'][:_error.shape[0]])
        
                    # Now normalize so that we get relative terms
                    control_error_metric[control] = (control_error_metric[control] /normalizer_error_metric - 1.0)*100
        
                    # Now that we have the error metrics, let's plot the trajectories
                    # Again, treat differently if y-z task versus x-y task
        
                    # First, plot the target trajectory
                    if control_task.value == "Circle":
                        ax.plot(zf_target_control[:, 1], zf_target_control[:, 2],
                               color='black', ls='--', linewidth=1, 
                                label='Target', zorder=1, alpha=0.9)
        
                        # Select dimensions of control trajectory I want to plot
                        trajectory_plot = z_centered[:, 1:]
        
                        ax.set_ylim(-1., 35)
                        ax.set_xlim(-25., 25)
                    else:
                        ax.plot(zf_target_control[:, 0], zf_target_control[:, 1],
                               color='black', ls='--', linewidth=1, 
                                label='Target', zorder=1, alpha=0.9)
        
                        # Select dimensions of control trajectory I want to plot
                        trajectory_plot = z_centered[:, :2]
        
                        if control_task.value == "Fast Figure 8":
                            ax.set_ylim(-20., 20.)
                            ax.set_xlim(-20., 20.)
        
                    line, = ax.plot(trajectory_plot[:, 0], trajectory_plot[:, 1],
                            color=color_legend[control],
                            label=legend_name[robot_type.value][control],
                            linewidth=2,
                            ls='-', markevery=20,
                            alpha=alpha[control])
        
                    handles.append(line)
                    labels.append(legend_name[robot_type.value][control])
        
        top_row_bbox = ax.get_tightbbox(fig.canvas.get_renderer()).transformed(fig.transFigure.inverted())
        
        top_ycoord = top_row_bbox.y0
        
        # Phew, now plot the error metric bars
        ax = fig.add_subplot(gs[1, 0])
        ax.yaxis.set_major_locator(MaxNLocator(3))
        bar_plot_params = {
            "robot": robot_type.value,
            "label": bar_plot_name[robot_type.value],
            "color": color_legend   
        }
        
        # Set threshold based on robot for nice scaling
        if robot_type.value == "diamond":
            bar_threshold = 280.
        else:
            bar_threshold = 800.
        
        plot_bar_chart(ax, bar_plot_params, control_error_metric, 
                       rmse_threshold=bar_threshold, set_threshold=True)
        
        ax.set_ylabel(metric_legend[selected_metric.value])
        middle_row_bbox = ax.get_tightbbox(fig.canvas.get_renderer()).transformed(fig.transFigure.inverted())
        
        middle_ycoord = middle_row_bbox.y1
        fig.suptitle(simulation_name[robot_type.value][control_task.value])
        
        # Now place the legend in the middle
        handle_label_dict = dict(zip(labels, handles))
        unique_labels = handle_label_dict.keys()
        unique_handles = [handle_label_dict[label] for label in unique_labels]
        
        offset = 0.3 * (top_ycoord - middle_ycoord)
        fig.legend(unique_handles, unique_labels, loc='center', 
                ncol=len(unique_labels), bbox_to_anchor=(0.5, middle_ycoord + offset),
                bbox_transform=fig.transFigure, fontsize='10.2')
        
        plt.show()
        fig
    return (
        ax,
        bar_plot_params,
        bar_threshold,
        control,
        control_data,
        control_error_metric,
        current_model_labels,
        fig,
        gs,
        gs_sub,
        handle_label_dict,
        handles,
        idx,
        k,
        l,
        labels,
        line,
        middle_row_bbox,
        middle_ycoord,
        offset,
        simData,
        top_row_axes,
        top_row_bbox,
        top_ycoord,
        trajectory_plot,
        unique_handles,
        unique_labels,
        z_centered,
        zf_target_control,
    )


@app.cell
def __(
    MaxNLocator,
    bar_plot_name,
    exists,
    get_metric_value,
    interp1d,
    isdir,
    join,
    listdir,
    load_data,
    metric_legend,
    np,
    plt,
    result_loc,
    task_file,
):
    # Helper functions for hardware
    darker_cyan = (0/255, 128/255, 128/255)  # RGB values normalized to [0, 1]
    model_colors = {'SSM': 'orange', 'linear': 'purple', 'koopman': 'green', 
                    'koopman_static': darker_cyan, 'tpwl': 'olive'}

    def find_closest_index(t, t_target):
        return np.max(np.argwhere(np.abs(np.asarray(t) - t_target) == np.min(np.abs(np.asarray(t)[np.asarray(t) <= t_target] - t_target))))

    def plot_task_models(task_name, outer_gs, num_trajs=10):
        results_dir = result_loc["diamond hardware"][task_name]
        
        # Filter directories only
        model_dirs = [d for d in listdir(results_dir) 
                      if not d.startswith('.') and isdir(join(results_dir, d))]

        model_plot_order = {'SSM': 0, 'linear': 1, 'koopman': 2, 'koopman_static': 2, 'tpwl': 3}
        
        for model_name in model_dirs:

            # Place to store error metric values for each simulation
            rmse_values = []
            
            ax = plt.subplot(outer_gs[model_plot_order[model_name]])

            traj_dir = join(results_dir, model_name)
            traj_files = [f for f in listdir(traj_dir) if f.endswith('.pkl')]
            ref_traj_path = task_file["diamond hardware"][task_name]

            if exists(ref_traj_path):
                with open(ref_traj_path, 'rb') as f:
                    ref_traj_data = np.load(f, allow_pickle=True)['z'][:, :2]

            if task_name == 'Circle':
                state_idxs = [1, 2]
            else:
                state_idxs = [0, 1]

            for traj_file in traj_files[:num_trajs]:
                with open(join(traj_dir, traj_file), 'rb') as f:
                    traj_data = np.load(f, allow_pickle=True)
                    traj = traj_data['z'][state_idxs, :np.shape(ref_traj_data)[0]].T
                    error = traj - ref_traj_data

                # Use RMSE as the metric for displaying the best trajectory
                rmse = np.sqrt(np.mean(np.linalg.norm(error, axis=-1)**2))
                rmse_values.append(rmse)

            # Find index of best trajectory
            highlight_index = np.argmin(rmse_values)

            for idx, traj_file in enumerate(traj_files[:num_trajs]):
                with open(join(traj_dir, traj_file), 'rb') as f:
                    traj_data = np.load(f, allow_pickle=True)
                
                time_data = traj_data['t']
                if task_name == "Figure 8":
                    idx_end = find_closest_index(time_data, 30.0)
                    time_data = time_data[:idx_end]
                else:
                    idx_end = np.shape(traj_data['z'])[1]
                
                state_data_0 = traj_data['z'][state_idxs[0], :idx_end]
                state_data_1 = traj_data['z'][state_idxs[1], :idx_end]
                
                if idx == highlight_index:
                    color = model_colors[model_name]
                    alpha = 1.0
                    linewidth = 3.0
                else:
                    color = model_colors[model_name]
                    alpha = 0.2
                    linewidth = 1.0

                # Plotting state data
                ax.plot(state_data_0, state_data_1, color=color, 
                        alpha=alpha, linewidth=linewidth)

            if exists(ref_traj_path):
                with open(ref_traj_path, 'rb') as f:
                    ref_traj_data = np.load(f, allow_pickle=True)
                    ax.plot(ref_traj_data['z'][:, state_idxs[0]], 
                            ref_traj_data['z'][:, state_idxs[1]], 'k', 
                            alpha=0.4, linewidth=1.0)
                    
                    if task_name == "Circle":
                        ax.set_ylim(-1., 28.)
                        ax.set_xlim(-15., 20.)
                        
            ax.yaxis.set_major_locator(MaxNLocator(3))  
            ax.xaxis.set_major_locator(MaxNLocator(3))
        
        return ax

    def plot_bar_chart_hardware(ax, rmse_data, model_plot_order, metric):
        # Sort the models based on the model_plot_order
        sorted_models = sorted(rmse_data.keys(), key=lambda x: model_plot_order[x])

        avg_rmse_values = [rmse_data[model]['avg_rmse'] for model in sorted_models]
        ci_values = [(rmse_data[model]['ci_min'], 
                      rmse_data[model]['ci_max']) for model in sorted_models]

        # Create the bar plot with error bars
        bars = ax.bar(sorted_models, avg_rmse_values, color=[model_colors[model] for model in sorted_models], zorder=999)
        ax.errorbar(sorted_models[1:], avg_rmse_values[1:], yerr=np.array(ci_values[1:]).T, ecolor='black', alpha=.4, fmt='none', 
                    capsize=5, markersize=3, zorder=1000)

        ax.set_xticks(range(len(sorted_models)))
        ax.set_xticklabels(bar_plot_name["diamond hardware"][model] for model in sorted_models)
        ax.set_ylabel(metric_legend[metric])
        ax.yaxis.grid(True)
        ax.yaxis.set_major_locator(MaxNLocator(3))
        ax.yaxis.grid(True, color='gray', linewidth=0.5, zorder=1)

        # Calculate the top of the error bars
        error_tops = np.array([avg + ci[1] for avg, ci in zip(avg_rmse_values, ci_values)])

        sorted_tops_vals = np.sort(error_tops)
        max_tops = sorted_tops_vals[-1]
        second_max_tops = sorted_tops_vals[-2]
        rmse_threshold = 1.2*second_max_tops if max_tops > 2 * second_max_tops else 1.2 * max_tops
        ax.set_ylim(0, rmse_threshold)


        # Add metric values above bars
        for bar, top, rmse_val in zip(bars, error_tops, avg_rmse_values):
            if np.isclose(round(rmse_val), 0.):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1, 'Baseline', 
                        ha='center', va='bottom', fontsize=10, fontweight='bold')
            else:
                ax.text(bar.get_x() + bar.get_width() / 2, top, f'↑{round(rmse_val)}%',
                        ha='center', va='bottom', color='black', fontsize=10, rotation=0, fontweight='bold')

    def calc_metric_hardware(task_name, models, num_trajs=10, metric="ISE"):
        rmse_results = {}
        ref_traj_path = task_file['diamond hardware'][task_name]

        control_normalizer = "SSM"
        rmse_normalizer_vals = []
        
        ref_traj_data = load_data(ref_traj_path)
        
        ########## Get normalizer ##########
        traj_dir_normalizer = join(result_loc["diamond hardware"][task_name], 
                                   control_normalizer)
        if not exists(traj_dir_normalizer):
            Exception("No normalizer found for task")
        
        # Get all valid experiments
        traj_files_normalizer = [f for f in listdir(traj_dir_normalizer) if f.endswith('.pkl')]

        for traj_file in traj_files_normalizer[:num_trajs]:
            traj_data_normalizer = load_data(join(traj_dir_normalizer, traj_file))

            time_data_normalizer = traj_data_normalizer['t']

            # TODO: Remove for later. Truncate to 30 seconds for TPWL and Figure 8
            if task_name == "Figure 8":
                idx_end_normalizer = find_closest_index(time_data_normalizer, 30.0)
                time_data_normalizer = time_data_normalizer[:idx_end_normalizer]
            else:
                idx_end_normalizer = np.shape(traj_data_normalizer['z'])[1]

            if task_name == "Circle":
                state_data_normalizer = traj_data_normalizer['z'][1:3, :idx_end_normalizer]
            else:
                state_data_normalizer = traj_data_normalizer['z'][0:2, :idx_end_normalizer]

            # Interpolate the reference trajectory to the current time data
            f_normalizer = interp1d(ref_traj_data['t'], ref_traj_data['z'], axis=0)
            zf_interp_normalizer = f_normalizer(time_data_normalizer)

            if task_name == "Circle":
                zf_interp_normalizer = zf_interp_normalizer[:, 1:].T
            else:
                zf_interp_normalizer = zf_interp_normalizer[:, :2].T

            error_normalizer = zf_interp_normalizer - state_data_normalizer
            rmse_normalizer_val = get_metric_value(metric, error_normalizer, 
                                    traj_data_normalizer['t'][:error_normalizer.shape[0]])
                    
            rmse_normalizer_vals.append(rmse_normalizer_val)
        
        rmse_mean_normalizer = np.mean(rmse_normalizer_vals)
        
        for model_name in models:
            traj_dir = join(result_loc['diamond hardware'][task_name], model_name)
            if not exists(traj_dir):
                print(f"Oh no, {traj_dir} does not exist")
                rmse_results[model_name] = {'avg_rmse': 0, 'ci_min': 0, 'ci_max': 0}
                continue

            traj_files = [f for f in listdir(traj_dir) if f.endswith('.pkl')]
            rmse_values = []
            for traj_file in traj_files[:num_trajs]:
                traj_data = load_data(join(traj_dir, traj_file))

                time_data = traj_data['t']

                # TODO: Remove for later. Truncate to 30 seconds for TPWL and Figure 8
                if task_name == "Figure 8":
                    idx_end = find_closest_index(time_data, 30.0)
                    time_data = time_data[:idx_end]
                else:
                    idx_end = np.shape(traj_data['z'])[1]

                if task_name == "Circle":
                    state_data = traj_data['z'][1:3, :idx_end]
                else:
                    state_data = traj_data['z'][0:2, :idx_end]

                # Interpolate the reference trajectory to the current time data
                f = interp1d(ref_traj_data['t'], ref_traj_data['z'], axis=0)
                zf_interp = f(time_data)

                if task_name == "Circle":
                    zf_interp = zf_interp[:, 1:].T
                else:
                    zf_interp = zf_interp[:, :2].T

                error = zf_interp - state_data
                error_metric = get_metric_value(metric, error, 
                                                 time_data[:error.shape[0]])
                
                rmse_values.append((error_metric / rmse_mean_normalizer - 1.0)*100)

            avg_rmse = np.mean(rmse_values)
            ci_val = confidence_interval(np.asarray(rmse_values))
            rmse_results[model_name] = {'avg_rmse': avg_rmse, 'ci_min': ci_val[0], 'ci_max': ci_val[1]}

        return rmse_results

    def confidence_interval(data, confidence=0.95, num_bootstrap_samples=1000):
        # Generate bootstrap samples
        bootstrap_samples = np.random.choice(data, (num_bootstrap_samples, len(data)), replace=True)
        
        # Calculate the mean for each bootstrap sample
        bootstrap_means = np.mean(bootstrap_samples, axis=1)
        
        # Calculate the lower and upper percentiles
        lower_percentile = (1 - confidence) / 2 * 100
        upper_percentile = (1 + confidence) / 2 * 100
        
        lower_bound = np.percentile(bootstrap_means, lower_percentile)
        upper_bound = np.percentile(bootstrap_means, upper_percentile)
        
        # Calculate the mean of the original data
        mean_data = np.mean(data)
        
        # Calculate the asymmetrical distances
        lower_distance = mean_data - lower_bound
        upper_distance = upper_bound - mean_data
        
        return lower_distance, upper_distance
    return (
        calc_metric_hardware,
        confidence_interval,
        darker_cyan,
        find_closest_index,
        model_colors,
        plot_bar_chart_hardware,
        plot_task_models,
    )


@app.cell
def __(
    calc_metric_hardware,
    control_task,
    gridspec,
    legend_name,
    model_colors,
    numexp_or_dt,
    plot_bar_chart_hardware,
    plot_task_models,
    plt,
    robot_type,
    selected_metric,
    simulation_name,
):
    # Plotting for hardware
    if robot_type.value == "diamond hardware":
        fig_hardware = plt.figure(figsize=(10, 6))  # Adjust the figure size as needed

        # Define the model plot order
        model_plot_order = {'SSM': 0, 'linear': 1, 'koopman': 2, 
                            'koopman_static': 2, 'tpwl': 3}

        # Create a GridSpec for the entire figure
        outer_gs = gridspec.GridSpec(2, 1, figure=fig_hardware, height_ratios=[1, 0.5], 
                                     wspace=0.1, hspace=0.25)

        task_gs = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=outer_gs[0, 0])
        ax_hardware = plot_task_models(control_task.value, task_gs, 
                                       num_trajs=numexp_or_dt.value)

        ax_rmse = plt.subplot(outer_gs[1, 0])
        rmse_data = calc_metric_hardware(control_task.value, model_plot_order.keys(), 
                                         metric=selected_metric.value)
        plot_bar_chart_hardware(ax_rmse, rmse_data, model_plot_order, selected_metric.value)

        
        # Create handles for the legend
        _handles = [plt.Rectangle((0,0),1,1, color=model_colors[model]) for model in model_plot_order]
        _labels = [legend_name['diamond hardware'][model] for model in model_plot_order]

        # # Place the legend in the middle row
        fig_hardware.legend(_handles, _labels, loc='upper center', ncol=len(_labels), bbox_to_anchor=(0.5, 0.390), fontsize=8.6)

        fig_hardware.suptitle(simulation_name['diamond hardware'][control_task.value])

        plt.tight_layout()
        plt.show()
        fig_hardware
    return (
        ax_hardware,
        ax_rmse,
        fig_hardware,
        model_plot_order,
        outer_gs,
        rmse_data,
        task_gs,
    )


@app.cell
def __():
    return


if __name__ == "__main__":
    app.run()
