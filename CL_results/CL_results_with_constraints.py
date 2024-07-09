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

    from utils import generate_obstacle_constraint, nested_dict, is_dominated, get_pareto_front, get_metric_value, get_pareto_data, plot_bar_chart, confidence_interval
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
        confidence_interval,
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
                "ETH": join(path, "trunk_results", "control_tasks", "custom_100_obstacles.pkl")
            },
        "diamond hardware":
            {
                "Stanford": join(path, "hardware_results", "control_tasks", "stanford.pkl")
            }
    }

    result_loc = {
        "trunk":
            {
                "ETH": join(path, "trunk_results", "obstacle_avoidance")
            },
        "diamond hardware":
            {
                "Stanford": join(path, "hardware_results", "stanford_runs")
            }
    }

    simulation_name = {
        "trunk":
        {
            "ETH": "Simulated Closed-Loop for ETH Trajectory w/ Obstacles (Trunk)"
        },
        "diamond hardware":
        {
            "Stanford": "Hardware Closed-Loop for Stanford Trajectory w/ Obstacles (Diamond)"
        }
    }

    sim_prefix = {
        "trunk":
        {
            "ssm": "ssmr_sim", 
            "linear": "linear_sim", 
            "koopman": "koopman_sim",
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
            'tpwl': "TPWL (28D)",
            'koopman': "Koopman\nEDMD (120D)",
            'DMD': "DMD (15D)"
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
        'diamond hardware':
        {
            'ssm': "SSMR\n(6D)",
            'linear': "SSSR\n(6D)",
            'tpwl': "TPWL\n(42D)",
            'koopman': "Koopman/\nEDMD (66D)",
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
    robot_type_options = ["trunk", "diamond hardware"]
    robot_type = mo.ui.dropdown(robot_type_options, value="trunk", label=f"Choose (simulated) robot type.").form()
    robot_type
    return robot_type, robot_type_options


@app.cell
def __(join, linearModel, load_data, np, path, robot_type):
    # Load  robot parameters

    if robot_type.value == "trunk":
        tip_node = 51
        num_nodes = 709
        rest_file = join(path, 'rest_qv_trunk.pkl')
        rest_data = load_data(rest_file)

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
        control_task_options = ["ETH"]
    else:
        control_task_options = ["Stanford"]

    control_task = mo.ui.dropdown(control_task_options, label=f"Choose control task.").form()
    control_task
    return control_task, control_task_options


@app.cell
def __(mo, robot_type):
    if robot_type.value == "trunk":
        num_experiments = mo.ui.slider(1, 100, 
                    label=f"Number of simulations for simulated trunk.").form()
    else:
        num_experiments = mo.ui.slider(1, 10, 
                    label=f"Number of experiments for diamond hardware.").form()

    num_experiments
    return num_experiments,


@app.cell
def __(mo):
    error_metric_options = ["rmse", "IAE", "ISE"]
    selected_metric = mo.ui.dropdown(error_metric_options, value="ISE", label=f"Choose error metric (y-axis)").form()
    selected_metric
    return error_metric_options, selected_metric


@app.cell
def __(
    MaxNLocator,
    confidence_interval,
    exists,
    get_metric_value,
    interp1d,
    isdir,
    join,
    listdir,
    load_data,
    np,
    patches,
    plt,
    result_loc,
    task_file,
):
    import re

    # Helper functions for hardware
    darker_cyan = (0/255, 128/255, 128/255)  # RGB values normalized to [0, 1]
    model_colors = {'ssm': 'orange','SSM': 'orange', 'linear': 'purple', 'koopman': 'green', 'koopman_static': darker_cyan, 'tpwl': 'olive'}

    # Custom sorting key function
    def extract_number(filename):
        match = re.search(r'_([0-9]+)\.pkl', filename)
        if match:
            return int(match.group(1))
        return -1  # Default value if no match is found

    def plot_task_models(params, outer_gs, model_plot_order, num_trajs=10, 
                         plot_all=False):
        # Params: (robot_type, task_name, Z_EQ, t0, constraints)

        robot_type = params["robot_type"]
        task_name = params["task_name"]
        z_eq = params["Z_EQ"]
        t0 = params["t0"]
        obstacles = params["constraints"]
        metric_type = params['metric_type']

        results_dir = result_loc[robot_type][task_name]
        rmse_vals_normalizer = []

        # Filter directories only
        model_dirs = [d for d in listdir(results_dir) 
                      if not d.startswith('.') and isdir(join(results_dir, d)) 
                      and d in model_plot_order.keys()]

        for model_name in model_dirs:

            # Place to store error metric values for each simulation
            rmse_values = []

            traj_dir = join(results_dir, model_name)
            traj_files = [f for f in listdir(traj_dir) if f.endswith('.pkl')]

            # Order these files
            traj_files = sorted(traj_files, key=extract_number)

            ref_traj_path = task_file[robot_type][task_name]

            # Load desired trajectory
            ref_traj_data = load_data(ref_traj_path)
            z_ref_interp = interp1d(ref_traj_data['t'], ref_traj_data['z'], axis=0)

            # Iterate through all of the results
            for j, traj_file in enumerate(traj_files[:num_trajs]):

                # Current result file
                traj_data = load_data(join(traj_dir, traj_file))
                idx = np.argwhere(traj_data['t'] > t0)[0][0]
                traj_data['t'] = traj_data['t'][idx:] - traj_data['t'][idx]

                if robot_type == "trunk":
                    traj = traj_data['z'][idx:, 3:-1] - z_eq[:-1]
                    current_obstacle_set = obstacles[j]
                else:
                    traj = traj_data['z'].T[idx:, :2] - z_eq[:-1]
                    current_obstacle_set = obstacles

                error = traj - z_ref_interp(traj_data['t'])[:, :2]

                # Use RMSE as the metric for displaying the best trajectory
                # rmse = get_metric_value(metric_type, error, None)
                # rmse_values.append(rmse)

                if model_name == "ssm" or model_name == "SSM":
                    ssm_rmse = get_metric_value(metric_type, error, None)
                    rmse_vals_normalizer.append(ssm_rmse)

        min_rmse_idx = np.argmin(rmse_vals_normalizer)

        # Find index of best trajectory: TODO
        highlight_index = min_rmse_idx

        # Plot the trajectories
        for model_name in model_dirs:

            ax = plt.subplot(outer_gs[model_plot_order[model_name]])

            traj_dir = join(results_dir, model_name)
            traj_files = [f for f in listdir(traj_dir) if f.endswith('.pkl')]

            # Order these files
            traj_files = sorted(traj_files, key=extract_number)

            ref_traj_path = task_file[robot_type][task_name]

            for j, traj_file in enumerate(traj_files[:num_trajs]):
                traj_data = load_data(join(traj_dir, traj_file))

                # Current result file
                idx = np.argwhere(traj_data['t'] > t0)[0][0]
                traj_data['t'] = traj_data['t'][idx:] - traj_data['t'][idx]

                if robot_type == "trunk":
                    traj = traj_data['z'][idx:, 3:-1] - z_eq[:-1]
                    current_obstacle_set = obstacles[j]
                else:
                    traj = traj_data['z'].T[idx:, :2] - z_eq[:-1]
                    current_obstacle_set = obstacles

                state_data_0 = traj[:, 0]
                state_data_1 = traj[:, 1]

                if j == highlight_index:
                    color = model_colors[model_name]
                    alpha = 1.0
                    linewidth = 3.0
                else:
                    color = model_colors[model_name]
                    alpha = 0.2
                    linewidth = 1.0

                # Plotting state data
                if j == highlight_index or plot_all:
                    ax.plot(state_data_0, state_data_1, 
                            color=color, alpha=alpha, linewidth=linewidth)

            if exists(ref_traj_path):
                with open(ref_traj_path, 'rb') as f:
                    ref_traj_data = np.load(f, allow_pickle=True)
                    ax.plot(ref_traj_data['z'][:, 0], 
                            ref_traj_data['z'][:, 1], 'k', 
                            alpha=0.4, linewidth=1.0)

            if task_name == "Circle":
                ax.set_ylim(-1., 28.)
                ax.set_xlim(-15., 20.)
            elif task_name == "ETH":
                ax.set_xlim(-40, 30)
                ax.set_ylim(-35, 35)

            # Plot constraints
            if task_name == "Stanford":
                obstacleDiameter = [3., 4., 6., 4., 4., 6.]
                obstacleLoc = [np.array([1.0, -5.]), np.array([6.5, 3.]), 
                               np.array([-3., 9.]), np.array([-11., -6.]), 
                               np.array([9., -11.]), np.array([-9., 14.])]
                for iObs in range(len(obstacleDiameter)):
                    circle = patches.Circle((obstacleLoc[iObs][0], 
                                             obstacleLoc[iObs][1]), 
                                            obstacleDiameter[iObs]/2, 
                                            edgecolor='red', facecolor='none')
                    # Add the circle to the axes
                    ax.add_patch(circle)
            else:
                curr_obsIdx = min_rmse_idx
                for iObs in range(len(obstacles[curr_obsIdx].center)):
                    circle = patches.Circle((obstacles[curr_obsIdx].center[iObs][0], obstacles[curr_obsIdx].center[iObs][1]), obstacles[curr_obsIdx].diameter[iObs]/2, edgecolor='red', facecolor='none')
                    # Add the circle to the axes
                    ax.add_patch(circle)

            ax.yaxis.set_major_locator(MaxNLocator(3))  
            ax.xaxis.set_major_locator(MaxNLocator(3))
            ax.tick_params(axis='both', labelsize=10)

        return ax

    def calc_metric_with_bars(params, models, num_trajs=10, metric="ISE"):
        rmse_results = {}

        robot_type = params["robot_type"]
        task_name = params["task_name"]
        z_eq = params["Z_EQ"]
        t0 = params["t0"]

        ref_traj_path = task_file[robot_type][task_name]

        # Select normalizer
        control_normalizer = "ssm"

        rmse_normalizer_vals = []

        ref_traj_data = load_data(ref_traj_path)

        ########## Get normalizer ##########
        traj_dir_normalizer = join(result_loc[robot_type][task_name], control_normalizer)
        if not exists(traj_dir_normalizer):
            Exception("No normalizer found for task")

        # Get all valid experiments
        traj_files_normalizer = [f for f in listdir(traj_dir_normalizer) if f.endswith('.pkl')]

        for traj_file in traj_files_normalizer[:num_trajs]:
            traj_data_normalizer = load_data(join(traj_dir_normalizer, traj_file))

            idx = np.argwhere(traj_data_normalizer['t'] > t0)[0][0]
            time_data_normalizer = traj_data_normalizer['t'][idx:] - traj_data_normalizer['t'][idx]

            # idx = 0
            # time_data_normalizer = traj_data_normalizer['t']

            if robot_type == "trunk":
                state_data_normalizer = traj_data_normalizer['z'][idx:, 3:-1] - z_eq[:-1]
            else:
                state_data_normalizer = traj_data_normalizer['z'].T[idx:, :2] - z_eq[:-1]

            # Interpolate the reference trajectory to the current time data
            f_normalizer = interp1d(ref_traj_data['t'], ref_traj_data['z'], axis=0)
            zf_interp_normalizer = f_normalizer(time_data_normalizer)

            error_normalizer = state_data_normalizer - zf_interp_normalizer[:, :2]

            rmse_val_normalizer = get_metric_value(metric, error_normalizer)

            rmse_normalizer_vals.append(rmse_val_normalizer)
        
        rmse_mean_normalizer = np.mean(rmse_normalizer_vals)

        ########## Get metrics for each model ##########
        for model_name in models:
            traj_dir = join(result_loc[robot_type][task_name], model_name)
            if not exists(traj_dir):
                rmse_results[model_name] = {'avg_rmse': 0, 'ci_min': 0, 'ci_max': 0}
                continue

            traj_files = [f for f in listdir(traj_dir) if f.endswith('.pkl')]
            rmse_values = []
            for traj_file in traj_files[:num_trajs]:
                traj_data = load_data(join(traj_dir, traj_file))

                idx = np.argwhere(traj_data['t'] > t0)[0][0]
                time_data = traj_data['t'][idx:] - traj_data['t'][idx]

                if robot_type == "trunk":
                    state_data = traj_data['z'][idx:, 3:-1] - z_eq[:-1]
                else:
                    state_data = traj_data['z'].T[idx:, :2] - z_eq[:-1]

                # Interpolate the reference trajectory to the current time data
                f = interp1d(ref_traj_data['t'], ref_traj_data['z'], axis=0)
                zf_interp = f(time_data)

                error = state_data - zf_interp[:, :2]

                rmse_val = get_metric_value(metric, error)

                # Append each model's task run
                rmse_values.append((rmse_val / rmse_mean_normalizer - 1.)*100)

            # Calculate metric mean and confidence interval for each model
            avg_rmse = np.mean(rmse_values)
            ci_val = confidence_interval(np.asarray(rmse_values))
            rmse_results[model_name] = {'avg_rmse': avg_rmse, 'ci_min': ci_val[0], 'ci_max': ci_val[1]}

        return rmse_results
    return (
        calc_metric_with_bars,
        darker_cyan,
        extract_number,
        model_colors,
        plot_task_models,
        re,
    )


@app.cell
def __(
    bar_plot_name,
    calc_metric_with_bars,
    confidence_interval,
    exists,
    extract_number,
    join,
    listdir,
    load_data,
    metric_legend,
    model_colors,
    np,
    result_loc,
    task_file,
):
    def plot_all_metrics(params, axs, model_plot_order, num_trajs=10, metric="rmse"):
        models = model_plot_order.keys()
        task_name = params["task_name"]
        constraint = params["constraints"]
        robot_type = params["robot_type"]
        t0 = params["t0"]
        z_eq = params["Z_EQ"]
        
        xlabels = [bar_plot_name[robot_type][model] for model in models]

        # Plot RMSE
        # TODO: Setup calc_metric
        rmse_data = calc_metric_with_bars(params, models, 
                                          num_trajs=num_trajs, metric=metric)
        sorted_models = sorted(rmse_data.keys(), key=lambda x: model_plot_order[x])
        avg_rmse_values = [rmse_data[model]['avg_rmse'] for model in sorted_models]
        ci_rmse_values = [(rmse_data[model]['ci_min'], rmse_data[model]['ci_max']) for model in sorted_models]
        bars = axs[0].bar(xlabels, avg_rmse_values, color=[model_colors[model] for model in sorted_models], zorder=999)

        axs[0].errorbar(xlabels[1:], avg_rmse_values[1:], 
                        yerr=np.array(ci_rmse_values[1:]).T, color="black", 
                        alpha=.25, fmt='o', capsize=5, markersize=3, zorder=1000)
        axs[0].set_ylabel(metric_legend[metric], fontsize=7)
        axs[0].yaxis.grid(True)
        axs[0].set_xticklabels(xlabels, fontsize=9)
        axs[0].tick_params(axis='y', labelsize=10)

        # Calculate the top of the error bars
        error_tops = np.array([avg + ci[1] for avg, ci in zip(avg_rmse_values, ci_rmse_values)])

        sorted_tops_vals = np.sort(error_tops)
        max_tops = sorted_tops_vals[-1]
        second_max_tops = sorted_tops_vals[-2]
        if robot_type == "diamond hardware":
            rmse_threshold = 1.2*second_max_tops if max_tops > 2 * second_max_tops else 1.2 * max_tops
            axs[0].set_ylim(0, rmse_threshold)


        # Add metric values above bars
        for bar, top, rmse_val in zip(bars, error_tops, avg_rmse_values):
            if np.isclose(round(rmse_val), 0.):
                axs[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1, 
                            'Baseline', ha='center', va='bottom', 
                            fontsize=10, fontweight='bold')
            else:
                axs[0].text(bar.get_x() + bar.get_width() / 2, top, f'↑{round(rmse_val)}%',
                            ha='center', va='bottom', color='black', fontsize=10,
                            rotation=0, fontweight='bold')

        # Plot % Violation and Max Violation
        avg_viol_values = []
        ci_viol_min = []
        ci_viol_max = []
        maxviol_values = []
        ci_maxviol_min = []
        ci_maxviol_max = []

        # Load reference trajectory
        ref_traj_path = task_file[robot_type][task_name]
        ref_traj_data = load_data(ref_traj_path)

        # TODO: finish this one
        for model_name in models:
            traj_dir = join(result_loc[robot_type][task_name], model_name)
            if not exists(traj_dir):
                avg_viol_values.append(0)
                ci_viol_min.append(0)
                ci_viol_max.append(0)
                maxviol_values.append(0)
                ci_maxviol_min.append(0)
                ci_maxviol_max.append(0)
                continue

            traj_files = [f for f in listdir(traj_dir) if f.endswith('.pkl')]
            # Order these files
            traj_files = sorted(traj_files, key=extract_number)
            
            viol_values = []
            viol_max_values = []
            for j, traj_file in enumerate(traj_files[:num_trajs]):
                traj_data = load_data(join(traj_dir, traj_file))

                idx = np.argwhere(traj_data['t'] > t0)[0][0]

                if robot_type == "trunk":
                    state_data = traj_data['z'][idx:, 3:-1] - z_eq[:-1]
                else:
                    state_data = traj_data['z'].T[idx:, :2] - z_eq[:-1]

                if robot_type == "trunk":
                    viol_bool = [constraint[j].get_constraint_violation(x=None, z=z) 
                                 for z in state_data]
                else:
                    viol_bool = [constraint.get_constraint_violation(x=None, z=z) 
                             for z in state_data]
                viol_idxs = [idx for idx, val in enumerate(viol_bool) if val]
                viol = len(viol_idxs) / len(viol_bool)*100

                if robot_type == "trunk":
                    max_viol = max([constraint[j].get_constraint_violation(x=None, z=z) 
                                    for z in state_data])
                else:
                    max_viol = max([constraint.get_constraint_violation(x=None, z=z) 
                                    for z in state_data])
                
                viol_values.append(viol)
                viol_max_values.append(max_viol)

            avg_viol_values.append(np.mean(viol_values))
            maxviol_values.append(np.mean(viol_max_values))
            ci_avg_viol_val = confidence_interval(np.asarray(viol_values))
            ci_viol_min.append(ci_avg_viol_val[0])
            ci_viol_max.append(ci_avg_viol_val[1])

            ci_max_viol_val = confidence_interval(np.asarray(viol_max_values))
            ci_maxviol_min.append(ci_max_viol_val[0])
            ci_maxviol_max.append(ci_max_viol_val[1])

        # Plot % Violation
        axs[1].bar(xlabels, avg_viol_values, color=[model_colors[model] for model in models], zorder=999)
        axs[1].errorbar(xlabels, avg_viol_values, yerr=(ci_viol_min, ci_viol_max), color="black", alpha=.25, fmt='o', capsize=5, markersize=3, zorder=1000)
        axs[1].set_ylabel('Violation Ratio\n[%]', fontsize=7)
        axs[1].yaxis.grid(True)
        if robot_type == "diamond hardware":
            axs[1].set_ylim(0, 15)
        axs[1].set_xticklabels(xlabels, fontsize=9)
        axs[1].tick_params(axis='y', labelsize=10)

        # Plot Max Violation
        axs[2].bar(xlabels, maxviol_values, color=[model_colors[model] for model in models], zorder=999)
        axs[2].errorbar(xlabels, maxviol_values, yerr=(ci_maxviol_min, ci_maxviol_max), color="black", alpha=.25, fmt='o', capsize=5, markersize=3, zorder=1000)
        axs[2].set_ylabel('Max Violation\n[mm]', fontsize=7)
        axs[2].yaxis.grid(True)
        axs[2].set_xticklabels(xlabels, fontsize=9)
        axs[2].tick_params(axis='y', labelsize=10)

        for ax in axs:
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
    return plot_all_metrics,


@app.cell
def __(
    Z_EQ,
    control_task,
    generate_obstacle_constraint,
    gridspec,
    load_data,
    np,
    num_experiments,
    plot_all_metrics,
    plot_task_models,
    plt,
    robot_type,
    selected_metric,
    task_file,
):
    fig = plt.figure(figsize=(7, 3))  # Adjust the figure size as needed

    # Create a GridSpec for the entire figure
    outer_gs = gridspec.GridSpec(1, 2, figure=fig)

    task_gs = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=outer_gs[0, 0], wspace=0.35, hspace=0.3)

    if robot_type.value == "trunk":
        # Define the model plot order
        model_plot_order = {'ssm': 0, 'linear': 1, 'koopman': 2, 'tpwl': 3}
        t0 = 1.0
        taskParams = load_data(task_file[robot_type.value][control_task.value])['X_list']
        plot_all = False
    else:
        # Define the model plot order
        model_plot_order = {'ssm': 0, 'linear': 1, 'koopman': 2, 'tpwl': 3}
        t0 = 0.0
        # Define constraints for hardware:
        OBSTACLE_DIAMETER = [3., 4., 6., 4., 4., 6.]
        OBSTACLE_LOC = np.array([np.array([1.0, -5.]), np.array([6.5, 3.]), 
                                 np.array([-3., 9.]), np.array([-11., -6.]), 
                                 np.array([9., -11.]), np.array([-9., 14.])])
        taskParams = generate_obstacle_constraint(OBSTACLE_DIAMETER, OBSTACLE_LOC)
        plot_all = True

    plot_params = {
        "robot_type": robot_type.value,
        "task_name": control_task.value,
        "Z_EQ": Z_EQ,
        "t0": t0,
        "constraints": taskParams,
        "metric_type": selected_metric.value
    }
    ax_traj = plot_task_models(plot_params, task_gs, model_plot_order, 
                               num_trajs=num_experiments.value, plot_all=plot_all)

    # Create a GridSpec for the RMSE, % violation, and max violation plots in the second column
    metrics_gs = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=outer_gs[1], hspace=1.0)
    ax_rmse = fig.add_subplot(metrics_gs[0])
    ax_viol = fig.add_subplot(metrics_gs[1])
    ax_max_viol = fig.add_subplot(metrics_gs[2])

    plot_all_metrics(plot_params, [ax_rmse, ax_viol, ax_max_viol], model_plot_order, num_trajs=num_experiments.value, metric=selected_metric.value)

    plt.show()
    fig
    return (
        OBSTACLE_DIAMETER,
        OBSTACLE_LOC,
        ax_max_viol,
        ax_rmse,
        ax_traj,
        ax_viol,
        fig,
        metrics_gs,
        model_plot_order,
        outer_gs,
        plot_all,
        plot_params,
        t0,
        taskParams,
        task_gs,
    )


@app.cell
def __():
    return


if __name__ == "__main__":
    app.run()
