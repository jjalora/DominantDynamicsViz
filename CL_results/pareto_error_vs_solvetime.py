import marimo

__generated_with = "0.3.3"
app = marimo.App()


@app.cell
def __():
    import marimo as mo
    from os.path import join, split, dirname, abspath, exists
    import copy
    import pickle
    import numpy as np
    from numpy.random import randint
    from tqdm.auto import tqdm
    import time
    np.set_printoptions(linewidth=100)
    import sys
    sys.path.append('..')

    from utils import generate_obstacle_constraint, nested_dict, is_dominated, get_pareto_front, get_metric_value, get_pareto_data
    import plot_utils as plot
    import matplotlib.pyplot as plt
    from collections import defaultdict
    from sofacontrol.utils import load_data, remove_decimal, CircleObstacle
    from sofacontrol.measurement_models import linearModel
    from scipy.interpolate import interp1d
    from matplotlib.lines import Line2D
    return (
        CircleObstacle,
        Line2D,
        abspath,
        copy,
        defaultdict,
        dirname,
        exists,
        generate_obstacle_constraint,
        get_metric_value,
        get_pareto_data,
        get_pareto_front,
        interp1d,
        is_dominated,
        join,
        linearModel,
        load_data,
        mo,
        nested_dict,
        np,
        pickle,
        plot,
        plt,
        randint,
        remove_decimal,
        split,
        sys,
        time,
        tqdm,
    )


@app.cell
def __(__file__, abspath, dirname, join, sys):
    path = dirname(abspath(__file__))
    root = dirname(path)
    sys.path.append(root)
    SAVE_DIR = join(root, "plot_results")
    return SAVE_DIR, path, root


@app.cell
def __(join, path):
    # Legend names
    metric_legend = {
        "rmse": "RMSE [mm]",
        "ITAE": r"ITAE [m s$^2$]",
        "IAE": r"IAE [m s]",
        "ISE": r"ISE [m$^2$ s]"
    }

    x_metric_legend = {
        "violation ratio": "Violation Ratio",
        "max violation": "Max Violation",
        "solve time": "Solve Time [ms]"
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
            }
    }

    sim_prefix = {
        "ssm": "ssmr_singleDelay_sim", 
        "linear": "ssmr_linear_sim", 
        "koopman": "koopman_sim",
        "DMD": "DMD_sim",
        "tpwl": "tpwl_sim"
    }

    legend_name = {
        'trunk':
        {
            'ssm': "SSMR (6D)",
            'linear': "SSSR (6D)",
            'tpwl': "TPWL (42D)",
            'koopman': "EDMD (120D)",
            'DMD': "DMD (15D)"
        },
        'diamond':
        {
            'ssm': "SSMR (6D)",
            'linear': "SSSR (6D)",
            'tpwl': "TPWL (42D)",
            'koopman': "EDMD (66D)",
            'DMD': "DMD (11D)"
        }
    }

    color_legend = {
        'ssm': 'tab:orange',
        'linear': 'tab:purple',
        'tpwl': 'tab:olive',
        'koopman': 'green',
        'DMD': 'tab:cyan'
    }
    return (
        color_legend,
        legend_name,
        metric_legend,
        result_loc,
        sim_prefix,
        task_file,
        x_metric_legend,
    )


@app.cell
def __(mo):
    robot_type_options = ["trunk", "diamond"]
    robot_type = mo.ui.dropdown(robot_type_options, value="diamond", label=f"Choose (simulated) robot type").form()
    robot_type
    return robot_type, robot_type_options


@app.cell
def __(mo, robot_type):
    if robot_type.value == "trunk":
        control_task_options = ["ASL", "Pacman", "Stanford"]
    else:
        control_task_options = ["Figure 8", "Fast Figure 8", "Circle"]

    control_tasks = mo.ui.multiselect(options=control_task_options, label=f"Choose control tasks to include in Pareto plot.").form()
    control_tasks
    return control_task_options, control_tasks


@app.cell
def __(control_tasks):
    marker_style = {}
    linestyle_legend = {}

    styles = ['o', 's', '^']
    line_styles = ['-.', '--', ':']
    for _i, cntrl_task in enumerate(control_tasks.value):
        marker_style[cntrl_task] = styles[_i]
        linestyle_legend[cntrl_task] = line_styles[_i]
    return cntrl_task, line_styles, linestyle_legend, marker_style, styles


@app.cell
def __(mo):
    dt_param_options = [0.02, 0.05]
    dt_params = mo.ui.multiselect(options=[str(_dt) for _dt in dt_param_options], label=f"Choose time-discretization.").form()
    dt_params
    return dt_param_options, dt_params


@app.cell
def __(mo):
    error_metric_options = ["rmse", "ITAE", "IAE", "ISE"]
    selected_metric = mo.ui.dropdown(error_metric_options, value="rmse", label=f"Choose error metric (y-axis)").form()
    selected_metric
    return error_metric_options, selected_metric


@app.cell
def __(nested_dict):
    error_metric_value = nested_dict()
    solve_time_value = nested_dict()
    z_target = nested_dict()
    return error_metric_value, solve_time_value, z_target


@app.cell
def __(
    control_tasks,
    dt_params,
    error_metric_value,
    get_pareto_data,
    interp1d,
    join,
    linearModel,
    load_data,
    np,
    path,
    pickle,
    result_loc,
    robot_type,
    selected_metric,
    sim_prefix,
    solve_time_value,
    task_file,
):
    # Gather pareto plot data for simulated trunk model
    t0 = 1.

    # Load simulated trunk robot parameters
    robot = robot_type.value
    if robot == "trunk":
        tip_node = 51
        num_nodes = 709
        rest_file = join(path, 'rest_qv_trunk.pkl')
    else:
        tip_node = 1354
        num_nodes = 1628
        rest_file = join(path, 'rest_qv_diamond.pkl')

    with open(rest_file, 'rb') as f:
        rest_data = pickle.load(f)
        rest_q = np.hstack((rest_data['q'][1], rest_data['q'][0]))

    # Load trunk equilibrium point
    outputModel = linearModel([tip_node], num_nodes, vel=False)
    Z_EQ = outputModel.evaluate(rest_q, qv=False)

    if robot == "trunk":
        Z_EQ[2] *= -1

    for task in control_tasks.value:
        simFolder = result_loc[robot][task]
        taskFile = task_file[robot][task]

        # Load control task
        taskParamsTrunk = load_data(taskFile)
        z_interp_trunk = interp1d(taskParamsTrunk['t'], taskParamsTrunk['z'], axis=0)

        robot_params = {
            "result_loc": simFolder,
            "sim_prefix": sim_prefix,
            "t0": 1.,
            "Z_EQ": Z_EQ,
            "z_interp": z_interp_trunk,
            "control_task": task,
            "constraints": None
        }

        error_metric_value[task], solve_time_value[task] = get_pareto_data(robot, robot_params, selected_metric.value, dt_params.value)
    return (
        Z_EQ,
        f,
        num_nodes,
        outputModel,
        rest_data,
        rest_file,
        rest_q,
        robot,
        robot_params,
        simFolder,
        t0,
        task,
        taskFile,
        taskParamsTrunk,
        tip_node,
        z_interp_trunk,
    )


@app.cell
def __(error_metric_value):
    # Store values for easy iteration
    error_metric_value
    return


@app.cell
def __(
    Line2D,
    color_legend,
    control_tasks,
    dt_params,
    error_metric_value,
    get_pareto_front,
    legend_name,
    linestyle_legend,
    marker_style,
    metric_legend,
    np,
    plt,
    remove_decimal,
    robot,
    selected_metric,
    solve_time_value,
):
    fig = plt.figure(figsize=(7, 3))

    all_points = []
    added_labels = set()
    legend_elements = []

    for _task in control_tasks.value:
        task_points = []
        legend_elements.append(Line2D([0], [0], marker=marker_style[_task], color='w', label=_task, markerfacecolor='gray', markersize=10))

        for _control in ["ssm", "linear", "koopman", "tpwl", "DMD"]:

            for delta_t in dt_params.value:
                _dt = remove_decimal(delta_t)
                label = legend_name[robot][_control]
                if label not in added_labels:
                    plt.scatter(solve_time_value[_task][_control][_dt], error_metric_value[_task][_control][_dt], color=color_legend[_control], alpha=0.7, marker=marker_style[_task], label=label)
                    added_labels.add(label)
                else:
                    plt.scatter(solve_time_value[_task][_control][_dt], error_metric_value[_task][_control][_dt], color=color_legend[_control], alpha=0.7, marker=marker_style[_task])

                task_points.append((solve_time_value[_task][_control][_dt], error_metric_value[_task][_control][_dt]))

                all_points.append((solve_time_value[_task][_control][_dt], error_metric_value[_task][_control][_dt]))

        # Pareto plot for each task
        pareto_x, pareto_y = get_pareto_front(task_points)
        plt.plot(pareto_x, pareto_y, color='black', linestyle=linestyle_legend[_task], label=f'Pareto Front for {_task}', alpha=0.5)

    all_solve_times = [point[0] for point in all_points]
    all_metric_vals = [point[1] for point in all_points]

    plt.xlim(np.min(all_solve_times)/2, max(all_solve_times)*2.0)
    plt.ylim(np.min(all_metric_vals)/10, max(all_metric_vals)*2.0)

    plt.title(f'Pareto Log-plot for simulated {robot}', fontsize="18")
    plt.yscale('log')  # Set y-axis to log scale
    plt.xscale('log')
    plt.ylabel(f'{metric_legend[selected_metric.value]}')
    plt.xlabel(f'Solve Time to Control Period')

    # Setup legend
    primary_legend = plt.legend(loc='upper left', bbox_to_anchor=(1, 1), handlelength=2.5, fontsize=9)
    # Add the additional custom legend
    custom_legend = plt.legend(handles=legend_elements, loc='lower left', bbox_to_anchor=(1, -0.02), ncol=len(control_tasks.value), fontsize=5.7, frameon=False)

    # Re-add the primary legend using add_artist()
    plt.gca().add_artist(primary_legend)

    plt.grid(True)
    plt.tight_layout()
    plt.show()

    fig
    return (
        added_labels,
        all_metric_vals,
        all_points,
        all_solve_times,
        custom_legend,
        delta_t,
        fig,
        label,
        legend_elements,
        pareto_x,
        pareto_y,
        primary_legend,
        task_points,
    )


@app.cell
def __():
    return


if __name__ == "__main__":
    app.run()
