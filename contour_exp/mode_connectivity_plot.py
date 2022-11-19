import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np
import argparse
import json
import os

_BASE_PATH = ".."


def save(name, data):
    if not os.path.exists(f"{_BASE_PATH}/graphics/{data}"):
        os.makedirs(f"{_BASE_PATH}/graphics/{data}")

    plt.tight_layout()
    plt.savefig(f"{_BASE_PATH}/graphics/{data}/{name}.pdf",)
    plt.savefig(f"{_BASE_PATH}/graphics/{data}/{name}.png",dpi=600)


def rescale_model_coords(model_coordinates, start, grid_size, width):
    model_coordinates -= start
    model_coordinates *= grid_size / width
    return model_coordinates


def mc_vis(result_mat, model_coordinates, ax, start, width, grid_size, levels, result_idx=(0, 1),
           plot_type="loss", labels=None):
    left_contour_color="r"
    right_contour_color="dimgrey"
    tj_color="k"
    mcx, mcy = model_coordinates[:, 0], model_coordinates[:, 1]

    ternary_idx = 1 if plot_type == "loss" else 0
    mat_1 = result_mat[:, :, ternary_idx, result_idx[0]]
    mat_2 = result_mat[:, :, ternary_idx, result_idx[1]]

    plot = ax.contour(mat_1.T, levels=levels, colors=left_contour_color, alpha=1)
    ax.clabel(plot, levels, inline=True, fmt='%2.2f', fontsize=10)

    plot = ax.contour(mat_2.T, levels=levels, colors=right_contour_color, alpha=1)
    ax.clabel(plot, levels, inline=True, fmt='%2.2f', fontsize=10)



    nb_ticks = 5
    ticks = np.linspace(0, grid_size, nb_ticks)
    tick_labels = ([f"${i}$" for i in np.linspace(start, start + width, nb_ticks)])
    labels = ["$w_j$", "$w_1$", "$w_2$"] if labels is None else labels

    ax.set_xticks(ticks)
    ax.set_xticklabels(tick_labels)
    ax.set_yticks(ticks)
    ax.set_yticklabels(tick_labels)
    ax.scatter(mcx, mcy, c=tj_color)

    for (x, y, s) in zip(mcx, mcy, labels):
        ax.text(x, y + grid_size // 10, s, fontsize=25,c=tj_color)


def plot_path(name, ax, data, start, width, grid_size):
    path = np.load(f"{_BASE_PATH}/models/{data}/{name}_path.npy")
    path -= start

    path *= grid_size / width

    ax.plot(path[:, 0], path[:, 1], '-x', c='k', linewidth=2)

    # ax.plot(path[:19, 0], path[:19, 1], '-x', c='k', linewidth=2)
    # ax.plot(path[19-1:, 0], path[19-1:, 1], '-x', c='y', linewidth=2)
    # ax.plot(path[19+18-1:, 0], path[19+18-1:, 1], '-x', c='g', linewidth=2)

def main():

    # Turn on for tex (labels might have to be set manually)
    # rc('text', usetex=True)
    # rc('font', size=16)

    parser = argparse.ArgumentParser()
    parser.add_argument('name')
    parser.add_argument('--data', choices=["mnist", "cifar", "min","cifar100"], default="mnist")
    parser.add_argument('--path', nargs='*', help="Names of paths")
    parser.add_argument('--aug_type',default="", choices=["None","Both"])
    parser.add_argument("--mem_iter",default=1,type=int)
    parser.add_argument('--exp',default="")
    args = parser.parse_args()

    result_mat = np.load(f"{_BASE_PATH}/results/{args.data}/{args.name}.npy")
    model_coordinates = np.load(f"{_BASE_PATH}/results/{args.data}/{args.name}_models.npy")

    with open(f"{_BASE_PATH}/results/{args.data}/{args.name}.json") as f:
        settings = json.load(f)

    if args.data == "mnist":
        levels = [0.01, 0.1, 0.15, 0.25, 0.5, 1.0]  # MNIST
    elif args.data == "cifar":
        levels = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]  # CIFAR
    elif args.data == "cifar100":
        #levels = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]  # CIFA
       # levels = [0.5, 0.6, 0.7, 0.8, 1.0,  1.2, 1.4,1.6,1.8]  # CIFA
       levels = [0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,]  # MIN
       #  levels = [0.5, 1.0, 2.0,2.5, 3.0, 3.5,4.0, 5.0,]  # MIN
    elif args.data == "min":
        levels = [0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0,10.0]  # MIN
    else:
        raise ValueError("Data unknown")

    model_coordinates = rescale_model_coords(model_coordinates, settings["start"], settings["grid"], settings["width"])

    ax_width, ax_height = 2, 1
    fig, axes = plt.subplots(ax_height, ax_width, figsize=(4 * ax_width, 4 * ax_height))
    if(args.aug_type == "Both"):
        tex_labels = ["$w_1$", "$w_{2,ft}$", "$w_{2,rar}$"]
    else:
        if(args.mem_iter ==1):
            tex_labels = ["$w_1$", "$w_{2,ft}$", "$w_{2,er}$"]
        else:

            tex_labels = ["$w_1$", "$w_{2,ft}$", "$w_{2,rer}$"]

    left_contour_color="r"
    right_contour_color="dimgrey"
    mc_vis(result_mat, model_coordinates, axes[0], settings['start'], settings['width'], settings['grid'], levels,
           result_idx=(0, 1), plot_type="loss", labels=tex_labels)
    proxy = [plt.Rectangle((0, 0), 1, 1, fc=left_contour_color),plt.Rectangle((0, 0), 1, 1, fc=right_contour_color)]
    axes[0].legend(proxy, ["T1 test","T2 test"],loc="upper left",bbox_to_anchor=[-0.04,1.18],ncol=2,fontsize=14)
    mc_vis(result_mat, model_coordinates, axes[1], settings['start'], settings['width'], settings['grid'], levels,
           result_idx=(2, 1), plot_type="loss", labels=tex_labels)
    axes[1].legend(proxy, ["T1 mem","T2 test"],loc="upper left",bbox_to_anchor=[-0.04,1.18],ncol=2,fontsize=14)

    if args.path is not None:
        for path in args.path:
            #print(path)
            plot_path(path, axes[0], args.data, settings["start"], settings["width"], settings["grid"])
            plot_path(path, axes[1], args.data, settings["start"], settings["width"], settings["grid"])

    save("contour_ER_t1_t2_v2", args.data+args.exp)
    plt.show()


if __name__ == '__main__':
    main()
