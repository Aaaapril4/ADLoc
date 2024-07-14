import os

import matplotlib.pyplot as plt
import numpy as np


# %%
def plotting(stations, figure_path, config, picks, events_init, events, station_term, iter=0):
    fig, ax = plt.subplots(2, 2, figsize=(10, 8))
    ax[0, 0].hist(events["adloc_score"], bins=30, edgecolor="white")
    ax[0, 0].set_yscale("log")
    ax[0, 0].set_title("ADLoc score")
    ax[0, 1].hist(events["num_picks"], bins=30, edgecolor="white")
    ax[0, 1].set_title("Number of picks")
    ax[1, 0].hist(events["adloc_residual_time"], bins=30, edgecolor="white")
    ax[1, 0].set_title("Event residual (s)")
    ax[1, 1].hist(picks[picks["mask"] == 1.0]["residual_time"], bins=30, edgecolor="white")
    ax[1, 1].set_title("Pick residual (s)")
    plt.savefig(os.path.join(figure_path, f"hist_{iter}.png"), bbox_inches="tight", dpi=300)

    xmin, xmax = config["xlim_km"]
    ymin, ymax = config["ylim_km"]
    zmin, zmax = config["zlim_km"]
    vmin, vmax = config["zlim_km"]
    alpha = 0.8
    fig, ax = plt.subplots(2, 2, figsize=(12, 8), gridspec_kw={"height_ratios": [2, 1]})
    # fig, ax = plt.subplots(2, 3, figsize=(15, 8), gridspec_kw={"height_ratios": [2, 1]})
    im = ax[0, 0].scatter(
        events["x_km"],
        events["y_km"],
        c=events["z_km"],
        cmap="viridis_r",
        s=1,
        marker="o",
        vmin=vmin,
        vmax=vmax,
        alpha=alpha,
    )
    # set ratio 1:1
    ax[0, 0].set_aspect("equal", "box")
    ax[0, 0].set_xlim([xmin, xmax])
    ax[0, 0].set_ylim([ymin, ymax])
    ax[0, 0].set_xlabel("X (km)")
    ax[0, 0].set_ylabel("Y (km)")
    cbar = fig.colorbar(im, ax=ax[0, 0])
    cbar.set_label("Depth (km)")
    ax[0, 0].set_title(f"ADLoc: {len(events)} events")

    im = ax[0, 1].scatter(
        stations["x_km"],
        stations["y_km"],
        c=stations["station_term"],
        cmap="viridis_r",
        s=100,
        marker="^",
        alpha=alpha,
    )
    ax[0, 1].set_aspect("equal", "box")
    ax[0, 1].set_xlim([xmin, xmax])
    ax[0, 1].set_ylim([ymin, ymax])
    ax[0, 1].set_xlabel("X (km)")
    ax[0, 1].set_ylabel("Y (km)")
    cbar = fig.colorbar(im, ax=ax[0, 1])
    cbar.set_label("Residual (s)")
    ax[0, 1].set_title(f"Station term: {np.mean(np.abs(stations['station_term'].values)):.4f} s")

    ## Separate P and S station term
    # im = ax[0, 1].scatter(
    #     stations["x_km"],
    #     stations["y_km"],
    #     c=stations["station_term_p"],
    #     cmap="viridis_r",
    #     s=100,
    #     marker="^",
    #     alpha=0.5,
    # )
    # ax[0, 1].set_xlim([xmin, xmax])
    # ax[0, 1].set_ylim([ymin, ymax])
    # cbar = fig.colorbar(im, ax=ax[0, 1])
    # cbar.set_label("Residual (s)")
    # ax[0, 1].set_title(f"Station term (P): {np.mean(np.abs(stations['station_term_p'].values)):.4f} s")

    # im = ax[0, 2].scatter(
    #     stations["x_km"],
    #     stations["y_km"],
    #     c=stations["station_term_s"],
    #     cmap="viridis_r",
    #     s=100,
    #     marker="^",
    #     alpha=0.5,
    # )
    # ax[0, 2].set_xlim([xmin, xmax])
    # ax[0, 2].set_ylim([ymin, ymax])
    # cbar = fig.colorbar(im, ax=ax[0, 2])
    # cbar.set_label("Residual (s)")
    # ax[0, 2].set_title(f"Station term (S): {np.mean(np.abs(stations['station_term_s'].values)):.4f} s")

    im = ax[1, 0].scatter(
        events["x_km"],
        events["z_km"],
        c=events["z_km"],
        cmap="viridis_r",
        s=1,
        marker="o",
        vmin=vmin,
        vmax=vmax,
        alpha=alpha,
    )
    # ax[1, 0].set_aspect("equal", "box")
    ax[1, 0].set_xlim([xmin, xmax])
    ax[1, 0].set_ylim([zmax, zmin])
    ax[1, 0].set_xlabel("X (km)")
    # ax[1, 0].set_ylabel("Depth (km)")
    cbar = fig.colorbar(im, ax=ax[1, 0])
    cbar.set_label("Depth (km)")

    im = ax[1, 1].scatter(
        events["y_km"],
        events["z_km"],
        c=events["z_km"],
        cmap="viridis_r",
        s=1,
        marker="o",
        vmin=vmin,
        vmax=vmax,
        alpha=alpha,
    )
    # ax[1, 1].set_aspect("equal", "box")
    ax[1, 1].set_xlim([ymin, ymax])
    ax[1, 1].set_ylim([zmax, zmin])
    ax[1, 1].set_xlabel("Y (km)")
    # ax[1, 1].set_ylabel("Depth (km)")
    cbar = fig.colorbar(im, ax=ax[1, 1])
    cbar.set_label("Depth (km)")
    plt.savefig(os.path.join(figure_path, f"location_{iter}.png"), bbox_inches="tight", dpi=300)
    plt.close(fig)
