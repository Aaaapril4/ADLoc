import os

import matplotlib.pyplot as plt
import numpy as np


# # %%
def plotting(stations, figure_path, config, picks, events_old, locations, station_term=None, suffix=""):

    vmin = min(locations["z_km"].min(), events_old["depth_km"].min())
    vmax = max(locations["z_km"].max(), events_old["depth_km"].max())
    # xmin, xmax = stations["x_km"].min(), stations["x_km"].max()
    # ymin, ymax = stations["y_km"].min(), stations["y_km"].max()
    xmin = min(stations["x_km"].min(), locations["x_km"].min())
    xmax = max(stations["x_km"].max(), locations["x_km"].max())
    ymin = min(stations["y_km"].min(), locations["y_km"].min())
    ymax = max(stations["y_km"].max(), locations["y_km"].max())
    zmin, zmax = config["zlim_km"]

    fig, ax = plt.subplots(2, 2, figsize=(10, 8), gridspec_kw={"height_ratios": [2, 1]})
    # fig, ax = plt.subplots(2, 3, figsize=(15, 8), gridspec_kw={"height_ratios": [2, 1]})
    im = ax[0, 0].scatter(
        locations["x_km"],
        locations["y_km"],
        c=locations["z_km"],
        cmap="viridis_r",
        s=1,
        marker="o",
        vmin=vmin,
        vmax=vmax,
    )
    ax[0, 0].set_xlim([xmin, xmax])
    ax[0, 0].set_ylim([ymin, ymax])
    cbar = fig.colorbar(im, ax=ax[0, 0])
    cbar.set_label("Depth (km)")
    ax[0, 0].set_title(f"ADLoc: {len(locations)} events")

    im = ax[0, 1].scatter(
        stations["x_km"],
        stations["y_km"],
        c=stations["station_term"],
        cmap="viridis_r",
        s=100,
        marker="^",
        alpha=0.5,
    )
    ax[0, 1].set_xlim([xmin, xmax])
    ax[0, 1].set_ylim([ymin, ymax])
    cbar = fig.colorbar(im, ax=ax[0, 1])
    cbar.set_label("Residual (s)")
    ax[0, 1].set_title(f"Station term: {np.mean(np.abs(stations['station_term'].values)):.4f} s")

    im = ax[1, 0].scatter(
        locations["x_km"],
        locations["z_km"],
        c=locations["z_km"],
        cmap="viridis_r",
        s=1,
        marker="o",
        vmin=vmin,
        vmax=vmax,
    )
    ax[1, 0].set_xlim([xmin, xmax])
    ax[1, 0].set_ylim([zmax, zmin])
    cbar = fig.colorbar(im, ax=ax[1, 0])
    cbar.set_label("Depth (km)")

    im = ax[1, 1].scatter(
        locations["y_km"],
        locations["z_km"],
        c=locations["z_km"],
        cmap="viridis_r",
        s=1,
        marker="o",
        vmin=vmin,
        vmax=vmax,
    )
    ax[1, 1].set_xlim([ymin, ymax])
    ax[1, 1].set_ylim([zmax, zmin])
    cbar = fig.colorbar(im, ax=ax[1, 1])
    cbar.set_label("Depth (km)")
    plt.savefig(os.path.join(figure_path, f"location_{suffix}.png"), bbox_inches="tight", dpi=300)
    plt.close(fig)


# %%
def plotting_dd(events_new, stations, config, figure_path, events_old, iter=0):

    vmin = min(events_new["z_km"].min(), events_old["z_km"].min())
    vmax = max(events_new["z_km"].max(), events_old["z_km"].max())
    xmin = min(stations["x_km"].min(), events_old["x_km"].min())
    xmax = max(stations["x_km"].max(), events_old["x_km"].max())
    ymin = min(stations["y_km"].min(), events_old["y_km"].min())
    ymax = max(stations["y_km"].max(), events_old["y_km"].max())
    zmin, zmax = config["zlim_km"]

    fig, ax = plt.subplots(2, 2, figsize=(10, 8), gridspec_kw={"height_ratios": [2, 1]})
    im = ax[0, 0].scatter(
        events_old["x_km"],
        events_old["y_km"],
        c=events_old["z_km"],
        cmap="viridis_r",
        s=1,
        marker="o",
        vmin=vmin,
        vmax=vmax,
        alpha=0.5,
    )
    ax[0, 0].set_xlim([xmin, xmax])
    ax[0, 0].set_ylim([ymin, ymax])
    cbar = fig.colorbar(im, ax=ax[0, 0])
    cbar.set_label("Depth (km)")
    ax[0, 0].set_title(f"ADLoc: {len(events_old)} events")

    im = ax[0, 1].scatter(
        events_new["x_km"],
        events_new["y_km"],
        c=events_new["z_km"],
        cmap="viridis_r",
        s=1,
        marker="o",
        vmin=vmin,
        vmax=vmax,
        alpha=0.5,
    )
    ax[0, 1].set_xlim([xmin, xmax])
    ax[0, 1].set_ylim([ymin, ymax])
    cbar = fig.colorbar(im, ax=ax[0, 1])
    cbar.set_label("Depth (km)")
    ax[0, 1].set_title(f"ADLoc DD: {len(events_new)} events")

    # im = ax[1, 0].scatter(
    #     events_new["x_km"],
    #     events_new["z_km"],
    #     c=events_new["z_km"],
    #     cmap="viridis_r",
    #     s=1,
    #     marker="o",
    #     vmin=vmin,
    #     vmax=vmax,
    # )
    # ax[1, 0].set_xlim([xmin, xmax])
    # ax[1, 0].set_ylim([zmax, zmin])
    # cbar = fig.colorbar(im, ax=ax[1, 0])
    # cbar.set_label("Depth (km)")

    im = ax[1, 0].scatter(
        events_old["y_km"],
        events_old["z_km"],
        c=events_old["z_km"],
        cmap="viridis_r",
        s=1,
        marker="o",
        vmin=vmin,
        vmax=vmax,
        alpha=0.5,
    )
    ax[1, 0].set_xlim([ymin, ymax])
    ax[1, 0].set_ylim([zmax, zmin])
    cbar = fig.colorbar(im, ax=ax[1, 0])
    cbar.set_label("Depth (km)")

    im = ax[1, 1].scatter(
        events_new["y_km"],
        events_new["z_km"],
        c=events_new["z_km"],
        cmap="viridis_r",
        s=1,
        marker="o",
        vmin=vmin,
        vmax=vmax,
        alpha=0.5,
    )
    ax[1, 1].set_xlim([ymin, ymax])
    ax[1, 1].set_ylim([zmax, zmin])
    cbar = fig.colorbar(im, ax=ax[1, 1])
    cbar.set_label("Depth (km)")
    plt.savefig(os.path.join(figure_path, f"location_{iter}.png"), bbox_inches="tight", dpi=300)


# %%
def plotting_ransac(stations, figure_path, config, picks, events_init, events, iter=0):
    fig, ax = plt.subplots(2, 3, figsize=(15, 8))
    ax[0, 0].hist(events["adloc_score"], bins=30, edgecolor="white")
    ax[0, 0].set_yscale("log")
    ax[0, 0].set_title("ADLoc score")
    ax[0, 1].hist(events["num_picks"], bins=30, edgecolor="white")
    ax[0, 1].set_title("Number of picks")
    ax[0, 2].hist(events["adloc_residual_amplitude"], bins=30, edgecolor="white")
    ax[0, 2].set_title("Event residual (log10 cm/s)")
    ax[1, 0].hist(events["adloc_residual_time"], bins=30, edgecolor="white")
    ax[1, 0].set_title("Event residual (s)")
    ax[1, 1].hist(picks[picks["mask"] == 1.0]["residual_time"], bins=30, edgecolor="white")
    ax[1, 1].set_title("Pick residual (s)")
    ax[1, 2].hist(picks[picks["mask"] == 1.0]["residual_amplitude"], bins=30, edgecolor="white")
    ax[1, 2].set_title("Pick residual (log10 cm/s)")
    plt.savefig(os.path.join(figure_path, f"hist_{iter}.png"), bbox_inches="tight", dpi=300)
    plt.close(fig)

    xmin, xmax = config["xlim_km"]
    ymin, ymax = config["ylim_km"]
    zmin, zmax = config["zlim_km"]
    vmin, vmax = config["zlim_km"]
    events = events.sort_values("time", ascending=True)
    s = max(0.1, min(10, 5000 / len(events)))
    alpha = 0.8
    fig, ax = plt.subplots(2, 3, figsize=(18, 8), gridspec_kw={"height_ratios": [2, 1]})
    # fig, ax = plt.subplots(2, 3, figsize=(15, 8), gridspec_kw={"height_ratios": [2, 1]})
    im = ax[0, 0].scatter(
        events["x_km"],
        events["y_km"],
        c=events["z_km"],
        cmap="viridis_r",
        s=s,
        marker="o",
        vmin=vmin,
        vmax=vmax,
        alpha=alpha,
        linewidth=0.,
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
        c=stations["station_term_time"],
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
    ax[0, 1].set_title(f"Station term: {np.mean(np.abs(stations['station_term_time'].values)):.4f} s")

    im = ax[0, 2].scatter(
        stations["x_km"],
        stations["y_km"],
        c=stations["station_term_amplitude"],
        cmap="viridis_r",
        s=100,
        marker="^",
        alpha=alpha,
    )
    ax[0, 2].set_aspect("equal", "box")
    ax[0, 2].set_xlim([xmin, xmax])
    ax[0, 2].set_ylim([ymin, ymax])
    ax[0, 2].set_xlabel("X (km)")
    ax[0, 2].set_ylabel("Y (km)")
    cbar = fig.colorbar(im, ax=ax[0, 2])
    cbar.set_label("Residual (log10 cm/s)")
    ax[0, 2].set_title(f"Station term: {np.mean(np.abs(stations['station_term_amplitude'].values)):.4f} s")

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
        s=s,
        marker="o",
        vmin=vmin,
        vmax=vmax,
        alpha=alpha,
        linewidth=0.,
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
        s=s,
        marker="o",
        vmin=vmin,
        vmax=vmax,
        alpha=alpha,
        linewidth=0.,
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
