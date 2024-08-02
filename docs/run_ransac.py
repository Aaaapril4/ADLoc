# %% Download test data
# !if [ -f demo.tar ]; then rm demo.tar; fi
# !if [ -d test_data ]; then rm -rf test_data; fi
# !wget -q https://github.com/AI4EPS/datasets/releases/download/test_data/test_data.tar
# !tar -xf test_data.tar

# %%
import json
import multiprocessing as mp
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyproj import Proj

from adloc.eikonal2d import init_eikonal2d
from adloc.sacloc2d import ADLoc
from adloc.utils import invert_location, invert_location_iter
from utils import plotting_ransac

# %%
if __name__ == "__main__":
    # %%
    # data_path = "test_data/ridgecrest/"
    # region = "synthetic"
    region = "ridgecrest"
    data_path = f"test_data/{region}/"

    ##################################### GaMMA Paper DATA #####################################
    os.system(
        f"[ -f {data_path}/events_osf.csv ] || curl -L https://osf.io/download/945dq/ -o {data_path}/events_osf.csv"
    )
    os.system(
        f"[ -f {data_path}/picks_osf.csv ] || curl -L https://osf.io/download/gwxtn/ -o {data_path}/picks_osf.csv"
    )
    os.system(
        f"[ -f {data_path}/stations_osf.csv ] || curl -L https://osf.io/download/km97w/ -o {data_path}/stations_osf.csv"
    )
    picks_file = os.path.join(data_path, "picks_osf.csv")
    events_file = os.path.join(data_path, "events_osf.csv")
    stations_file = os.path.join(data_path, "stations_osf.csv")

    picks = pd.read_csv(picks_file, sep="\t")
    picks.rename(
        {
            "id": "station_id",
            "timestamp": "phase_time",
            "type": "phase_type",
            "prob": "phase_score",
            "amp": "phase_amplitude",
            "event_idx": "event_index",
        },
        axis=1,
        inplace=True,
    )
    picks["phase_type"] = picks["phase_type"].str.upper()
    picks["phase_time"] = pd.to_datetime(picks["phase_time"])
    events_init = pd.read_csv(events_file, sep="\t")
    events_init.rename({"event_idx": "event_index"}, axis=1, inplace=True)
    events_init["depth_km"] = events_init["depth(m)"] / 1000.0
    events_init["time"] = pd.to_datetime(events_init["time"])

    picks = picks[picks["phase_time"] < pd.to_datetime("2019-07-05 00:00:00")]
    events_init = events_init[events_init["time"] < pd.to_datetime("2019-07-05 00:00:00")]

    stations = pd.read_csv(stations_file, sep="\t")
    stations.rename({"station": "station_id", "elevation(m)": "elevation_m"}, axis=1, inplace=True)
    ##################################### GaMMA Paper DATA #####################################

    # picks_file = os.path.join(data_path, "gamma_picks.csv")
    # events_file = os.path.join(data_path, "gamma_events.csv")
    # stations_file = os.path.join(data_path, "stations.csv")

    config = json.load(open(os.path.join(data_path, "config.json")))
    # picks = pd.read_csv(picks_file, parse_dates=["phase_time"])
    # events_init = pd.read_csv(events_file, parse_dates=["time"])
    # picks = pd.read_csv(os.path.join(data_path, "phasenet_plus_picks.csv"), parse_dates=["phase_time"])
    # stations = pd.read_csv(stations_file)
    stations["depth_km"] = -stations["elevation_m"] / 1000
    if "station_term_time" not in stations.columns:
        stations["station_term_time"] = 0.0
    if "station_term_amplitude" not in stations.columns:
        stations["station_term_amplitude"] = 0.0
    result_path = f"results/{region}/"
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    figure_path = f"figures/{region}/"
    if not os.path.exists(figure_path):
        os.makedirs(figure_path)

    # %%
    ## Automatic region; you can also specify a region
    lon0 = stations["longitude"].median()
    lat0 = stations["latitude"].median()
    proj = Proj(f"+proj=sterea +lon_0={lon0} +lat_0={lat0}  +units=km")
    stations[["x_km", "y_km"]] = stations.apply(
        lambda x: pd.Series(proj(longitude=x.longitude, latitude=x.latitude)), axis=1
    )
    stations["z_km"] = stations["elevation_m"].apply(lambda x: -x / 1e3)

    events_init[["x_km", "y_km"]] = events_init.apply(
        lambda x: pd.Series(proj(longitude=x.longitude, latitude=x.latitude)), axis=1
    )
    events_init["z_km"] = events_init["depth_km"]

    ## set up the config; you can also specify the region manually
    if ("xlim_km" not in config) or ("ylim_km" not in config) or ("zlim_km" not in config):

        # project minlatitude, maxlatitude, minlongitude, maxlongitude to ymin, ymax, xmin, xmax
        xmin, ymin = proj(config["minlongitude"], config["minlatitude"])
        xmax, ymax = proj(config["maxlongitude"], config["maxlatitude"])
        zmin = stations["z_km"].min()
        zmax = 20
        config = {}
        config["xlim_km"] = (xmin, xmax)
        config["ylim_km"] = (ymin, ymax)
        config["zlim_km"] = (zmin, zmax)

    config["vel"] = {"P": 6.0, "S": 6.0 / 1.73}

    # %%
    config["eikonal"] = None
    # ## Eikonal for 1D velocity model
    zz = [0.0, 5.5, 16.0, 32.0]
    vp = [5.5, 5.5, 6.7, 7.8]
    vp_vs_ratio = 1.73
    vs = [v / vp_vs_ratio for v in vp]
    h = 0.3
    vel = {"Z": zz, "P": vp, "S": vs}
    config["eikonal"] = {
        "vel": vel,
        "h": h,
        "xlim_km": config["xlim_km"],
        "ylim_km": config["ylim_km"],
        "zlim_km": config["zlim_km"],
    }
    config["eikonal"] = init_eikonal2d(config["eikonal"])

    config["use_amplitude"] = True

    # %% config for location
    config["min_picks"] = 8
    config["min_picks_ratio"] = 0.2
    config["max_residual_time"] = 1.0
    config["max_residual_amplitude"] = 1.0
    config["min_score"] = 0.6
    config["min_s_picks"] = 2
    config["min_p_picks"] = 2

    config["bfgs_bounds"] = (
        (config["xlim_km"][0] - 1, config["xlim_km"][1] + 1),  # x
        (config["ylim_km"][0] - 1, config["ylim_km"][1] + 1),  # y
        # (config["zlim_km"][0], config["zlim_km"][1] + 1),  # z
        (0, config["zlim_km"][1] + 1),
        (None, None),  # t
    )

    # %%
    plt.figure()
    plt.scatter(stations["x_km"], stations["y_km"], c=stations["depth_km"], cmap="viridis_r", s=100, marker="^")
    plt.colorbar(label="Depth (km)")
    plt.xlabel("X (km)")
    plt.ylabel("Y (km)")
    plt.xlim(config["xlim_km"])
    plt.ylim(config["ylim_km"])
    plt.title("Stations")
    plt.savefig(os.path.join(figure_path, "stations.png"), bbox_inches="tight", dpi=300)

    # %%
    mapping_phase_type_int = {"P": 0, "S": 1}
    config["vel"] = {mapping_phase_type_int[k]: v for k, v in config["vel"].items()}
    picks["phase_type"] = picks["phase_type"].map(mapping_phase_type_int)
    picks["phase_amplitude"] = picks["phase_amplitude"].apply(lambda x: np.log10(x) + 2.0)  # convert to log10(cm/s)

    # %%
    # stations["station_term"] = 0.0
    stations["idx_sta"] = stations.index  # reindex in case the index does not start from 0 or is not continuous
    picks = picks.merge(stations[["station_id", "idx_sta"]], on="station_id")

    if events_init is not None:
        events_init.reset_index(inplace=True)
        events_init["idx_eve"] = (
            events_init.index
        )  # reindex in case the index does not start from 0 or is not continuous
        picks = picks.merge(events_init[["event_index", "idx_eve"]], on="event_index")
    else:
        picks["idx_eve"] = picks["event_index"]

    # %%
    estimator = ADLoc(config, stations=stations[["x_km", "y_km", "z_km"]].values, eikonal=config["eikonal"])

    # %%
    NCPU = mp.cpu_count()
    MAX_SST_ITER = 10
    # MIN_SST_S = 0.01
    iter = 0
    events = None
    while iter < MAX_SST_ITER:
        # picks, events = invert_location_iter(picks, stations, config, estimator, events_init=events_init, iter=iter)
        picks, events = invert_location(picks, stations, config, estimator, events_init=events_init, iter=iter)
        # station_term = picks[picks["mask"] == 1.0].groupby("idx_sta").agg({"residual_time": "mean"}).reset_index()
        station_term_time = picks[picks["mask"] == 1.0].groupby("idx_sta").agg({"residual_time": "mean"}).reset_index()
        station_term_amp = (
            picks[picks["mask"] == 1.0].groupby("idx_sta").agg({"residual_amplitude": "mean"}).reset_index()
        )
        stations["station_term_time"] += (
            stations["idx_sta"].map(station_term_time.set_index("idx_sta")["residual_time"]).fillna(0)
        )
        stations["station_term_amplitude"] += (
            stations["idx_sta"].map(station_term_amp.set_index("idx_sta")["residual_amplitude"]).fillna(0)
        )
        ## Separate P and S station term
        # station_term = (
        #     picks[picks["mask"] == 1.0].groupby(["idx_sta", "phase_type"]).agg({"residual_s": "mean"}).reset_index()
        # )
        # stations["station_term_p"] = (
        #     stations["idx_sta"]
        #     .map(station_term[station_term["phase_type"] == 0].set_index("idx_sta")["residual_s"])
        #     .fillna(0)
        # )
        # stations["station_term_s"] = (
        #     stations["idx_sta"]
        #     .map(station_term[station_term["phase_type"] == 1].set_index("idx_sta")["residual_s"])
        #     .fillna(0)
        # )

        plotting_ransac(stations, figure_path, config, picks, events_init, events, iter=iter)

        if iter == 0:
            MIN_SST_S = (
                np.mean(np.abs(station_term_time["residual_time"])) / 10.0
            )  # break at 10% of the initial station term
            print(f"MIN_SST (s): {MIN_SST_S}")
        if np.mean(np.abs(station_term_time["residual_time"])) < MIN_SST_S:
            print(f"Mean station term: {np.mean(np.abs(station_term_time['residual_time']))}")
            # break
        iter += 1

    # %%
    picks.rename({"mask": "adloc_mask", "residual_s": "adloc_residual_s"}, axis=1, inplace=True)
    picks["phase_type"] = picks["phase_type"].map({0: "P", 1: "S"})
    picks.drop(["idx_eve", "idx_sta"], axis=1, inplace=True, errors="ignore")
    events[["longitude", "latitude"]] = events.apply(
        lambda x: pd.Series(proj(x["x_km"], x["y_km"], inverse=True)), axis=1
    )
    events["depth_km"] = events["z_km"]
    events.drop(["idx_eve", "x_km", "y_km", "z_km"], axis=1, inplace=True, errors="ignore")
    stations.drop(["idx_sta", "x_km", "y_km", "z_km"], axis=1, inplace=True, errors="ignore")
    # stations.rename({"station_term": "adloc_station_term_s"}, axis=1, inplace=True)
    picks.sort_values(["phase_time"], inplace=True)
    events.sort_values(["time"], inplace=True)
    picks.to_csv(os.path.join(result_path, "adloc_picks.csv"), index=False)
    events.to_csv(os.path.join(result_path, "adloc_events.csv"), index=False)
    stations.to_csv(os.path.join(result_path, "adloc_stations.csv"), index=False)

# %%
