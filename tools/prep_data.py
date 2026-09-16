"""
Build the static data bundle for the CO2LOGIX UK web explorer.

Reads the same inputs as the published CO2LOGIX release (Github/input_data)
plus the UK outline and UK basin outlines from CO2LOGIX v1.0/Shapefiles,
and writes a single self-contained `data.js` for the browser app.

Everything that needs geopandas / CoolProp is done here, once.
The browser then only needs arithmetic.
"""

import base64
import json
import os
import sys

import numpy as np
import pandas as pd
import geopandas as gpd
import CoolProp.CoolProp as CP
from shapely.geometry import Polygon, MultiPolygon

HERE = os.path.dirname(os.path.abspath(__file__))
# Search upward for the CO2LOGIX inputs, so this works whether the explorer sits in
# the model repo (docs/tools/ -> ../../input_data) or beside it in a working folder.
ROOTS = [os.environ["CO2LOGIX_ROOT"]] if os.environ.get("CO2LOGIX_ROOT") else [
    os.path.abspath(os.path.join(HERE, "..", "..")),        # <repo>/docs/tools -> <repo>
    os.path.abspath(os.path.join(HERE, "..", "..", "..")),  # one level further out
]


def find(*candidates):
    """First existing path built from ROOTS x candidate relative paths."""
    tried = []
    for root in ROOTS:
        for rel in candidates:
            p = os.path.join(root, *rel)
            tried.append(p)
            if os.path.exists(p):
                return p
    raise SystemExit(
        "Could not locate a required CO2LOGIX input. Looked in:\n  "
        + "\n  ".join(tried)
        + "\n\nSet CO2LOGIX_ROOT to the folder holding input_data/ (and, for the UK\n"
          "coastline, Shapefiles/uk_project.shp)."
    )


SHP_AQ = find(("input_data", "saline_aquifers.shp"),
              ("Github", "input_data", "saline_aquifers.shp"))
CSV_IN = find(("input_data", "input_table.csv"),
              ("Github", "input_data", "input_table.csv"))
SHP_UK = find(("input_data", "uk_project.shp"),
              ("Shapefiles", "uk_project.shp"),
              ("CO2LOGIX v1.0", "Shapefiles", "uk_project.shp"))

CRS = "EPSG:32631"          # model CRS, metres
WELL_LATTICE = 1000         # m - candidate well lattice (as in example.py)
RASTER_CELL = 2500          # m - display raster cell size
POOL_SIZE = 15000           # candidate well locations shipped to the browser
                            # (covers the 14,000-well ceiling plus ceil() overhead)
POOL_SEED = 42              # matches random_state=42 in model/pressure.py

# Reference parameters from Github/example.py
H_GRAD = 10.0    # hydrostatic gradient, MPa/km
L_GRAD = 23.0    # lithostatic gradient, MPa/km
T_GRAD = 25.0    # temperature gradient, degC/km
T_SURF = 15.0    # surface / seabed temperature, degC
C_W = 5e-10      # brine compressibility, 1/Pa


def estimate_frac_pres(sv, pp):
    """Zhang & Yin (2017): frac pressure ~ pp + 0.75 * (sv - pp)."""
    return pp + 0.75 * (sv - pp)


# --------------------------------------------------------------------------
# 1. Aquifers + properties  (mirrors example.py)
# --------------------------------------------------------------------------

S = gpd.read_file(SHP_AQ).to_crs(CRS)
tbl = pd.read_csv(CSV_IN)

S["unit"] = tbl["Name"].values
S["phi"] = tbl["Porosity"].values
S["h"] = (tbl["Gross thickness"] * tbl["Net-to-gross"]).values
S["k_md"] = tbl["Permeability"].values
S["z"] = tbl["Depth"].values

S["k"] = S["k_md"] * 9.869233e-16
# Hall (1953) rock compressibility
S["c_r"] = ((1.782 / S["phi"] ** 0.438) * 1e-6) * 145.038 * 1e-6
S["c_tot"] = S["c_r"] + S["phi"] * C_W

S["p_ref"] = H_GRAD * S["z"] / 1000.0
S["t_ref"] = T_GRAD * S["z"] / 1000.0 + T_SURF

T_K = S["t_ref"].values + 273.15
P_PA = S["p_ref"].values * 1e6
S["u_w"] = [CP.PropsSI("V", "T", t, "P", p, "Water") for t, p in zip(T_K, P_PA)]
S["u_c"] = [CP.PropsSI("V", "T", t, "P", p, "CO2") for t, p in zip(T_K, P_PA)]
S["rho_c"] = [CP.PropsSI("D", "T", t, "P", p, "CO2") for t, p in zip(T_K, P_PA)]

S["gamma"] = S["u_c"] / S["u_w"]
S["omega"] = (S["u_c"] + S["u_w"]) / (S["u_c"] - S["u_w"]) * np.log(np.sqrt(S["gamma"])) - 1.0
S["D"] = S["k"] / (S["c_tot"] * S["u_w"])

S["sv"] = L_GRAD * S["z"] / 1000.0
S["p_frac"] = estimate_frac_pres(S["sv"], S["p_ref"])

S["area_m2"] = S.geometry.area
S["well_fraction"] = S["area_m2"] / S["area_m2"].sum()

print("Aquifer properties")
print(S[["unit", "phi", "h", "k_md", "z", "rho_c", "u_w", "u_c",
         "D", "p_ref", "p_frac"]].to_string(float_format=lambda v: f"{v:.4g}"))

# --------------------------------------------------------------------------
# 2. Model domain / grids
# --------------------------------------------------------------------------

xmin, ymin, xmax, ymax = S.total_bounds
# snap outwards to whole raster cells
x0 = np.floor(xmin / RASTER_CELL) * RASTER_CELL
y0 = np.floor(ymin / RASTER_CELL) * RASTER_CELL
nx = int(np.ceil((xmax - x0) / RASTER_CELL))
ny = int(np.ceil((ymax - y0) / RASTER_CELL))
print(f"\nDisplay raster: {nx} x {ny} cells @ {RASTER_CELL} m  ({nx*ny:,} cells)")

rx = x0 + RASTER_CELL / 2 + np.arange(nx) * RASTER_CELL
ry = y0 + RASTER_CELL / 2 + np.arange(ny) * RASTER_CELL
gx, gy = np.meshgrid(rx, ry)

cells = gpd.GeoDataFrame(
    geometry=gpd.points_from_xy(gx.ravel(), gy.ravel()), crs=CRS
)
joined = gpd.sjoin(cells, S[["geometry"]], how="left", predicate="within")
joined = joined[~joined.index.duplicated(keep="first")]
aq_idx = joined["index_right"].to_numpy()
aq_idx = np.where(np.isnan(aq_idx), -1, aq_idx).astype(np.int8)
print(f"Raster cells inside an aquifer: {(aq_idx >= 0).sum():,}")

# --------------------------------------------------------------------------
# 3. Candidate well pool on the 1 km lattice (area-weighted, as in the model)
# --------------------------------------------------------------------------

wx = np.arange(xmin + WELL_LATTICE / 2, xmax, WELL_LATTICE)
wy = np.arange(ymin + WELL_LATTICE / 2, ymax, WELL_LATTICE)
wgx, wgy = np.meshgrid(wx, wy)
wpts = gpd.GeoDataFrame(geometry=gpd.points_from_xy(wgx.ravel(), wgy.ravel()), crs=CRS)
wj = gpd.sjoin(wpts, S[["well_fraction", "geometry"]], how="inner", predicate="within")
wj = wj[~wj.index.duplicated(keep="first")]
print(f"1 km lattice points inside aquifers: {len(wj):,}")

def make_pool(weights, label):
    """Draw a candidate well pool without replacement, and report its split."""
    pool = wj.sample(
        n=min(POOL_SIZE, len(wj)),
        weights=weights,
        replace=False,
        random_state=POOL_SEED,
    )
    aq = pool["index_right"].to_numpy().astype(np.int8)
    print(f"  {label}: " + ", ".join(
        f"{S['unit'][i].split()[0]} {int((aq == i).sum())}" for i in range(len(S))))
    return {
        # km offsets from (x0, y0) at 1 km precision -> int16 is ample
        "x": np.round((pool.geometry.x.to_numpy() - x0) / 1000.0).astype(np.int16),
        "y": np.round((pool.geometry.y.to_numpy() - y0) / 1000.0).astype(np.int16),
        "aq": aq,
    }


# Well placement weighting.
#   "area"      - unweighted draw from the uniform 1 km lattice, so wells land in
#                 proportion to unit area. This is what example.py's own comment
#                 describes: "proportion of total wells to place in each aquifer
#                 outline (based on area)".
#   "published" - every lattice point carries its unit's area fraction as its
#                 sampling weight, as in pressure.py. Since each unit already
#                 contributes points in proportion to its area, this places wells
#                 in proportion to area SQUARED, starving the smaller units.
WELL_WEIGHTING = "area"

print(f"Candidate well pool ({POOL_SIZE:,} locations, weighting: {WELL_WEIGHTING}):")
pool = make_pool(
    wj["well_fraction"] if WELL_WEIGHTING == "published" else None,
    WELL_WEIGHTING,
)

# --------------------------------------------------------------------------
# 4. Vector context layers
# --------------------------------------------------------------------------

def rings(gdf, tol, min_area_km2=0.0, props=None):
    """Simplify + flatten to lists of [x_km, y_km] rings."""
    out = []
    for i, row in gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        geom = geom.simplify(tol, preserve_topology=True)
        parts = geom.geoms if isinstance(geom, MultiPolygon) else [geom]
        feat_rings = []
        for part in parts:
            if not isinstance(part, Polygon) or part.is_empty:
                continue
            if part.area / 1e6 < min_area_km2:
                continue
            xs, ys = part.exterior.coords.xy
            ring = [
                [round(float(x) / 1000.0, 2), round(float(y) / 1000.0, 2)]
                for x, y in zip(xs, ys)
            ]
            if len(ring) >= 4:
                feat_rings.append(ring)
        if not feat_rings:
            continue
        item = {"rings": feat_rings}
        if props:
            item.update({k: row[v] for k, v in props.items()})
        out.append(item)
    return out


aq_geo = rings(S, tol=400, props={"name": "unit"})

uk = gpd.read_file(SHP_UK)
uk = uk.set_crs("EPSG:3857", allow_override=True).to_crs(CRS)
uk_geo = rings(uk, tol=1200, min_area_km2=25.0)

print(f"\nVector layers: {len(aq_geo)} aquifers, "
      f"{sum(len(f['rings']) for f in uk_geo)} UK rings")

# --------------------------------------------------------------------------
# 5. Encode + write
# --------------------------------------------------------------------------

def b64(arr):
    return base64.b64encode(np.ascontiguousarray(arr).tobytes()).decode("ascii")


def rle(arr):
    """Run-length encode the int8 aquifer-index raster -> [value, count, ...]."""
    a = np.asarray(arr).ravel()
    changes = np.flatnonzero(np.diff(a)) + 1
    starts = np.concatenate(([0], changes))
    ends = np.concatenate((changes, [a.size]))
    out = []
    for s, e in zip(starts, ends):
        out.append(int(a[s]))
        out.append(int(e - s))
    return out


aq_rle = rle(aq_idx)
print(f"Aquifer raster RLE runs: {len(aq_rle)//2:,}")

data = {
    "meta": {
        "crs": CRS,
        "source": "CO2LOGIX v1.0 (Github release): model/pressure.py, model/growth.py, model/utils.py",
        "hydrostaticGradient": H_GRAD,
        "lithostaticGradient": L_GRAD,
        "tempGradient": T_GRAD,
        "surfaceTemp": T_SURF,
        "brineCompressibility": C_W,
    },
    "grid": {
        "x0": float(x0), "y0": float(y0),
        "nx": nx, "ny": ny, "cell": RASTER_CELL,
        "aquiferRLE": aq_rle,
    },
    "units": [
        {
            "name": str(S["unit"][i]),
            "phi": float(S["phi"][i]),
            "h": float(S["h"][i]),
            "k_md": float(S["k_md"][i]),
            "k": float(S["k"][i]),
            "z": float(S["z"][i]),
            "areaKm2": float(S["area_m2"][i] / 1e6),
            "wellFraction": float(S["well_fraction"][i]),
            "c_r": float(S["c_r"][i]),
            "c_tot": float(S["c_tot"][i]),
            "p_ref": float(S["p_ref"][i]),
            "t_ref": float(S["t_ref"][i]),
            "sv": float(S["sv"][i]),
            "p_frac": float(S["p_frac"][i]),
            "u_w": float(S["u_w"][i]),
            "u_c": float(S["u_c"][i]),
            "rho_c": float(S["rho_c"][i]),
            "gamma": float(S["gamma"][i]),
            "omega": float(S["omega"][i]),
            "D": float(S["D"][i]),
        }
        for i in range(len(S))
    ],
    "wellPool": {
        "n": int(len(pool["x"])),
        "weighting": WELL_WEIGHTING,
        "x": b64(pool["x"]),    # int16, km east of grid.x0
        "y": b64(pool["y"]),    # int16, km north of grid.y0
        "aq": b64(pool["aq"]),  # int8 storage unit index
    },
    "geo": {"aquifers": aq_geo, "uk": uk_geo},
}

out_dir = sys.argv[1] if len(sys.argv) > 1 else "."
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "data.js")
with open(out_path, "w", encoding="utf-8") as f:
    f.write("// Generated by prep_data.py - do not edit by hand.\n")
    f.write("window.CO2LOGIX_DATA = ")
    json.dump(data, f, separators=(",", ":"))
    f.write(";\n")

print(f"\nWrote {out_path}  ({os.path.getsize(out_path)/1024:.0f} KB)")
