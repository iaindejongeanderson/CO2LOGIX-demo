"""
Check that the fast well-to-well superposition used by the web app reproduces
the full-grid nanmax produced by the published CO2LOGIX model.

Run from the Github release folder so `from model.pressure import run_model`
resolves to the published code.
"""

import os
import sys
sys.path.insert(0, os.getcwd())

import numpy as np
import pandas as pd
import geopandas as gpd
import CoolProp.CoolProp as CP
from shapely.geometry import box, Point
from collections import defaultdict

from model.pressure import run_model
from model.growth import solve_tm, generate_well_schedule_logistic
from model.utils import estimate_frac_pres

params = {}
params['L'] = 100
params['grid_size'] = 1000
params['h_grad'] = 10
params['l_grad'] = 23
params['t_grad'] = 25
params['t_surf'] = 15
params['c_w'] = 5E-10
params['inj_rate'] = 1
params['model_years'] = 60
params['inj_years'] = 25
params['domain'] = 'open'
params['rc'] = 10000
params['k_growth'] = 0.13

S_gdf = gpd.read_file(r'Input_data\saline_aquifers.shp')
data = pd.read_csv(r'Input_data\input_table.csv')
S_gdf['phi'] = data['Porosity']
S_gdf['h'] = data['Gross thickness'] * data['Net-to-gross']
S_gdf['k_md'] = data['Permeability']
S_gdf['z'] = data['Depth']

params['xmin'], params['ymin'], params['xmax'], params['ymax'] = S_gdf.total_bounds
x_coords = np.arange(params['xmin'] + params['grid_size'] / 2.0, params['xmax'], params['grid_size'])
y_coords = np.arange(params['ymin'] + params['grid_size'] / 2.0, params['ymax'], params['grid_size'])
params['gx'], params['gy'] = np.meshgrid(x_coords, y_coords)
S_gdf['area_m2'] = S_gdf.geometry.area
S_gdf['well_fraction'] = S_gdf['area_m2'] / S_gdf['area_m2'].sum()

S_gdf['k'] = S_gdf['k_md'] * 9.869233e-16
S_gdf['c_r'] = ((1.782/S_gdf['phi']**0.438)*1E-6)*145.038*1E-6
S_gdf['p_ref'] = params['h_grad'] * S_gdf['z']/1000
S_gdf['t_ref'] = params['t_grad'] * S_gdf['z']/1000 + params['t_surf']
S_gdf['u_w'] = CP.PropsSI('V', 'T', S_gdf['t_ref'].values + 273.15, 'P', S_gdf['p_ref'].values * 1e6, 'Water')
S_gdf['u_c'] = CP.PropsSI('V', 'T', S_gdf['t_ref'].values + 273.15, 'P', S_gdf['p_ref'].values * 1e6, 'CO2')
S_gdf['rho_c'] = CP.PropsSI('D', 'T', S_gdf['t_ref'].values + 273.15, 'P', S_gdf['p_ref'].values * 1e6, 'CO2')
S_gdf['Q'] = params['inj_rate'] * 1e9 / S_gdf['rho_c'] / 365 / 86400
S_gdf['c_tot'] = S_gdf['c_r'] + S_gdf['phi'] * params['c_w']
S_gdf['gamma'] = S_gdf['u_c'] / S_gdf['u_w']
S_gdf['omega'] = ((S_gdf['u_c'] + S_gdf['u_w']) / (S_gdf['u_c'] - S_gdf['u_w']) * np.log(np.sqrt(S_gdf['gamma'])) - 1.0)
S_gdf['D'] = S_gdf['k']/(S_gdf['c_tot']*S_gdf['u_w'])
S_gdf['p_c'] = (S_gdf['Q'] * S_gdf['u_w']) / (2.0 * np.pi * S_gdf['h'] * S_gdf['k']) / 1e6
S_gdf['sv'] = params['l_grad'] * (S_gdf['z'] / 1000)
S_gdf['p_frac'] = estimate_frac_pres(S_gdf['sv'], S_gdf['p_ref'])

x_flat = params['gx'].ravel()
y_flat = params['gy'].ravel()
half = params['grid_size'] / 2
polygons = [box(x - half, y - half, x + half, y + half) for x, y in zip(x_flat, y_flat)]
grid_gdf = gpd.GeoDataFrame({'geometry': polygons}, crs=S_gdf.crs)
pfrac_join = gpd.sjoin(grid_gdf, S_gdf[['geometry', 'p_frac']], how='left', predicate='intersects').drop_duplicates(subset='geometry')
pref_join = gpd.sjoin(grid_gdf, S_gdf[['geometry', 'p_ref']], how='left', predicate='intersects').drop_duplicates(subset='geometry')
params['p_frac_grid'] = pfrac_join['p_frac'].values.reshape(params['gx'].shape)
params['p_ref_grid'] = pref_join['p_ref'].values.reshape(params['gx'].shape)

params['t0'] = solve_tm(params['L'], params['k_growth'], 1, 2)
print("t0 =", params['t0'])

# ---- reference: published model -----------------------------------------
import time
t = time.time()
all_dP, max_dP_ref, wells_per_year, capacity_ref = run_model(S_gdf, params)
print(f"published model: {time.time()-t:.1f} s, {len(max_dP_ref)} years")


# ---- fast method: superposition evaluated at well locations only ---------
def fast_max(S_gdf, params):
    grid_points = gpd.points_from_xy(x=params['gx'].ravel(), y=params['gy'].ravel())
    points_gdf = gpd.GeoDataFrame(geometry=grid_points, crs=S_gdf.crs)
    pwp = gpd.sjoin(points_gdf, S_gdf[['well_fraction', 'geometry']], how='inner', predicate='within')
    sampled = pwp.sample(n=params['L']+100, weights=pwp['well_fraction'], replace=False, random_state=42)
    cand = np.column_stack((sampled.geometry.x.to_numpy(), sampled.geometry.y.to_numpy()))
    used = [0]
    wells, wpy = generate_well_schedule_logistic(params['model_years'], params['L'],
                                                 params['k_growth'], params['t0'],
                                                 params['inj_years'], cand, used)
    for w in wells:
        w['aquifer_idx'] = S_gdf.contains(Point(np.array([w['x'], w['y']]))).tolist().index(True)

    D_a = S_gdf['D'].to_numpy(); Q_a = S_gdf['Q'].to_numpy()
    phi_a = S_gdf['phi'].to_numpy(); h_a = S_gdf['h'].to_numpy()
    om_a = S_gdf['omega'].to_numpy(); ga_a = S_gdf['gamma'].to_numpy()
    pc_a = S_gdf['p_c'].to_numpy()
    pref_a = S_gdf['p_ref'].to_numpy(); pfrac_a = S_gdf['p_frac'].to_numpy()

    years = range(min(w['start_year'] for w in wells),
                  max(w['start_year'] for w in wells) + params['inj_years'] + 1)
    out, cap = [], []
    for year in years:
        xs, ys, ai, ages, act = [], [], [], [], []
        for w in wells:
            if w['start_year'] <= year:
                xs.append(w['x']); ys.append(w['y']); ai.append(w['aquifer_idx'])
                if year <= w['end_year']:
                    ages.append(year - w['start_year'] + 1); act.append(True)
                else:
                    ages.append(params['inj_years']); act.append(False)
        xs = np.array(xs); ys = np.array(ys); ai = np.array(ai)
        t_s = np.array(ages) * 86400 * 365
        R = np.sqrt(2.25 * D_a[ai] * t_s)
        csi = np.sqrt(Q_a[ai] * t_s / (np.pi * phi_a[ai] * h_a[ai]))
        psi = np.exp(om_a[ai]) * csi
        ga = ga_a[ai]; pc = pc_a[ai]

        # distance from every well (target) to every source well
        dx = xs[:, None] - xs[None, :]
        dy = ys[:, None] - ys[None, :]
        r = np.sqrt(dx*dx + dy*dy)
        np.fill_diagonal(r, 0.1)          # model sets r == 0 to 0.1 m
        PD = np.zeros_like(r)
        inpl = r <= psi[None, :]
        outpl = (r > psi[None, :]) & (r <= R[None, :])
        P = np.broadcast_to(psi[None, :], r.shape)
        G = np.broadcast_to(ga[None, :], r.shape)
        RR = np.broadcast_to(R[None, :], r.shape)
        PD[inpl] = G[inpl]*np.log(P[inpl]/r[inpl]) + np.log(RR[inpl]/P[inpl])
        PD[outpl] = np.log(RR[outpl]/r[outpl])
        dP = (PD * pc[None, :]).sum(axis=1)
        val = (dP + pref_a[ai]) / pfrac_a[ai] * 100
        out.append(val.max())
        cap.append(sum(act))
    return out, cap, wpy


t = time.time()
max_dP_fast, cap_fast, wpy_fast = fast_max(S_gdf, params)
print(f"fast method: {time.time()-t:.2f} s")

n = min(len(max_dP_ref), len(max_dP_fast))
a = np.array(max_dP_ref[:n]); b = np.array(max_dP_fast[:n])
print("\nyear  published   fast     diff")
for i in range(0, n, 4):
    print(f"{i+1:4d}  {a[i]:9.4f} {b[i]:8.4f} {b[i]-a[i]:8.5f}")
print(f"\nmax abs diff: {np.nanmax(np.abs(a-b)):.6f} %-of-frac")
print(f"capacity identical: {cap_fast[:n] == capacity_ref[:n]}")
