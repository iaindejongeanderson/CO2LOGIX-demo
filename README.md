# CO2LOGIX UK Explorer

An interactive, browser-based UK case study for **CO2LOGIX** — a first-order model of
pressure-constrained CO₂ geological storage growth at the basin scale.

> Subsurface pressure increases from CO₂ injection can constrain injectivity and safe operating
> margins, reducing the dynamic capacity of CO₂ geological storage systems. Engineered pressure
> management approaches are effective but carry additional cost and risk. [...] We present
> CO2LOGIX, a model which evaluates the scale of subsurface pressure buildup under different
> growth trajectories with high computational efficiency. We demonstrate the value of CO2LOGIX
> through a UK case study.
>
> — de Jonge-Anderson et al. (2026)

Change the growth parameters, injection rate and injection duration, then watch pressure build
across the seven UK saline aquifer storage units and read off how much CO₂ has been stored by the
time peak reservoir pressure reaches 90% of fracture pressure.

**Live:** <https://iaindejongeanderson.github.io/CO2LOGIX-demo/>

---

## What it does

| Panel | Shows |
|---|---|
| **Map** | Pressure as a 2.5 km raster over the seven storage units, with the UK coastline and every well coloured by its own node pressure. Scrub or play through the run. |
| **Peak reservoir pressure** | The highest pressure anywhere in the storage units, as % of that unit's fracture pressure, against the 90% limit. |
| **Injection rate** | Active wells × rate, in Mt CO₂/yr. |
| **Cumulative stored** | Gt CO₂ injected to date. |
| **Headline** | Cumulative CO₂ stored at the year peak pressure first reaches 90% of fracture pressure. |
| **Storage units** | Wells, CO₂ stored and peak pressure per unit. |

Controls: growth rate `k`, carrying capacity `L` (50 to 14,000 wells, on a log slider — the top
of the range is roughly the number of wells drilled on the UKCS to date), rate per well, injection
years per well, drilling period, and the placement draw. The three published growth scenarios
(`k` = 0.086 / 0.13 / 0.22) are one click away.

On historic industry growth rates (`k` = 0.086), pressure reaches the 90% limit after **81 years**
with **10.9 Gt** stored — close to the paper's "83 years, ~12 GtCO₂ stored by 2100".

---

## Model

This is the published **CO2LOGIX v1.0** formulation, unchanged:

- **Growth** — wells start on a logistic schedule, `L / (1 + e^(−k(t−t₀)))`, with `t₀` solved so
  two wells exist in year 1 (`model/growth.py`).
- **Pressure** — the Nordbotten two-region analytical solution for CO₂ injection into a
  brine-filled aquifer: a mobile plume of radius `ψ` inside a pressure front at
  `R = √(2.25·D·t)`. Single-well solutions superpose linearly (`model/pressure.py`).
- **Fracture pressure** — Zhang & Yin (2017), `p_frac = p_p + 0.75(σ_v − p_p)`, on 10 MPa/km
  hydrostatic and 23 MPa/km lithostatic gradients (`model/utils.py`).
- **Fluid properties** — CO₂ density/viscosity and brine viscosity from CoolProp at each unit's
  reference pressure and temperature; rock compressibility from the Hall (1953) correlation.

Only the **open** boundary case is included. The closed-boundary branch in `pressure.py` is not
exposed here.

### How the browser version stays faithful

The published model sweeps a 1 km raster each year and takes `np.nanmax`. Because the Nordbotten
solution is radial and decreases monotonically away from each well, the maximum of the superposed
field always lands on a well node — so the explorer evaluates the superposition **well-to-well**
instead. This was checked against `model/pressure.py` over an 85-year run: the two max-pressure
series agree to **0.000000 % of fracture pressure**, and the capacity series is identical.
Run `tools/validate_against_model.py` from the release folder to reproduce that check.

Distances between wells never change across a run — only each well's `R`, `ψ` and `p_c` do. So the
neighbour list is built once and reused every year, with distances held in **log space**: the
Nordbotten solution is a difference of logs, so the per-year inner loop needs no `sqrt` and no
`log` at all. Pairs are stored once (`i < j`) and sorted by `j`, which makes the wells drilled by
any given year a plain prefix of the list. At 14,000 wells this is 15–20× faster than recomputing
each year (a 200-year run drops from ~82 s to ~4 s) and returns bit-identical results.

### Where the peak actually is

The reported peak sits at the wellbore node, where the logarithmic solution is singular — the
published model pins that node distance at 0.1 m, and so does this.

The map's 2.5 km raster is a separate display grid and reads considerably cooler. In the
reference scenario at the limit year, the raster tops out at **81.7%** while the hottest well node
is at **90.1%**; across the whole run no raster cell ever exceeds 100%, while three well nodes do.
So the ≥90% classes are shown **on well markers**, which get a halo once they reach the limit, and
the legend reports how many wells are at or past it. Treat the raster as the regional field and
the well markers as the limit check.

### Well placement — the one deliberate departure

Candidate locations are drawn from a 1 km lattice inside the seven unit outlines, as 12,000
unweighted samples without replacement. Because the lattice is uniform, that places wells in
proportion to storage unit area — the intent stated in `example.py`: *"proportion of total wells
to place in each aquifer outline (based on area)"*.

`pressure.py` instead weights each lattice point by its unit's `well_fraction`. Since a unit
already contributes lattice points in proportion to its area, that applies area twice and places
wells in proportion to area **squared**:

| | Bunter | Mey | St Bees | Dornoch | Otter | Heimdal | Captain |
|---|---|---|---|---|---|---|---|
| unit area | 45.6% | 24.8% | 9.5% | 7.5% | 4.9% | 4.7% | 2.7% |
| **used here** (∝ area) | 46.3% | 24.5% | 9.3% | 7.6% | 5.0% | 4.6% | **2.7%** |
| `pressure.py` (∝ area²) | 71.5% | 21.5% | 3.1% | 2.0% | 0.8% | 0.8% | **0.3%** |

The squared weighting starves the smaller units: the Captain Sandstone — the most permeable at
7000 mD — would receive 0.3% of wells, and the thin Dornoch and Heimdal units almost none.

To reproduce the published behaviour instead, set `WELL_WEIGHTING = "published"` in
`tools/prep_data.py` and regenerate `data.js`. On the reference scenario that shifts the result
from 10.85 Gt at year 81 (peak 94.6%) to 11.29 Gt at year 82 (peak 100.6%).

Draw 1 uses the pool exactly as shipped (`random_state=42`); higher draws reshuffle it.

---

## Data

| File | Source |
|---|---|
| Storage unit outlines | `Github/input_data/saline_aquifers.shp` (EPSG:32631) |
| Unit properties | `Github/input_data/input_table.csv` |
| UK coastline | `CO2LOGIX v1.0/Shapefiles/uk_project.shp` |

All of it is baked into `data.js` (121 KB) by `tools/prep_data.py`, which also runs the CoolProp
lookups and the well sampling. The browser then needs only arithmetic — no server, no build step,
no dependencies.

To regenerate after changing the inputs:

```bash
python tools/prep_data.py .
```

Requires `geopandas`, `pandas`, `numpy`, `shapely` and `CoolProp`. Set `CO2LOGIX_ROOT` if the
repo does not sit alongside the `Github` and `CO2LOGIX v1.0` folders.

---

## Hosting on GitHub Pages

`index.html`, `app.js` and `data.js` are the whole site — no build step. In the repository,
go to **Settings → Pages → Deploy from a branch** and select branch `main`, folder `/ (root)`.
`.nojekyll` is already present so nothing is filtered.

The page also runs straight from disk — open `index.html` in a browser. Data is loaded via a
`<script>` tag rather than `fetch`, so there is no `file://` CORS problem.

---

## Citation

de Jonge-Anderson, I., Johnson, G., Alcalde, J. & Roberts, J. J. (2026). CO2LOGIX: A first-order
model of pressure-constrained CO₂ geological storage growth at the basin scale. *International
Journal of Greenhouse Gas Control* **151**, 104608.
<https://doi.org/10.1016/j.ijggc.2026.104608>

de Jonge-Anderson, I., Johnson, G., Alcalde, J. & Roberts, J. J. (2025).
iaindejongeanderson/CO2LOGIX: UK case study release (v1.0). Zenodo.
<https://doi.org/10.5281/zenodo.17484442> · CC BY 4.0

Department of Civil & Environmental Engineering, University of Strathclyde.
