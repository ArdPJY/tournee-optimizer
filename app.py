import time
import requests
import pandas as pd
import streamlit as st
from dataclasses import dataclass
from typing import List, Optional, Tuple

from ortools.constraint_solver import pywrapcp, routing_enums_pb2


# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="Optimisation tournées multi-techniciens (OSM)", layout="wide")
st.title("Optimisation tournées multi-techniciens (OSM/OSRM) — affectation + ordre + budgets")

st.caption(
    "Sans Google : géocodage via OpenStreetMap (Nominatim) + temps de trajet OSRM. "
    "Le solveur assigne les interventions aux techniciens et calcule l’ordre optimal, "
    "avec des plafonds par technicien (en € ou h ou minutes)."
)


# -----------------------------
# Data models
# -----------------------------
@dataclass
class Tech:
    name: str
    address_input: str
    address_resolved: str
    lat: float
    lon: float
    rate_cents_per_min: int
    max_work_min: int  # plafond en minutes (trajet+service)


@dataclass
class Job:
    name: str
    address_input: str
    address_resolved: str
    lat: float
    lon: float
    service_min: int


# -----------------------------
# Geocoding (OSM Nominatim)
# -----------------------------
NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"

@st.cache_data(show_spinner=False, ttl=24 * 3600)
def geocode_osm(address: str) -> Optional[Tuple[float, float, str]]:
    if not address or not address.strip():
        return None
    headers = {
        "User-Agent": "tournee-optimizer/1.0 (internal)",
        "Accept-Language": "fr",
    }
    params = {"q": address, "format": "json", "limit": 1}
    r = requests.get(NOMINATIM_URL, params=params, headers=headers, timeout=20)
    r.raise_for_status()
    data = r.json()
    if not data:
        return None
    lat = float(data[0]["lat"])
    lon = float(data[0]["lon"])
    disp = data[0].get("display_name", address)
    return lat, lon, disp


# -----------------------------
# OSRM time matrix (minutes)
# -----------------------------
OSRM_TABLE_URL = "https://router.project-osrm.org/table/v1/driving/"

@st.cache_data(show_spinner=False, ttl=6 * 3600)
def osrm_table_minutes(coords: List[Tuple[float, float]]) -> List[List[int]]:
    if len(coords) < 2:
        return [[0]]
    coord_str = ";".join([f"{lon},{lat}" for (lat, lon) in coords])
    url = OSRM_TABLE_URL + coord_str
    r = requests.get(url, params={"annotations": "duration"}, timeout=30)
    r.raise_for_status()
    durations = r.json()["durations"]  # seconds
    return [[int(round((d or 0) / 60.0)) for d in row] for row in durations]


# -----------------------------
# Unit conversion helpers
# -----------------------------
RATE_UNITS = ["€/h", "€/jour (8h)"]
CAP_UNITS = ["€", "h", "min"]

def rate_to_cents_per_min(rate_value: float, unit: str) -> int:
    """
    Convertit un taux en centimes/min.
    - €/h : rate_value € / heure
    - €/jour (8h) : rate_value € / jour -> 8h
    """
    if rate_value < 0:
        rate_value = 0
    if unit == "€/h":
        euros_per_min = rate_value / 60.0
    elif unit == "€/jour (8h)":
        euros_per_min = rate_value / (8.0 * 60.0)
    else:
        euros_per_min = rate_value / 60.0
    return int(round(euros_per_min * 100))

def cap_to_max_minutes(cap_value: float, cap_unit: str, rate_cents_per_min: int) -> int:
    """
    Convertit le plafond en minutes max (trajet+service).
    - € : convertit en minutes via le taux
    - h : convertit en minutes
    - min : direct
    """
    if cap_value < 0:
        cap_value = 0

    if cap_unit == "min":
        return int(round(cap_value))
    if cap_unit == "h":
        return int(round(cap_value * 60))

    # cap en €
    # minutes = cap_euros / euros_per_min
    euros_per_min = max(rate_cents_per_min, 1) / 100.0
    return int(round(cap_value / euros_per_min))


# -----------------------------
# VRP solver (multi-tech)
# -----------------------------
def solve_multi_tech_vrp(
    time_matrix_min: List[List[int]],
    service_min: List[int],
    techs: List[Tech],
    starts: List[int],
    ends: List[int],
    time_limit_s: int = 10,
):
    """
    Optimise :
    - affectation jobs -> tech
    - ordre de visite
    - objectif : minimiser coût total (temps * taux tech)
    - contrainte : minutes totales (trajet+service) <= max_work_min par tech
    """
    n = len(time_matrix_min)
    num_vehicles = len(techs)

    manager = pywrapcp.RoutingIndexManager(n, num_vehicles, starts, ends)
    routing = pywrapcp.RoutingModel(manager)

    # 1) Coût (objectif) dépend du véhicule : minutes * rate_cents_per_min
    cost_callbacks = []
    for v in range(num_vehicles):
        rate = techs[v].rate_cents_per_min

        def make_cost_cb(rate_cents):
            def cb(from_i, to_i):
                f = manager.IndexToNode(from_i)
                t = manager.IndexToNode(to_i)
                minutes = time_matrix_min[f][t] + service_min[t]
                return int(minutes * rate_cents)
            return cb

        cb_index = routing.RegisterTransitCallback(make_cost_cb(rate))
        routing.SetArcCostEvaluatorOfVehicle(cb_index, v)
        cost_callbacks.append(cb_index)

    # 2) Dimension "Work" en minutes (trajet+service), contrainte par tech
    #    Ici on met un transit callback en minutes (pas en €)
    def work_minutes_cb(from_i, to_i):
        f = manager.IndexToNode(from_i)
        t = manager.IndexToNode(to_i)
        return int(time_matrix_min[f][t] + service_min[t])

    work_cb_idx = routing.RegisterTransitCallback(work_minutes_cb)

    routing.AddDimension(
        work_cb_idx,
        0,               # slack
        10**9,           # capacity temporaire (on fixe par vehicle après)
        True,            # start cumul to zero
        "Work"
    )
    work_dim = routing.GetDimensionOrDie("Work")

    for v, tech in enumerate(techs):
        end_idx = routing.End(v)
        # borne max au nœud de fin
        work_dim.CumulVar(end_idx).SetMax(tech.max_work_min)

    # Paramètres de recherche
    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    params.time_limit.FromSeconds(time_limit_s)

    sol = routing.SolveWithParameters(params)
    if sol is None:
        return None

    # Extraction des routes
    routes = []
    total_cost_cents = 0

    for v in range(num_vehicles):
        idx = routing.Start(v)
        route_nodes = []
        route_cost_cents = 0
        route_work_min = 0

        while not routing.IsEnd(idx):
            node = manager.IndexToNode(idx)
            route_nodes.append(node)
            nxt = sol.Value(routing.NextVar(idx))
            nxt_node = manager.IndexToNode(nxt)

            if not routing.IsEnd(nxt):
                minutes = time_matrix_min[node][nxt_node] + service_min[nxt_node]
                route_work_min += minutes
                route_cost_cents += minutes * techs[v].rate_cents_per_min

            idx = nxt

        route_nodes.append(manager.IndexToNode(idx))
        total_cost_cents += route_cost_cents

        routes.append({
            "vehicle": v,
            "tech_name": techs[v].name,
            "nodes": route_nodes,
            "work_min": route_work_min,
            "cost_cents": int(route_cost_cents),
            "max_work_min": techs[v].max_work_min
        })

    return {
        "routes": routes,
        "total_cost_cents": int(total_cost_cents)
    }


# -----------------------------
# UI inputs
# -----------------------------
left, right = st.columns([1.05, 1.95])

with left:
    st.subheader("1) Techniciens")
    st.write("Ajoute autant de techniciens que nécessaire (adresse de départ + taux + plafond).")

    if "techs_df" not in st.session_state:
        st.session_state.techs_df = pd.DataFrame([
            {"Tech": "Tech 1", "Adresse départ": "", "Taux": 60, "Unité taux": "€/h", "Plafond": 8, "Unité plafond": "h"},
            {"Tech": "Tech 2", "Adresse départ": "", "Taux": 55, "Unité taux": "€/h", "Plafond": 350, "Unité plafond": "€"},
        ])

    techs_df = st.data_editor(
        st.session_state.techs_df,
        num_rows="dynamic",
        use_container_width=True,
        hide_index=True,
        column_config={
            "Tech": st.column_config.TextColumn(required=True),
            "Adresse départ": st.column_config.TextColumn(required=True),
            "Taux": st.column_config.NumberColumn(min_value=0.0, step=1.0, required=True),
            "Unité taux": st.column_config.SelectboxColumn(options=RATE_UNITS, required=True),
            "Plafond": st.column_config.NumberColumn(min_value=0.0, step=1.0, required=True),
            "Unité plafond": st.column_config.SelectboxColumn(options=CAP_UNITS, required=True),
        }
    )
    st.session_state.techs_df = techs_df

    st.subheader("2) Interventions")
    st.write("Ordre des lignes = ordre initial (référence), mais ici l’objectif est surtout l’affectation + tournée.")

    if "jobs_df" not in st.session_state:
        st.session_state.jobs_df = pd.DataFrame([
            {"Intervention": "Job 1", "Adresse": "", "Durée (min)": 30},
            {"Intervention": "Job 2", "Adresse": "", "Durée (min)": 45},
        ])

    jobs_df = st.data_editor(
        st.session_state.jobs_df,
        num_rows="dynamic",
        use_container_width=True,
        hide_index=True,
        column_config={
            "Intervention": st.column_config.TextColumn(required=True),
            "Adresse": st.column_config.TextColumn(required=True),
            "Durée (min)": st.column_config.NumberColumn(min_value=0, step=5, required=True),
        }
    )
    st.session_state.jobs_df = jobs_df

    c1, c2 = st.columns(2)
    with c1:
        return_to_start = st.checkbox("Retour au point de départ pour chaque tech", value=True)
    with c2:
        time_limit = st.slider("Temps de calcul max (s)", 3, 30, 10)

    run = st.button("🚀 Optimiser affectation + tournées", type="primary")


with right:
    st.subheader("Résultats")

    if not run:
        st.info("Renseigne des techniciens + des interventions, puis clique sur **Optimiser**.")
        st.stop()

    # Clean input data
    techs_df = st.session_state.techs_df.copy().dropna(subset=["Tech", "Adresse départ", "Taux", "Unité taux", "Plafond", "Unité plafond"])
    jobs_df = st.session_state.jobs_df.copy().dropna(subset=["Intervention", "Adresse", "Durée (min)"])

    if len(techs_df) == 0:
        st.error("Ajoute au moins 1 technicien.")
        st.stop()
    if len(jobs_df) == 0:
        st.error("Ajoute au moins 1 intervention.")
        st.stop()

    # Geocode techs
    techs: List[Tech] = []
    bad_tech = []
    with st.spinner("Géocodage techniciens (OSM) ..."):
        for _, row in techs_df.iterrows():
            name = str(row["Tech"]).strip()
            addr = str(row["Adresse départ"]).strip()
            rate_val = float(row["Taux"])
            rate_unit = str(row["Unité taux"])
            cap_val = float(row["Plafond"])
            cap_unit = str(row["Unité plafond"])

            g = geocode_osm(addr)
            if g is None:
                bad_tech.append((name, addr))
                continue
            lat, lon, disp = g

            rate_cents_min = rate_to_cents_per_min(rate_val, rate_unit)
            max_work_min = cap_to_max_minutes(cap_val, cap_unit, rate_cents_min)

            techs.append(Tech(
                name=name,
                address_input=addr,
                address_resolved=disp,
                lat=lat,
                lon=lon,
                rate_cents_per_min=rate_cents_min,
                max_work_min=max_work_min
            ))
            time.sleep(0.1)

    if bad_tech:
        st.error("Techniciens non géocodés (adresse invalide). Ajoute ville + CP + France.")
        st.dataframe(pd.DataFrame(bad_tech, columns=["Tech", "Adresse départ"]), use_container_width=True)
        st.stop()

    # Geocode jobs
    jobs: List[Job] = []
    bad_jobs = []
    with st.spinner("Géocodage interventions (OSM) ..."):
        for _, row in jobs_df.iterrows():
            name = str(row["Intervention"]).strip()
            addr = str(row["Adresse"]).strip()
            dur = int(row["Durée (min)"])

            g = geocode_osm(addr)
            if g is None:
                bad_jobs.append((name, addr))
                continue
            lat, lon, disp = g
            jobs.append(Job(
                name=name,
                address_input=addr,
                address_resolved=disp,
                lat=lat,
                lon=lon,
                service_min=dur
            ))
            time.sleep(0.1)

    if bad_jobs:
        st.error("Certaines interventions n’ont pas été géocodées. Ajoute ville + CP + France.")
        st.dataframe(pd.DataFrame(bad_jobs, columns=["Intervention", "Adresse"]), use_container_width=True)
        st.stop()

    # Build nodes:
    # For VRP multi-start, simplest: create one start node per tech.
    # End node per tech = same as start if retour demandé (we duplicate end nodes to avoid ambiguity).
    # Then job nodes.
    start_nodes = []
    end_nodes = []
    nodes_latlon = []
    service_min = []

    # Start nodes
    for t in techs:
        start_nodes.append(len(nodes_latlon))
        nodes_latlon.append((t.lat, t.lon))
        service_min.append(0)

    # End nodes (duplicate start coords if return)
    if return_to_start:
        for t in techs:
            end_nodes.append(len(nodes_latlon))
            nodes_latlon.append((t.lat, t.lon))
            service_min.append(0)
    else:
        # not implemented cleanly here; we force return for correctness
        st.warning("Mode sans retour non activé dans cette version. On force le retour.")
        for t in techs:
            end_nodes.append(len(nodes_latlon))
            nodes_latlon.append((t.lat, t.lon))
            service_min.append(0)

    # Job nodes
    job_start_index = len(nodes_latlon)
    for j in jobs:
        nodes_latlon.append((j.lat, j.lon))
        service_min.append(j.service_min)

    # time matrix
    with st.spinner("Calcul des temps de trajet (OSRM) ..."):
        tm = osrm_table_minutes(nodes_latlon)

    # Starts/Ends indexes per vehicle
    starts = start_nodes
    ends = end_nodes

    # Solve
    with st.spinner("Optimisation VRP (OR-Tools) ..."):
        sol = solve_multi_tech_vrp(
            time_matrix_min=tm,
            service_min=service_min,
            techs=techs,
            starts=starts,
            ends=ends,
            time_limit_s=int(time_limit),
        )

    if sol is None:
        st.error(
            "Aucune solution faisable avec ces plafonds. "
            "➡️ Augmente un plafond, ajoute un technicien, ou réduis la charge."
        )
        st.stop()

    total_cost_eur = sol["total_cost_cents"] / 100.0
    st.success(f"Solution trouvée ✅ — Coût total estimé (trajet+service × taux): **{total_cost_eur:.2f} €**")

    # Helper label for nodes
    def node_label(i: int) -> str:
        # start nodes
        if i < len(techs):
            return f"Départ {techs[i].name}"
        # end nodes
        if len(techs) <= i < len(techs) + len(techs):
            return f"Retour {techs[i - len(techs)].name}"
        # jobs
        j_idx = i - (len(techs) + len(techs))
        if 0 <= j_idx < len(jobs):
            return f"{jobs[j_idx].name}"
        return f"Node {i}"

    # Display routes per tech
    st.markdown("### Tournées par technicien")

    for r in sol["routes"]:
        tech = techs[r["vehicle"]]
        route_nodes = r["nodes"]
        # Extract only job nodes in route (for readability)
        jobs_in_route = []
        for n in route_nodes:
            if n >= job_start_index:
                j = jobs[n - job_start_index]
                jobs_in_route.append(j)

        work_min = r["work_min"]
        cost_eur = r["cost_cents"] / 100.0
        max_min = r["max_work_min"]

        with st.expander(f"{tech.name} — {len(jobs_in_route)} interventions — {work_min} min / plafond {max_min} min — {cost_eur:.2f} €", expanded=True):
            if not jobs_in_route:
                st.write("Aucune intervention affectée.")
            else:
                df = pd.DataFrame({
                    "Ordre": list(range(1, len(jobs_in_route) + 1)),
                    "Intervention": [j.name for j in jobs_in_route],
                    "Adresse (OSM)": [j.address_resolved for j in jobs_in_route],
                    "Durée sur site (min)": [j.service_min for j in jobs_in_route],
                })
                st.dataframe(df, use_container_width=True, hide_index=True)

            # Show raw route labels
            st.caption("Chemin (nœuds) : " + " → ".join(node_label(n) for n in route_nodes))

    # Summary assignment table
    st.markdown("### Synthèse d’affectation (intervention → technicien)")
    assign_rows = []
    for r in sol["routes"]:
        tech = techs[r["vehicle"]].name
        for n in r["nodes"]:
            if n >= job_start_index:
                j = jobs[n - job_start_index]
                assign_rows.append({"Intervention": j.name, "Technicien": tech})
    st.dataframe(pd.DataFrame(assign_rows).sort_values(["Technicien", "Intervention"]), use_container_width=True, hide_index=True)
