import time
import math
import requests
import pandas as pd
import numpy as np
import streamlit as st
from dataclasses import dataclass
from typing import List, Optional, Tuple, Set, Dict

from ortools.constraint_solver import pywrapcp, routing_enums_pb2


# =========================================================
# UI
# =========================================================
st.set_page_config(
    page_title="Optimisation tournées multi-techniciens (OSM/OSRM)",
    layout="wide",
)

st.title("Optimisation tournées multi-techniciens (OSM/OSRM)")
st.caption(
    "Affectation + ordre de visite. Contraintes : heures max/technicien, critères (skills), "
    "SLA (deadline), priorité. Sans Google : OSM (Nominatim) + OSRM."
)


# =========================================================
# Data models
# =========================================================
@dataclass
class Tech:
    name: str
    address_input: str
    address_resolved: str
    lat: float
    lon: float
    max_minutes: int
    skills: Set[str]


@dataclass
class Job:
    name: str
    address_input: str
    address_resolved: str
    lat: float
    lon: float
    service_min: int
    required_skills: Set[str]
    deadline_min: Optional[int]  # None = pas de SLA
    priority: int                # 1..5 (5 = le + prioritaire)


# =========================================================
# Helpers
# =========================================================
def parse_tags(s: str) -> Set[str]:
    """Parse 'hauteur, elec; caces' -> {'hauteur','elec','caces'}"""
    if s is None:
        return set()
    s = str(s).strip()
    if s == "" or s.lower() == "nan":
        return set()
    parts = s.replace(";", ",").split(",")
    out = {p.strip().lower() for p in parts if p.strip()}
    return out


def to_float_or_none(x) -> Optional[float]:
    """Convertit une cellule (texte/NaN) en float si possible, sinon None."""
    if x is None:
        return None
    if isinstance(x, float) and math.isnan(x):
        return None
    s = str(x).strip()
    if s == "" or s.lower() == "nan":
        return None
    try:
        return float(s)
    except Exception:
        return None


def hours_to_minutes(h: float) -> int:
    if h is None:
        return 0
    if h < 0:
        h = 0
    return int(round(h * 60))


# =========================================================
# Geocoding (OSM Nominatim)
# =========================================================
NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"

@st.cache_data(show_spinner=False, ttl=24 * 3600)
def geocode_osm(address: str) -> Optional[Tuple[float, float, str]]:
    """Retourne (lat, lon, display_name) via Nominatim."""
    if not address or not str(address).strip():
        return None

    headers = {
        "User-Agent": "tournee-optimizer/1.0 (internal)",
        "Accept-Language": "fr",
    }
    params = {"q": str(address).strip(), "format": "json", "limit": 1}

    r = requests.get(NOMINATIM_URL, params=params, headers=headers, timeout=20)
    r.raise_for_status()
    data = r.json()
    if not data:
        return None

    lat = float(data[0]["lat"])
    lon = float(data[0]["lon"])
    disp = data[0].get("display_name", str(address))
    return lat, lon, disp


# =========================================================
# OSRM table (minutes)
# =========================================================
OSRM_TABLE_URL = "https://router.project-osrm.org/table/v1/driving/"

@st.cache_data(show_spinner=False, ttl=6 * 3600)
def osrm_table_minutes(coords: List[Tuple[float, float]]) -> List[List[int]]:
    """
    coords: liste de (lat, lon)
    Retour: matrice [i][j] en minutes
    """
    if len(coords) < 2:
        return [[0]]

    coord_str = ";".join([f"{lon},{lat}" for (lat, lon) in coords])
    url = OSRM_TABLE_URL + coord_str

    r = requests.get(url, params={"annotations": "duration"}, timeout=30)
    r.raise_for_status()
    durations = r.json()["durations"]  # secondes

    # Convert to minutes, robust if None
    return [[int(round((d or 0) / 60.0)) for d in row] for row in durations]


# =========================================================
# OR-Tools VRP solver (multi-tech)
# =========================================================
def solve_vrp_multi_tech(
    tm_min: List[List[int]],
    service_min: List[int],
    techs: List[Tech],
    starts: List[int],
    ends: List[int],
    job_nodes: List[int],
    job_deadlines: Dict[int, int],
    allowed_vehicles: Dict[int, List[int]],
    priority_by_node: Dict[int, int],
    allow_drop: bool,
    time_limit_s: int,
):
    """
    Minimise le temps de trajet total (OSRM), sous contraintes:
    - heures max par technicien (dimension Time: travel + service)
    - skills (allowed vehicles per job)
    - SLA deadlines (time windows)
    - priorité (si allow_drop=True -> disjunction penalty)
    """
    n = len(tm_min)
    m = len(techs)

    manager = pywrapcp.RoutingIndexManager(n, m, starts, ends)
    routing = pywrapcp.RoutingModel(manager)

    # ---- Objective: travel time only (minutes)
    def travel_cb(from_i, to_i):
        f = manager.IndexToNode(from_i)
        t = manager.IndexToNode(to_i)
        return int(tm_min[f][t])

    travel_cb_idx = routing.RegisterTransitCallback(travel_cb)
    routing.SetArcCostEvaluatorOfAllVehicles(travel_cb_idx)

    # ---- Time dimension: travel + service at FROM
    def time_cb(from_i, to_i):
        f = manager.IndexToNode(from_i)
        t = manager.IndexToNode(to_i)
        return int(tm_min[f][t] + service_min[f])

    time_cb_idx = routing.RegisterTransitCallback(time_cb)

    routing.AddDimension(
        time_cb_idx,
        0,          # slack
        10**9,      # large, bounded per vehicle
        True,       # start cumul = 0
        "Time"
    )
    time_dim = routing.GetDimensionOrDie("Time")

    # ---- Max minutes per tech (hard)
    for v, tech in enumerate(techs):
        end_index = routing.End(v)
        time_dim.CumulVar(end_index).SetMax(int(tech.max_minutes))

    # ---- SLA deadlines: arrival time <= deadline
    for node, deadline in job_deadlines.items():
        node = int(node)
        idx = manager.NodeToIndex(node)
        # arrival at job is CumulVar(job)
        time_dim.CumulVar(idx).SetRange(0, int(deadline))

    # ---- Skills eligibility (allowed vehicles for job nodes)
    for node, vehs in allowed_vehicles.items():
        node = int(node)
        idx = manager.NodeToIndex(node)

        # Cast int natif Python (évite numpy.int64 => TypeError)
        vehs_clean = [int(v) for v in vehs]

        # Ne jamais appliquer sur start/end (sécurité)
        if routing.IsStart(idx) or routing.IsEnd(idx):
            continue

        routing.SetAllowedVehiclesForIndex(vehs_clean, idx)

    # ---- Optional drop (report) with penalty based on priority
    if allow_drop:
        for node in job_nodes:
            node = int(node)
            idx = manager.NodeToIndex(node)
            prio = int(priority_by_node.get(node, 3))
            # pénalité forte, plus prio = plus pénalité -> on reporte en dernier les priorités hautes
            penalty = 200000 * prio
            routing.AddDisjunction([idx], penalty)

    # ---- Search params
    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    params.time_limit.FromSeconds(int(time_limit_s))

    sol = routing.SolveWithParameters(params)
    if sol is None:
        return None

    # ---- Extract routes + dropped jobs
    routes = []
    dropped = []

    if allow_drop:
        for node in job_nodes:
            node = int(node)
            idx = manager.NodeToIndex(node)
            # dropped if next is itself
            if sol.Value(routing.NextVar(idx)) == idx:
                dropped.append(node)

    for v, tech in enumerate(techs):
        idx = routing.Start(v)
        route_nodes = []

        while not routing.IsEnd(idx):
            route_nodes.append(manager.IndexToNode(idx))
            idx = sol.Value(routing.NextVar(idx))
        route_nodes.append(manager.IndexToNode(idx))

        end_time = sol.Value(time_dim.CumulVar(routing.End(v)))
        routes.append({
            "vehicle": v,
            "tech": tech.name,
            "nodes": route_nodes,
            "end_time_min": int(end_time),
            "max_min": int(tech.max_minutes)
        })

    return {"routes": routes, "dropped": dropped}


# =========================================================
# UI Inputs
# =========================================================
left, right = st.columns([1.15, 1.85])

with left:
    st.subheader("1) Techniciens")
    st.write("Chaque technicien : adresse départ + heures max + skills (tags séparés par virgules).")

    if "techs_df" not in st.session_state:
        st.session_state.techs_df = pd.DataFrame([
            {"Tech": "Tech A", "Adresse départ": "", "Heures max": 7.5, "Skills (ex: hauteur,elec)": "hauteur, elec"},
            {"Tech": "Tech B", "Adresse départ": "", "Heures max": 8.0, "Skills (ex: hauteur,elec)": "elec"},
        ])

    techs_df = st.data_editor(
        st.session_state.techs_df,
        num_rows="dynamic",
        use_container_width=True,
        hide_index=True,
        column_config={
            "Tech": st.column_config.TextColumn(required=True),
            "Adresse départ": st.column_config.TextColumn(required=True),
            "Heures max": st.column_config.NumberColumn(min_value=0.0, step=0.5, required=True),
            "Skills (ex: hauteur,elec)": st.column_config.TextColumn(required=False),
        }
    )
    st.session_state.techs_df = techs_df

    st.subheader("2) Interventions")
    st.write("Chaque intervention : adresse + durée + skills requis + SLA (deadline en heures, optionnel) + priorité.")

    if "jobs_df" not in st.session_state:
        st.session_state.jobs_df = pd.DataFrame([
            {"Intervention": "Job 1", "Adresse": "", "Durée (min)": 30, "Skills requis": "elec", "SLA (h) (optionnel)": 4.0, "Priorité (1-5)": 5},
            {"Intervention": "Job 2", "Adresse": "", "Durée (min)": 45, "Skills requis": "hauteur", "SLA (h) (optionnel)": "", "Priorité (1-5)": 3},
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
            "Skills requis": st.column_config.TextColumn(required=False),
            "SLA (h) (optionnel)": st.column_config.TextColumn(required=False),
            "Priorité (1-5)": st.column_config.NumberColumn(min_value=1, max_value=5, step=1, required=True),
        }
    )
    st.session_state.jobs_df = jobs_df

    c1, c2 = st.columns(2)
    with c1:
        allow_drop = st.checkbox("Autoriser le report (drop) si impossible", value=False)
    with c2:
        time_limit = st.slider("Temps de calcul max (s)", 5, 60, 15)

    run = st.button("🚀 Optimiser affectation + tournées", type="primary")


# =========================================================
# Run + Results
# =========================================================
with right:
    st.subheader("Résultats")

    if not run:
        st.info("Renseigne des techniciens + interventions, puis clique sur **Optimiser**.")
        st.stop()

    # Clean dataframes
    techs_df = st.session_state.techs_df.copy().dropna(subset=["Tech", "Adresse départ", "Heures max"])
    jobs_df = st.session_state.jobs_df.copy().dropna(subset=["Intervention", "Adresse", "Durée (min)", "Priorité (1-5)"])

    if len(techs_df) == 0:
        st.error("Ajoute au moins 1 technicien.")
        st.stop()
    if len(jobs_df) == 0:
        st.error("Ajoute au moins 1 intervention.")
        st.stop()

    # ---- Geocode techs
    techs: List[Tech] = []
    bad_tech = []

    with st.spinner("Géocodage techniciens (OSM)…"):
        for _, row in techs_df.iterrows():
            name = str(row["Tech"]).strip()
            addr = str(row["Adresse départ"]).strip()
            max_h = float(row["Heures max"])
            skills = parse_tags(row.get("Skills (ex: hauteur,elec)", ""))

            g = geocode_osm(addr)
            if g is None:
                bad_tech.append((name, addr))
                continue
            lat, lon, disp = g
            techs.append(
                Tech(
                    name=name,
                    address_input=addr,
                    address_resolved=disp,
                    lat=lat,
                    lon=lon,
                    max_minutes=hours_to_minutes(max_h),
                    skills=skills
                )
            )
            time.sleep(0.1)

    if bad_tech:
        st.error("Techniciens non géocodés. Ajoute 'ville + code postal + France'.")
        st.dataframe(pd.DataFrame(bad_tech, columns=["Tech", "Adresse"]), use_container_width=True, hide_index=True)
        st.stop()

    # ---- Geocode jobs
    jobs: List[Job] = []
    bad_jobs = []

    with st.spinner("Géocodage interventions (OSM)…"):
        for _, row in jobs_df.iterrows():
            name = str(row["Intervention"]).strip()
            addr = str(row["Adresse"]).strip()
            dur = int(row["Durée (min)"])
            req = parse_tags(row.get("Skills requis", ""))
            prio = int(row["Priorité (1-5)"])

            sla_h = to_float_or_none(row.get("SLA (h) (optionnel)", None))
            deadline = hours_to_minutes(sla_h) if sla_h is not None else None

            g = geocode_osm(addr)
            if g is None:
                bad_jobs.append((name, addr))
                continue
            lat, lon, disp = g

            jobs.append(
                Job(
                    name=name,
                    address_input=addr,
                    address_resolved=disp,
                    lat=lat,
                    lon=lon,
                    service_min=dur,
                    required_skills=req,
                    deadline_min=deadline,
                    priority=prio
                )
            )
            time.sleep(0.1)

    if bad_jobs:
        st.error("Certaines interventions n’ont pas été géocodées. Ajoute 'ville + code postal + France'.")
        st.dataframe(pd.DataFrame(bad_jobs, columns=["Intervention", "Adresse"]), use_container_width=True, hide_index=True)
        st.stop()

    # ---- Build nodes: start per tech + end per tech (duplicate) + jobs
    coords: List[Tuple[float, float]] = []
    service_min: List[int] = []
    starts: List[int] = []
    ends: List[int] = []

    # Start nodes
    for t in techs:
        starts.append(len(coords))
        coords.append((t.lat, t.lon))
        service_min.append(0)

    # End nodes (duplicate starts)
    for t in techs:
        ends.append(len(coords))
        coords.append((t.lat, t.lon))
        service_min.append(0)

    job_offset = len(coords)
    job_nodes: List[int] = []

    for j in jobs:
        job_nodes.append(len(coords))
        coords.append((j.lat, j.lon))
        service_min.append(int(j.service_min))

    # ---- Time matrix
    with st.spinner("Calcul des temps de trajet (OSRM)…"):
        tm = osrm_table_minutes(coords)

    # ---- Skills eligibility
    allowed_vehicles: Dict[int, List[int]] = {}
    for node, job in zip(job_nodes, jobs):
        ok = []
        for v, tech in enumerate(techs):
            if job.required_skills.issubset(tech.skills):
                ok.append(v)
        if not ok:
            st.error(
                f"Aucun technicien ne possède les skills requis pour '{job.name}'. "
                f"Requis={sorted(job.required_skills)}. "
                f"➡️ ajoute le skill à un tech ou corrige la saisie."
            )
            st.stop()
        allowed_vehicles[int(node)] = [int(v) for v in ok]

    # ---- Deadlines & priority map
    job_deadlines: Dict[int, int] = {}
    priority_by_node: Dict[int, int] = {}
    for node, job in zip(job_nodes, jobs):
        node = int(node)
        priority_by_node[node] = int(job.priority)
        if job.deadline_min is not None:
            job_deadlines[node] = int(job.deadline_min)

    # ---- Solve
    with st.spinner("Optimisation (OR-Tools)…"):
        sol = solve_vrp_multi_tech(
            tm_min=tm,
            service_min=service_min,
            techs=techs,
            starts=starts,
            ends=ends,
            job_nodes=job_nodes,
            job_deadlines=job_deadlines,
            allowed_vehicles=allowed_vehicles,
            priority_by_node=priority_by_node,
            allow_drop=allow_drop,
            time_limit_s=int(time_limit),
        )

    if sol is None:
        st.error(
            "Aucune solution faisable avec ces contraintes (heures max / SLA / skills). "
            "➡️ augmente les heures max, ajoute un technicien, ou assouplis les SLA."
        )
        st.stop()

    dropped = sol["dropped"]
    if dropped:
        dropped_names = [jobs[int(n) - job_offset].name for n in dropped]
        st.warning("Interventions reportées : " + ", ".join(dropped_names))

    # ---- Display
    st.success("Solution trouvée ✅")
    st.markdown("### Tournées par technicien")

    def node_label(node: int) -> str:
        node = int(node)
        if node < len(techs):
            return f"Départ {techs[node].name}"
        if len(techs) <= node < 2 * len(techs):
            return f"Retour {techs[node - len(techs)].name}"
        return jobs[node - job_offset].name

    assignment_rows = []

    for r in sol["routes"]:
        tech_name = r["tech"]
        nodes_route = r["nodes"]
        end_time = r["end_time_min"]
        max_min = r["max_min"]

        jobs_in_route: List[Job] = []
        for n in nodes_route:
            if int(n) >= job_offset:
                jb = jobs[int(n) - job_offset]
                jobs_in_route.append(jb)
                assignment_rows.append({
                    "Intervention": jb.name,
                    "Technicien": tech_name,
                    "Priorité": jb.priority,
                    "SLA (h)": ("" if jb.deadline_min is None else round(jb.deadline_min / 60.0, 2)),
                    "Skills requis": ", ".join(sorted(jb.required_skills)),
                })

        with st.expander(
            f"{tech_name} — {len(jobs_in_route)} interventions — fin à {end_time} min (max {max_min} min)",
            expanded=True
        ):
            if not jobs_in_route:
                st.write("Aucune intervention affectée.")
            else:
                df = pd.DataFrame({
                    "Ordre": list(range(1, len(jobs_in_route) + 1)),
                    "Intervention": [j.name for j in jobs_in_route],
                    "Adresse (OSM)": [j.address_resolved for j in jobs_in_route],
                    "Durée sur site (min)": [j.service_min for j in jobs_in_route],
                    "Skills requis": [", ".join(sorted(j.required_skills)) for j in jobs_in_route],
                    "Priorité": [j.priority for j in jobs_in_route],
                    "SLA (h)": [("" if j.deadline_min is None else round(j.deadline_min / 60.0, 2)) for j in jobs_in_route],
                })
                st.dataframe(df, use_container_width=True, hide_index=True)

            st.caption("Chemin: " + " → ".join(node_label(n) for n in nodes_route))

    st.markdown("### Synthèse d’affectation")
    if assignment_rows:
        st.dataframe(
            pd.DataFrame(assignment_rows).sort_values(["Technicien", "Priorité"], ascending=[True, False]),
            use_container_width=True,
            hide_index=True
        )
    else:
        st.write("Aucune intervention affectée.")


# =========================================================
# requirements.txt reminder (unchanged)
# =========================================================
# streamlit
# pandas
# requests
# ortools
