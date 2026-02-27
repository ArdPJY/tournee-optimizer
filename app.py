import time
import requests
import pandas as pd
import streamlit as st
from dataclasses import dataclass
from typing import List, Optional, Tuple, Set, Dict

from ortools.constraint_solver import pywrapcp, routing_enums_pb2

# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="Tournées multi-tech (OSM) - Heures, Skills, SLA", layout="wide")
st.title("Optimisation tournées multi-techniciens (OSM/OSRM)")
st.caption(
    "Affectation + ordre de visite. Contraintes: heures max/technicien, critères (skills), SLA (deadline), priorité. "
    "Sans Google: OSM (Nominatim) + OSRM."
)

# -----------------------------
# Models
# -----------------------------
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
    priority: int                # 1..5 (5 = très prioritaire)

# -----------------------------
# Helpers
# -----------------------------
def parse_tags(s: str) -> Set[str]:
    if not s:
        return set()
    # sépare par virgule/point-virgule
    raw = s.replace(";", ",").split(",")
    return {t.strip().lower() for t in raw if t.strip()}

def hours_to_minutes(h: float) -> int:
    if h < 0:
        h = 0
    return int(round(h * 60))

# -----------------------------
# Geocoding OSM (Nominatim)
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
# OSRM table (minutes)
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
# Solver VRP multi-tech
# -----------------------------
def solve_vrp_multi_tech(
    tm: List[List[int]],
    service_min: List[int],
    techs: List[Tech],
    starts: List[int],
    ends: List[int],
    job_nodes: List[int],
    job_deadlines: Dict[int, int],              # node -> deadline_min
    allowed_vehicles: Dict[int, List[int]],     # node -> list of tech indexes
    allow_drop: bool,
    priority_by_node: Dict[int, int],
    time_limit_s: int = 15,
):
    n = len(tm)
    m = len(techs)
    manager = pywrapcp.RoutingIndexManager(n, m, starts, ends)
    routing = pywrapcp.RoutingModel(manager)

    # ---- Cost: minimise total travel time (simple, no €)
    # (le solveur minimise les minutes totales, ce qui est cohérent si tu veux optimiser la prod)
    def cost_cb(from_i, to_i):
        f = manager.IndexToNode(from_i)
        t = manager.IndexToNode(to_i)
        return tm[f][t]
    cost_idx = routing.RegisterTransitCallback(cost_cb)
    routing.SetArcCostEvaluatorOfAllVehicles(cost_idx)

    # ---- Time dimension: cumul = arrival time
    # transit = travel + service_at_from
    def time_cb(from_i, to_i):
        f = manager.IndexToNode(from_i)
        t = manager.IndexToNode(to_i)
        return tm[f][t] + service_min[f]
    time_idx = routing.RegisterTransitCallback(time_cb)

    routing.AddDimension(
        time_idx,
        0,          # slack
        10**9,      # capacity temporary, bounded per vehicle below
        True,       # start cumul = 0
        "Time"
    )
    time_dim = routing.GetDimensionOrDie("Time")

    # ---- Max hours per tech (contrainte dure)
    for v, tech in enumerate(techs):
        end_idx = routing.End(v)
        time_dim.CumulVar(end_idx).SetMax(tech.max_minutes)

    # ---- SLA deadlines (time windows): arrival_time <= deadline
    for node, deadline in job_deadlines.items():
        idx = manager.NodeToIndex(node)
        time_dim.CumulVar(idx).SetRange(0, deadline)

    # ---- Skills eligibility: job can only be served by allowed vehicles
    for node, vehs in allowed_vehicles.items():
        idx = manager.NodeToIndex(node)
        routing.SetAllowedVehiclesForIndex(vehs, idx)

    # ---- Optional dropping with penalty (priority)
    # If allow_drop=False, we keep all jobs mandatory by not adding disjunction.
    if allow_drop:
        for node in job_nodes:
            idx = manager.NodeToIndex(node)
            prio = priority_by_node.get(node, 3)
            # pénalité forte, croissante avec la priorité
            # tu peux ajuster: ici base 50k minutes "virtuel"
            penalty = 50000 * prio
            routing.AddDisjunction([idx], penalty)

    # ---- Search params
    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    params.time_limit.FromSeconds(int(time_limit_s))

    sol = routing.SolveWithParameters(params)
    if sol is None:
        return None

    # Extract solution
    routes = []
    dropped = []
    for node in job_nodes:
        idx = manager.NodeToIndex(node)
        if sol.Value(routing.NextVar(idx)) == idx:  # dropped in disjunction case
            dropped.append(node)

    for v, tech in enumerate(techs):
        idx = routing.Start(v)
        nodes = []
        while not routing.IsEnd(idx):
            nodes.append(manager.IndexToNode(idx))
            idx = sol.Value(routing.NextVar(idx))
        nodes.append(manager.IndexToNode(idx))

        end_time = sol.Value(time_dim.CumulVar(routing.End(v)))
        routes.append({"tech": tech.name, "vehicle": v, "nodes": nodes, "end_time_min": end_time, "max_min": tech.max_minutes})

    return {"routes": routes, "dropped": dropped}

# -----------------------------
# UI Inputs
# -----------------------------
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
    st.write("Chaque intervention : adresse + durée + skills requis + SLA (deadline en heures) + priorité.")
    if "jobs_df" not in st.session_state:
        st.session_state.jobs_df = pd.DataFrame([
            {"Intervention": "Job 1", "Adresse": "", "Durée (min)": 30, "Skills requis": "elec", "SLA (h) (optionnel)": 4.0, "Priorité (1-5)": 5},
            {"Intervention": "Job 2", "Adresse": "", "Durée (min)": 45, "Skills requis": "", "SLA (h) (optionnel)": "", "Priorité (1-5)": 3},
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
        time_limit = st.slider("Temps de calcul max (s)", 5, 45, 15)

    run = st.button("🚀 Optimiser affectation + tournées", type="primary")

with right:
    st.subheader("Résultats")
    if not run:
        st.info("Renseigne des techniciens + interventions puis clique sur **Optimiser**.")
        st.stop()

    techs_df = st.session_state.techs_df.copy().dropna(subset=["Tech", "Adresse départ", "Heures max"])
    jobs_df = st.session_state.jobs_df.copy().dropna(subset=["Intervention", "Adresse", "Durée (min)", "Priorité (1-5)"])

    if len(techs_df) == 0:
        st.error("Ajoute au moins 1 technicien.")
        st.stop()
    if len(jobs_df) == 0:
        st.error("Ajoute au moins 1 intervention.")
        st.stop()

    # --- Geocode techs
    techs: List[Tech] = []
    bad_tech = []
    with st.spinner("Géocodage techniciens (OSM)..."):
        for _, row in techs_df.iterrows():
            name = str(row["Tech"]).strip()
            addr = str(row["Adresse départ"]).strip()
            max_h = float(row["Heures max"])
            skills = parse_tags(str(row.get("Skills (ex: hauteur,elec)", "")))

            g = geocode_osm(addr)
            if g is None:
                bad_tech.append((name, addr))
                continue
            lat, lon, disp = g
            techs.append(Tech(name=name, address_input=addr, address_resolved=disp, lat=lat, lon=lon, max_minutes=hours_to_minutes(max_h), skills=skills))
            time.sleep(0.1)

    if bad_tech:
        st.error("Techniciens non géocodés. Ajoute ville + CP + France.")
        st.dataframe(pd.DataFrame(bad_tech, columns=["Tech", "Adresse"]), use_container_width=True)
        st.stop()

    # --- Geocode jobs
    jobs: List[Job] = []
    bad_jobs = []
    with st.spinner("Géocodage interventions (OSM)..."):
        for _, row in jobs_df.iterrows():
            name = str(row["Intervention"]).strip()
            addr = str(row["Adresse"]).strip()
            dur = int(row["Durée (min)"])
            req = parse_tags(str(row.get("Skills requis", "")))
            prio = int(row["Priorité (1-5)"])
            sla_raw = str(row.get("SLA (h) (optionnel)", "")).strip()

            deadline = None
            if sla_raw != "":
                try:
                    deadline = hours_to_minutes(float(sla_raw))
                except:
                    deadline = None

            g = geocode_osm(addr)
            if g is None:
                bad_jobs.append((name, addr))
                continue
            lat, lon, disp = g
            jobs.append(Job(
                name=name, address_input=addr, address_resolved=disp, lat=lat, lon=lon,
                service_min=dur, required_skills=req, deadline_min=deadline, priority=prio
            ))
            time.sleep(0.1)

    if bad_jobs:
        st.error("Interventions non géocodées. Ajoute ville + CP + France.")
        st.dataframe(pd.DataFrame(bad_jobs, columns=["Intervention", "Adresse"]), use_container_width=True)
        st.stop()

    # --- Build nodes: start per tech + end per tech (duplicate) + all jobs
    nodes: List[Tuple[float, float]] = []
    service_min: List[int] = []

    starts = []
    ends = []

    # start nodes
    for t in techs:
        starts.append(len(nodes))
        nodes.append((t.lat, t.lon))
        service_min.append(0)

    # end nodes (duplicate)
    for t in techs:
        ends.append(len(nodes))
        nodes.append((t.lat, t.lon))
        service_min.append(0)

    job_offset = len(nodes)
    job_nodes = []
    for j in jobs:
        job_nodes.append(len(nodes))
        nodes.append((j.lat, j.lon))
        service_min.append(j.service_min)

    # OSRM matrix
    with st.spinner("Calcul des temps de trajet (OSRM)..."):
        tm = osrm_table_minutes(nodes)

    # --- Eligibility (skills)
    allowed_vehicles: Dict[int, List[int]] = {}
    for idx_node, job in zip(job_nodes, jobs):
        ok = []
        for v, tech in enumerate(techs):
            if job.required_skills.issubset(tech.skills):
                ok.append(v)
        if not ok:
            st.error(
                f"Aucune compétence compatible pour '{job.name}'. "
                f"Requis={sorted(job.required_skills)} ; vérifie les skills des techniciens."
            )
            st.stop()
        allowed_vehicles[idx_node] = ok

    # --- Deadlines
    job_deadlines = {}
    priority_by_node = {}
    for idx_node, job in zip(job_nodes, jobs):
        priority_by_node[idx_node] = job.priority
        if job.deadline_min is not None:
            job_deadlines[idx_node] = job.deadline_min

    # Solve
    with st.spinner("Optimisation (OR-Tools VRP)..."):
        sol = solve_vrp_multi_tech(
            tm=tm,
            service_min=service_min,
            techs=techs,
            starts=starts,
            ends=ends,
            job_nodes=job_nodes,
            job_deadlines=job_deadlines,
            allowed_vehicles=allowed_vehicles,
            allow_drop=allow_drop,
            priority_by_node=priority_by_node,
            time_limit_s=int(time_limit),
        )

    if sol is None:
        st.error(
            "Aucune solution faisable avec ces contraintes (heures max / SLA / skills). "
            "➡️ augmente les heures max, ajoute un tech, ou assouplis les SLA."
        )
        st.stop()

    dropped = sol["dropped"]
    if dropped:
        dropped_names = []
        for node in dropped:
            j = jobs[node - job_offset]
            dropped_names.append(f"{j.name} (prio {j.priority})")
        st.warning("Interventions reportées (drop) : " + ", ".join(dropped_names))

    # Node label
    def label(node: int) -> str:
        if node < len(techs):
            return f"Départ {techs[node].name}"
        if len(techs) <= node < 2 * len(techs):
            return f"Retour {techs[node - len(techs)].name}"
        j = jobs[node - job_offset]
        return j.name

    st.success("Solution trouvée ✅")
    st.markdown("### Tournées par technicien")

    # Display per tech
    assign_rows = []
    for r in sol["routes"]:
        tech_name = r["tech"]
        nodes_route = r["nodes"]
        end_time = r["end_time_min"]
        max_min = r["max_min"]

        jobs_in_route = []
        for n in nodes_route:
            if n >= job_offset:
                job = jobs[n - job_offset]
                jobs_in_route.append(job)
                assign_rows.append({"Intervention": job.name, "Technicien": tech_name, "Priorité": job.priority, "SLA(h)": (None if job.deadline_min is None else job.deadline_min/60.0)})

        with st.expander(f"{tech_name} — {len(jobs_in_route)} interventions — fin à {end_time} min (max {max_min} min)", expanded=True):
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
                    "SLA (h)": [("" if j.deadline_min is None else round(j.deadline_min/60.0, 2)) for j in jobs_in_route],
                })
                st.dataframe(df, use_container_width=True, hide_index=True)

            st.caption("Chemin: " + " → ".join(label(n) for n in nodes_route))

    st.markdown("### Synthèse (affectation)")
    if assign_rows:
        st.dataframe(pd.DataFrame(assign_rows).sort_values(["Technicien", "Priorité"], ascending=[True, False]), use_container_width=True, hide_index=True)
    else:
        st.write("Aucune affectation (toutes les interventions ont été reportées ou aucune saisie).")
