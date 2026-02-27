import time
import math
import requests
import pandas as pd
import streamlit as st
from dataclasses import dataclass
from typing import List, Optional, Tuple, Set, Dict
from datetime import datetime, time as dtime
from zoneinfo import ZoneInfo

from ortools.constraint_solver import pywrapcp, routing_enums_pb2

# ---- Carte optionnelle
MAP_ENABLED = True
try:
    import folium
    from streamlit_folium import st_folium
except Exception:
    MAP_ENABLED = False


# =========================================================
# UI
# =========================================================
st.set_page_config(page_title="Optimisation tournées multi-tech (OSM/OSRM)", layout="wide")
st.title("Optimisation tournées multi-techniciens (OSM/OSRM)")
st.caption(
    "Affectation + ordre. Contraintes: heures max/tech, skills (menus), SLA, HO/HNO, validation adresses. "
    "Options par technicien: enchaîner ou retour agence entre jobs, et retour agence en fin."
)

# =========================================================
# Models
# =========================================================
@dataclass
class Tech:
    tech_id: str
    name: str
    address_input: str
    address_resolved: str
    lat: float
    lon: float
    max_minutes: int
    chain_jobs: bool          # True: job->job autorisé / False: retour agence entre chaque job
    return_end: bool          # True: fin à l'agence / False: fin libre
    skills: Set[str]

@dataclass
class Job:
    job_id: str
    name: str
    address_input: str
    address_resolved: str
    lat: float
    lon: float
    service_min: int
    required_skills: Set[str]
    deadline_min: Optional[int]
    priority: int
    ho_mode: str


# =========================================================
# Helpers
# =========================================================
def safe_str(x) -> str:
    if x is None:
        return ""
    s = str(x)
    return "" if s.lower() == "nan" else s

def normalize_skill(s: str) -> str:
    return str(s).strip().lower().replace(" ", "_")

def clean_skills_list(skills: List[str]) -> List[str]:
    return sorted({normalize_skill(s) for s in skills if str(s).strip() and str(s).strip().lower() != "nan"})

def to_float_or_none(x) -> Optional[float]:
    if x is None:
        return None
    s = str(x).strip()
    if s == "" or s.lower() == "nan":
        return None
    try:
        return float(s)
    except:
        return None

def hours_to_minutes(h: float) -> int:
    if h is None or (isinstance(h, float) and math.isnan(h)):
        return 0
    if h < 0:
        h = 0
    return int(round(h * 60))

def parse_hhmm(s: str) -> dtime:
    hh, mm = s.split(":")
    return dtime(hour=int(hh), minute=int(mm))

def minutes_since_day_start(dt: datetime) -> int:
    return dt.hour * 60 + dt.minute


# =========================================================
# Geocoding OSM
# =========================================================
NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"

@st.cache_data(show_spinner=False, ttl=24 * 3600)
def geocode_osm(address: str) -> Optional[Tuple[float, float, str]]:
    address = safe_str(address).strip()
    if not address:
        return None
    headers = {"User-Agent": "tournee-optimizer/1.0", "Accept-Language": "fr"}
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


# =========================================================
# OSRM matrix
# =========================================================
OSRM_TABLE_URL = "https://router.project-osrm.org/table/v1/driving/"

@st.cache_data(show_spinner=False, ttl=6 * 3600)
def osrm_table_minutes(coords: List[Tuple[float, float]]) -> List[List[int]]:
    if len(coords) < 2:
        return [[0]]
    coord_str = ";".join([f"{lon},{lat}" for (lat, lon) in coords])
    url = OSRM_TABLE_URL + coord_str
    r = requests.get(url, params={"annotations": "duration"}, timeout=30)
    r.raise_for_status()
    durations = r.json()["durations"]
    return [[int(round((d or 0) / 60.0)) for d in row] for row in durations]


# =========================================================
# HO/HNO -> one time window
# =========================================================
def compute_job_time_window(
    ho_mode: str,
    start_min: int,
    open_start_min: int,
    open_end_min: int,
    horizon_min: int,
) -> Optional[Tuple[int, int]]:
    ho_mode = ho_mode.upper().strip()
    if ho_mode == "INDIFF":
        return (start_min, horizon_min)

    if ho_mode == "HO":
        if start_min >= open_end_min:
            return None
        earliest = max(start_min, open_start_min)
        latest = open_end_min
        if earliest > latest:
            return None
        return (earliest, latest)

    if ho_mode == "HNO":
        if start_min < open_start_min:
            return (start_min, open_start_min)
        if open_start_min <= start_min < open_end_min:
            return (open_end_min, horizon_min)
        return (start_min, horizon_min)

    return (start_min, horizon_min)


# =========================================================
# Solver
# =========================================================
def solve_vrp_multi_tech(
    tm_min: List[List[int]],
    service_min: List[int],
    techs: List[Tech],
    starts: List[int],
    ends: List[int],
    job_nodes: List[int],
    job_time_windows: Dict[int, Tuple[int, int]],
    job_deadlines: Dict[int, int],
    allowed_vehicles: Dict[int, List[int]],
    priority_by_node: Dict[int, int],
    allow_drop: bool,
    time_limit_s: int,
    # needed for "enchaîner" rule:
    is_depot_node: List[bool],
    is_job_node: List[bool],
):
    n = len(tm_min)
    m = len(techs)
    manager = pywrapcp.RoutingIndexManager(n, m, starts, ends)
    routing = pywrapcp.RoutingModel(manager)

    BIGM = 10**7  # cost to forbid transitions

    # ---- cost callback PER VEHICLE to enforce "return between jobs"
    def make_vehicle_cost_cb(v: int):
        chain_ok = techs[v].chain_jobs

        def cb(from_i, to_i):
            f = manager.IndexToNode(from_i)
            t = manager.IndexToNode(to_i)

            base = int(tm_min[f][t])

            if not chain_ok:
                # forbid job->job for this vehicle
                if is_job_node[f] and is_job_node[t]:
                    return BIGM
            return base

        return cb

    cost_cbs = []
    for v in range(m):
        idx = routing.RegisterTransitCallback(make_vehicle_cost_cb(v))
        routing.SetArcCostEvaluatorOfVehicle(idx, v)
        cost_cbs.append(idx)

    # ---- time dimension (travel + service)
    def time_cb(from_i, to_i):
        f = manager.IndexToNode(from_i)
        t = manager.IndexToNode(to_i)
        return int(tm_min[f][t] + service_min[f])

    time_idx = routing.RegisterTransitCallback(time_cb)
    routing.AddDimension(time_idx, 0, 10**9, True, "Time")
    time_dim = routing.GetDimensionOrDie("Time")

    # max time per tech
    for v, tech in enumerate(techs):
        time_dim.CumulVar(routing.End(v)).SetMax(int(tech.max_minutes))

    # HO/HNO windows
    for node, (a, b) in job_time_windows.items():
        idx = manager.NodeToIndex(int(node))
        time_dim.CumulVar(idx).SetRange(int(a), int(b))

    # SLA deadlines
    for node, dl in job_deadlines.items():
        idx = manager.NodeToIndex(int(node))
        time_dim.CumulVar(idx).SetRange(0, int(dl))

    # skills eligibility (robust)
    for node, vehs in allowed_vehicles.items():
        idx = int(manager.NodeToIndex(int(node)))
        if routing.IsStart(idx) or routing.IsEnd(idx):
            continue
        vehs_clean = sorted({int(v) for v in vehs})
        routing.VehicleVar(idx).SetValues(vehs_clean)

    # drop (optional)
    if allow_drop:
        for node in job_nodes:
            idx = manager.NodeToIndex(int(node))
            prio = int(priority_by_node.get(int(node), 3))
            penalty = 200000 * prio
            routing.AddDisjunction([idx], penalty)

    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    params.time_limit.FromSeconds(int(time_limit_s))

    sol = routing.SolveWithParameters(params)
    if sol is None:
        return None

    dropped = []
    if allow_drop:
        for node in job_nodes:
            idx = manager.NodeToIndex(int(node))
            if sol.Value(routing.NextVar(idx)) == idx:
                dropped.append(int(node))

    routes = []
    for v, tech in enumerate(techs):
        idx = routing.Start(v)
        nodes = []
        while not routing.IsEnd(idx):
            nodes.append(manager.IndexToNode(idx))
            idx = sol.Value(routing.NextVar(idx))
        nodes.append(manager.IndexToNode(idx))
        end_time = sol.Value(time_dim.CumulVar(routing.End(v)))
        routes.append({"vehicle": v, "tech": tech.name, "tech_id": tech.tech_id, "nodes": nodes, "end_time_min": int(end_time), "max_min": int(tech.max_minutes)})

    return {"routes": routes, "dropped": dropped}


# =========================================================
# Session state init (IDs + options)
# =========================================================
if "skills_df" not in st.session_state:
    st.session_state.skills_df = pd.DataFrame([{"Skill": "elec_b1v"}, {"Skill": "travail_en_hauteur"}])

if "techs_df" not in st.session_state:
    st.session_state.techs_df = pd.DataFrame(
        [
            {"ID": "T1", "Tech": "Tech A", "Adresse agence": "", "Heures max": 7.5, "Enchaîner ?": "OUI", "Retour fin ?": "OUI"},
            {"ID": "T2", "Tech": "Tech B", "Adresse agence": "", "Heures max": 8.0, "Enchaîner ?": "OUI", "Retour fin ?": "OUI"},
        ]
    )

if "jobs_df" not in st.session_state:
    st.session_state.jobs_df = pd.DataFrame(
        [
            {"ID": "J1", "Intervention": "Job 1", "Adresse": "", "Durée (min)": 30, "HO/HNO": "HO", "SLA (h) optionnel": 4.0, "Priorité (1-5)": 5},
            {"ID": "J2", "Intervention": "Job 2", "Adresse": "", "Durée (min)": 45, "HO/HNO": "INDIFF", "SLA (h) optionnel": "", "Priorité (1-5)": 3},
        ]
    )

if "tech_skills" not in st.session_state:
    st.session_state.tech_skills = {}  # tech_id -> list[str]
if "job_skills" not in st.session_state:
    st.session_state.job_skills = {}   # job_id  -> list[str]


# =========================================================
# Sidebar params
# =========================================================
with st.sidebar:
    st.header("Paramètres journée")
    tz_name = st.selectbox("Fuseau horaire", ["Europe/Paris", "UTC"], index=0)
    tz = ZoneInfo(tz_name)

    now = datetime.now(tz)
    default_start = now.strftime("%H:%M")
    start_hhmm = st.text_input("Heure de départ (HH:MM)", value=default_start)
    open_start = st.text_input("Début horaires ouvrés (HH:MM)", value="08:00")
    open_end = st.text_input("Fin horaires ouvrés (HH:MM)", value="18:00")

    horizon_hours = st.slider("Horizon planification (heures)", 4, 24, 12)
    allow_drop = st.checkbox("Autoriser le report (drop) si impossible", value=False)
    time_limit = st.slider("Temps de calcul max (s)", 5, 60, 15)


# =========================================================
# Layout
# =========================================================
left, right = st.columns([1.25, 1.75])

with left:
    st.subheader("0) Référentiel skills")
    skills_df = st.data_editor(
        st.session_state.skills_df,
        num_rows="dynamic",
        use_container_width=True,
        hide_index=True,
        column_config={"Skill": st.column_config.TextColumn(required=True)},
    )
    skills_list = clean_skills_list(skills_df["Skill"].tolist() if "Skill" in skills_df.columns else [])
    st.session_state.skills_df = pd.DataFrame([{"Skill": s} for s in skills_list]) if skills_list else pd.DataFrame([{"Skill": ""}])

    st.divider()
    st.subheader("1) Techniciens (agence + règles)")
    techs_df = st.data_editor(
        st.session_state.techs_df,
        num_rows="dynamic",
        use_container_width=True,
        hide_index=True,
        column_config={
            "ID": st.column_config.TextColumn(required=True),
            "Tech": st.column_config.TextColumn(required=True),
            "Adresse agence": st.column_config.TextColumn(required=True),
            "Heures max": st.column_config.NumberColumn(min_value=0.0, step=0.5, required=True),
            "Enchaîner ?": st.column_config.SelectboxColumn(options=["OUI", "NON"], required=True),
            "Retour fin ?": st.column_config.SelectboxColumn(options=["OUI", "NON"], required=True),
        },
    )
    st.session_state.techs_df = techs_df

    if skills_list:
        st.markdown("**Skills par technicien (menus)**")
        for _, row in techs_df.iterrows():
            tech_id = safe_str(row.get("ID", "")).strip()
            tech_name = safe_str(row.get("Tech", "")).strip()
            if not tech_id:
                continue
            default = st.session_state.tech_skills.get(tech_id, [])
            chosen = st.multiselect(
                f"Skills — {tech_name or tech_id}",
                options=skills_list,
                default=[s for s in default if s in skills_list],
                key=f"tech_skills__{tech_id}",
            )
            st.session_state.tech_skills[tech_id] = chosen

    st.divider()
    st.subheader("2) Interventions")
    jobs_df = st.data_editor(
        st.session_state.jobs_df,
        num_rows="dynamic",
        use_container_width=True,
        hide_index=True,
        column_config={
            "ID": st.column_config.TextColumn(required=True),
            "Intervention": st.column_config.TextColumn(required=True),
            "Adresse": st.column_config.TextColumn(required=True),
            "Durée (min)": st.column_config.NumberColumn(min_value=0, step=5, required=True),
            "HO/HNO": st.column_config.SelectboxColumn(options=["HO", "HNO", "INDIFF"], required=True),
            "SLA (h) optionnel": st.column_config.TextColumn(required=False),
            "Priorité (1-5)": st.column_config.NumberColumn(min_value=1, max_value=5, step=1, required=True),
        },
    )
    st.session_state.jobs_df = jobs_df

    if skills_list:
        st.markdown("**Skills requis par intervention (menus)**")
        for _, row in jobs_df.iterrows():
            job_id = safe_str(row.get("ID", "")).strip()
            job_name = safe_str(row.get("Intervention", "")).strip()
            if not job_id:
                continue
            default = st.session_state.job_skills.get(job_id, [])
            chosen = st.multiselect(
                f"Skills requis — {job_name or job_id}",
                options=skills_list,
                default=[s for s in default if s in skills_list],
                key=f"job_skills__{job_id}",
            )
            st.session_state.job_skills[job_id] = chosen

    st.divider()
    run = st.button("🚀 Optimiser", type="primary")


with right:
    st.subheader("Résultats")

    # Parse time params
    try:
        start_dt = datetime.combine(now.date(), parse_hhmm(start_hhmm), tzinfo=tz)
        start_min = minutes_since_day_start(start_dt)
        open_start_min = parse_hhmm(open_start).hour * 60 + parse_hhmm(open_start).minute
        open_end_min = parse_hhmm(open_end).hour * 60 + parse_hhmm(open_end).minute
    except Exception:
        st.error("Format d’heure invalide. Utilise HH:MM (ex: 08:00).")
        st.stop()

    horizon_min = min(1440, horizon_hours * 60)

    if not run:
        st.info("Renseigne les tableaux, puis clique sur **Optimiser**.")
        st.stop()

    # Dataframes & checks
    techs_df = st.session_state.techs_df.copy().dropna(subset=["ID", "Tech", "Adresse agence", "Heures max", "Enchaîner ?", "Retour fin ?"])
    jobs_df = st.session_state.jobs_df.copy().dropna(subset=["ID", "Intervention", "Adresse", "Durée (min)", "HO/HNO", "Priorité (1-5)"])

    if len(techs_df) == 0:
        st.error("Ajoute au moins 1 technicien.")
        st.stop()
    if len(jobs_df) == 0:
        st.error("Ajoute au moins 1 intervention.")
        st.stop()

    if techs_df["ID"].astype(str).duplicated().any():
        st.error("IDs techniciens dupliqués (T1, T2, ...).")
        st.stop()
    if jobs_df["ID"].astype(str).duplicated().any():
        st.error("IDs jobs dupliqués (J1, J2, ...).")
        st.stop()

    # -----------------------------------------------------
    # Geocoding + validation
    # -----------------------------------------------------
    tech_rows = []
    bad_tech = []
    with st.spinner("Validation adresses techniciens (OSM)…"):
        for _, row in techs_df.iterrows():
            tech_id = safe_str(row["ID"]).strip()
            name = safe_str(row["Tech"]).strip()
            addr = safe_str(row["Adresse agence"]).strip()
            max_h = float(row["Heures max"])
            chain_jobs = safe_str(row["Enchaîner ?"]).strip().upper() == "OUI"
            return_end = safe_str(row["Retour fin ?"]).strip().upper() == "OUI"
            skills = set(st.session_state.tech_skills.get(tech_id, []))

            g = None
            try:
                g = geocode_osm(addr)
            except Exception:
                g = None

            if g is None:
                bad_tech.append({"Type": "Technicien", "ID": tech_id, "Nom": name, "Adresse": addr,
                                "Erreur": "Adresse non reconnue. Ajoute: n°, rue, CP, ville, France."})
                tech_rows.append({"ID": tech_id, "Tech": name, "Adresse saisie": addr, "Statut": "❌ invalide"})
                continue

            lat, lon, disp = g
            tech_rows.append({"ID": tech_id, "Tech": name, "Statut": "✅ OK", "Adresse résolue": disp,
                              "Enchaîner": "OUI" if chain_jobs else "NON", "Retour fin": "OUI" if return_end else "NON"})
            tech_rows[-1]["_lat"] = lat
            tech_rows[-1]["_lon"] = lon
            tech_rows[-1]["_max_min"] = hours_to_minutes(max_h)
            tech_rows[-1]["_chain"] = chain_jobs
            tech_rows[-1]["_return_end"] = return_end
            tech_rows[-1]["_skills"] = skills
            time.sleep(0.08)

    job_rows = []
    bad_jobs = []
    with st.spinner("Validation adresses interventions (OSM)…"):
        for _, row in jobs_df.iterrows():
            job_id = safe_str(row["ID"]).strip()
            name = safe_str(row["Intervention"]).strip()
            addr = safe_str(row["Adresse"]).strip()
            dur = int(row["Durée (min)"])
            ho_mode = safe_str(row["HO/HNO"]).strip().upper()
            prio = int(row["Priorité (1-5)"])
            sla_h = to_float_or_none(row.get("SLA (h) optionnel", None))
            deadline = hours_to_minutes(sla_h) if sla_h is not None else None
            req_skills = set(st.session_state.job_skills.get(job_id, []))

            g = None
            try:
                g = geocode_osm(addr)
            except Exception:
                g = None

            if g is None:
                bad_jobs.append({"Type": "Intervention", "ID": job_id, "Nom": name, "Adresse": addr,
                                 "Erreur": "Adresse non reconnue. Ajoute: n°, rue, CP, ville, France."})
                job_rows.append({"ID": job_id, "Intervention": name, "Adresse saisie": addr, "Statut": "❌ invalide"})
                continue

            lat, lon, disp = g
            job_rows.append({"ID": job_id, "Intervention": name, "Statut": "✅ OK", "Adresse résolue": disp,
                             "HO/HNO": ho_mode, "Priorité": prio, "SLA(h)": "" if deadline is None else round(deadline/60.0, 2)})
            job_rows[-1]["_lat"] = lat
            job_rows[-1]["_lon"] = lon
            job_rows[-1]["_dur"] = dur
            job_rows[-1]["_ho"] = ho_mode
            job_rows[-1]["_prio"] = prio
            job_rows[-1]["_deadline"] = deadline
            job_rows[-1]["_req_skills"] = req_skills
            time.sleep(0.08)

    st.markdown("### Validation des adresses")
    cA, cB = st.columns(2)
    with cA:
        st.caption("Techniciens")
        st.dataframe(pd.DataFrame(tech_rows), use_container_width=True, hide_index=True)
    with cB:
        st.caption("Interventions")
        st.dataframe(pd.DataFrame(job_rows), use_container_width=True, hide_index=True)

    if bad_tech or bad_jobs:
        st.error("Certaines adresses sont invalides. Corrige-les puis relance.")
        if bad_tech:
            st.dataframe(pd.DataFrame(bad_tech), use_container_width=True, hide_index=True)
        if bad_jobs:
            st.dataframe(pd.DataFrame(bad_jobs), use_container_width=True, hide_index=True)
        st.stop()

    # -----------------------------------------------------
    # Build objects
    # -----------------------------------------------------
    techs: List[Tech] = []
    for r in tech_rows:
        techs.append(Tech(
            tech_id=r["ID"],
            name=r["Tech"],
            address_input=next((x for x in techs_df[techs_df["ID"] == r["ID"]]["Adresse agence"].tolist()), ""),
            address_resolved=r.get("Adresse résolue", ""),
            lat=r["_lat"], lon=r["_lon"],
            max_minutes=r["_max_min"],
            chain_jobs=r["_chain"],
            return_end=r["_return_end"],
            skills=r["_skills"],
        ))

    jobs: List[Job] = []
    for r in job_rows:
        jobs.append(Job(
            job_id=r["ID"],
            name=r["Intervention"],
            address_input=next((x for x in jobs_df[jobs_df["ID"] == r["ID"]]["Adresse"].tolist()), ""),
            address_resolved=r.get("Adresse résolue", ""),
            lat=r["_lat"], lon=r["_lon"],
            service_min=r["_dur"],
            required_skills=r["_req_skills"],
            deadline_min=r["_deadline"],
            priority=r["_prio"],
            ho_mode=r["_ho"],
        ))

    # -----------------------------------------------------
    # Nodes:
    # start per tech (agency)
    # end per tech: if return_end=True -> agency end node, else -> virtual end node
    # + jobs nodes
    # + virtual end nodes (one per tech) placed at same coords as agency (only for matrix)
    # -----------------------------------------------------
    coords: List[Tuple[float, float]] = []
    service_min: List[int] = []
    starts: List[int] = []

    # start nodes
    for t in techs:
        starts.append(len(coords))
        coords.append((t.lat, t.lon))
        service_min.append(0)

    # end nodes — we will create either agency end node or virtual end node
    ends: List[int] = []
    # keep mapping per vehicle
    end_is_virtual: List[bool] = []

    # create ends now:
    for t in techs:
        ends.append(len(coords))
        coords.append((t.lat, t.lon))  # same coord
        service_min.append(0)
        # mark virtual or not (we'll use it in cost callback via big matrix tweak)
        end_is_virtual.append(not t.return_end)

    # job nodes
    job_offset = len(coords)
    job_nodes: List[int] = []
    for j in jobs:
        job_nodes.append(len(coords))
        coords.append((j.lat, j.lon))
        service_min.append(int(j.service_min))

    # -----------------------------------------------------
    # Build matrix
    # -----------------------------------------------------
    with st.spinner("Calcul des temps de trajet (OSRM)…"):
        tm = osrm_table_minutes(coords)

    # -----------------------------------------------------
    # If end is virtual, we want travel cost to end to be zero from any node (finish anywhere)
    # We do that by forcing tm[f][end_node] = 0 for all f, for that vehicle’s end node.
    # Time dimension uses tm too, so we set to 0 for time as well (finish anywhere without extra travel).
    # -----------------------------------------------------
    for v, isvirt in enumerate(end_is_virtual):
        if isvirt:
            end_node = ends[v]
            for f in range(len(tm)):
                tm[f][end_node] = 0  # finish anywhere with no added travel

    # -----------------------------------------------------
    # Constraints: skills, HO/HNO, SLA, priorities
    # -----------------------------------------------------
    allowed_vehicles: Dict[int, List[int]] = {}
    job_time_windows: Dict[int, Tuple[int, int]] = {}
    job_deadlines: Dict[int, int] = {}
    priority_by_node: Dict[int, int] = {}

    for node, job in zip(job_nodes, jobs):
        tw = compute_job_time_window(
            ho_mode=job.ho_mode,
            start_min=start_min,
            open_start_min=open_start_min,
            open_end_min=open_end_min,
            horizon_min=min(1440, horizon_hours * 60),
        )
        if tw is None:
            if allow_drop:
                job_time_windows[int(node)] = (start_min, min(1440, horizon_hours * 60))
            else:
                st.error(f"'{job.name}' impossible aujourd’hui avec HO/HNO + heure de départ. Active report ou change l’heure.")
                st.stop()
        else:
            job_time_windows[int(node)] = (int(tw[0]), int(tw[1]))

        if job.deadline_min is not None:
            job_deadlines[int(node)] = int(job.deadline_min)

        priority_by_node[int(node)] = int(job.priority)

        ok = []
        for v, tech in enumerate(techs):
            if job.required_skills.issubset(tech.skills):
                ok.append(v)

        if not ok:
            st.error(
                f"Aucun technicien compatible pour '{job.name}'. "
                f"Skills requis={sorted(job.required_skills)}."
            )
            st.stop()

        allowed_vehicles[int(node)] = [int(v) for v in ok]

    # Flags nodes for "return between jobs" restriction
    n_nodes = len(tm)
    is_depot_node = [False] * n_nodes
    is_job_node = [False] * n_nodes

    # depot nodes are starts and ends (they represent the agency)
    for s in starts:
        is_depot_node[s] = True
    for e in ends:
        is_depot_node[e] = True
    for jn in job_nodes:
        is_job_node[jn] = True

    # -----------------------------------------------------
    # Solve
    # -----------------------------------------------------
    with st.spinner("Optimisation (OR-Tools)…"):
        sol = solve_vrp_multi_tech(
            tm_min=tm,
            service_min=service_min,
            techs=techs,
            starts=starts,
            ends=ends,
            job_nodes=job_nodes,
            job_time_windows=job_time_windows,
            job_deadlines=job_deadlines,
            allowed_vehicles=allowed_vehicles,
            priority_by_node=priority_by_node,
            allow_drop=allow_drop,
            time_limit_s=int(time_limit),
            is_depot_node=is_depot_node,
            is_job_node=is_job_node,
        )

    if sol is None:
        st.error("Aucune solution faisable. ➜ augmente heures max / assouplis SLA / ajoute un tech / active report.")
        st.stop()

    dropped = sol["dropped"]
    if dropped:
        dropped_names = [jobs[int(n) - job_offset].name for n in dropped]
        st.warning("Interventions reportées : " + ", ".join(dropped_names))

    st.success("Solution trouvée ✅")
    st.markdown("### Tournées par technicien")

    def label(node: int) -> str:
        node = int(node)
        if node < len(techs):
            return f"Agence départ {techs[node].name}"
        if len(techs) <= node < 2 * len(techs):
            # end node (agency or virtual)
            v = node - len(techs)
            return "Fin libre" if end_is_virtual[v] else f"Retour agence {techs[v].name}"
        return jobs[node - job_offset].name

    assign_rows = []
    routes_for_map = []

    for r in sol["routes"]:
        tech_name = r["tech"]
        v = r["vehicle"]
        nodes_route = r["nodes"]
        end_time = r["end_time_min"]
        max_min = r["max_min"]

        route_jobs = []
        route_latlon = []

        for n in nodes_route:
            n = int(n)
            route_latlon.append(coords[n])
            if n >= job_offset:
                jb = jobs[n - job_offset]
                route_jobs.append(jb)
                assign_rows.append({
                    "Intervention": jb.name,
                    "ID": jb.job_id,
                    "Technicien": tech_name,
                    "Enchaîner ?": "OUI" if techs[v].chain_jobs else "NON",
                    "Retour fin ?": "OUI" if techs[v].return_end else "NON",
                    "HO/HNO": jb.ho_mode,
                    "Priorité": jb.priority,
                    "SLA(h)": "" if jb.deadline_min is None else round(jb.deadline_min/60.0, 2),
                    "Skills requis": ", ".join(sorted(jb.required_skills)),
                })

        routes_for_map.append((tech_name, route_latlon))

        with st.expander(
            f"{tech_name} — {len(route_jobs)} interventions — fin {end_time} min (max {max_min} min) "
            f"| Enchaîner={'OUI' if techs[v].chain_jobs else 'NON'} | Retour fin={'OUI' if techs[v].return_end else 'NON'}",
            expanded=True
        ):
            if not route_jobs:
                st.write("Aucune intervention affectée.")
            else:
                st.dataframe(pd.DataFrame({
                    "Ordre": list(range(1, len(route_jobs)+1)),
                    "ID": [j.job_id for j in route_jobs],
                    "Intervention": [j.name for j in route_jobs],
                    "Adresse (OSM)": [j.address_resolved for j in route_jobs],
                    "Durée (min)": [j.service_min for j in route_jobs],
                    "HO/HNO": [j.ho_mode for j in route_jobs],
                    "Priorité": [j.priority for j in route_jobs],
                    "SLA(h)": ["" if j.deadline_min is None else round(j.deadline_min/60.0, 2) for j in route_jobs],
                    "Skills requis": [", ".join(sorted(j.required_skills)) for j in route_jobs],
                }), use_container_width=True, hide_index=True)

            st.caption("Chemin : " + " → ".join(label(n) for n in nodes_route))

    st.markdown("### Synthèse affectation")
    if assign_rows:
        st.dataframe(
            pd.DataFrame(assign_rows).sort_values(["Technicien", "Priorité"], ascending=[True, False]),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.write("Aucune affectation.")

    # -----------------------------------------------------
    # Map
    # -----------------------------------------------------
    st.markdown("### Carte (points + tournées)")
    if not MAP_ENABLED:
        st.warning("Carte désactivée (folium/streamlit-folium non installés). Ajoute-les dans requirements.txt pour l’activer.")
    else:
        all_lat = [c[0] for c in coords]
        all_lon = [c[1] for c in coords]
        center = (sum(all_lat)/len(all_lat), sum(all_lon)/len(all_lon))

        m = folium.Map(location=center, zoom_start=11, control_scale=True)

        # agency markers
        for t in techs:
            folium.Marker(
                location=(t.lat, t.lon),
                tooltip=f"Agence {t.name} ({t.tech_id})",
                icon=folium.Icon(icon="home", prefix="fa"),
            ).add_to(m)

        # jobs markers
        for j in jobs:
            folium.CircleMarker(
                location=(j.lat, j.lon),
                radius=6,
                tooltip=f"{j.name} ({j.job_id}) | {j.ho_mode} | prio {j.priority}",
                fill=True,
            ).add_to(m)

        # polylines (simple)
        for tech_name, latlon_list in routes_for_map:
            poly = [(lat, lon) for (lat, lon) in latlon_list]
            folium.PolyLine(poly, tooltip=f"Tournée {tech_name}").add_to(m)

        st_folium(m, use_container_width=True, height=520)
