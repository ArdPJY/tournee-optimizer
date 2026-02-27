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

# ------------------- Map (optional) -------------------
MAP_ENABLED = True
try:
    import folium
    from streamlit_folium import st_folium
except Exception:
    MAP_ENABLED = False

# =========================================================
# UI
# =========================================================
st.set_page_config(page_title="Optimisation tournées (OSM/OSRM)", layout="wide")
st.title("Optimisation tournées multi-techniciens (OSM/OSRM)")
st.caption(
    "Affectation + ordre. Contraintes: heures max, skills (checkbox dans les tableaux), SLA, HO/HNO, "
    "validation adresses, agences. Règle par intervention: 'Retour agence après ?'."
)

# =========================================================
# Helpers
# =========================================================
def safe_str(x) -> str:
    if x is None:
        return ""
    s = str(x)
    return "" if s.lower() == "nan" else s

def normalize_skill(s: str) -> str:
    return str(s).strip().lower().replace(" ", "_").replace("-", "_")

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

def ensure_skill_columns(df: pd.DataFrame, skills: List[str], prefix: str) -> pd.DataFrame:
    """
    Ajoute/enlève dynamiquement des colonnes checkbox de skills:
    prefix = "S_" pour techniciens, "R_" pour jobs.
    """
    df = df.copy()
    wanted_cols = [f"{prefix}{s}" for s in skills]

    # add missing
    for c in wanted_cols:
        if c not in df.columns:
            df[c] = False

    # drop obsolete (skills supprimées du référentiel)
    for c in list(df.columns):
        if c.startswith(prefix) and c not in wanted_cols:
            df.drop(columns=[c], inplace=True)

    return df

def skills_from_row(row: pd.Series, skills: List[str], prefix: str) -> Set[str]:
    out = set()
    for s in skills:
        c = f"{prefix}{s}"
        if c in row and bool(row[c]):
            out.add(s)
    return out

# =========================================================
# OSM Geocoding (Nominatim)
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
# OSRM Matrix
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
# HO/HNO time window
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
# Data models
# =========================================================
@dataclass
class Agency:
    agency_id: str
    name: str
    address_input: str
    address_resolved: str
    lat: float
    lon: float

@dataclass
class Tech:
    tech_id: str
    name: str
    agency_id: str
    max_minutes: int
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
    return_after: bool  # ✅ ce que tu demandes

# =========================================================
# OR-Tools Solver with "return-after-job"
# =========================================================
def solve(
    tm_min: List[List[int]],
    service_min: List[int],
    techs: List[Tech],
    start_nodes: List[int],
    end_free_nodes: List[int],       # end = free end (cost 0 from anywhere)
    depot_nodes: List[int],          # per vehicle "depot return node"
    job_nodes: List[int],
    job_time_windows: Dict[int, Tuple[int, int]],
    job_deadlines: Dict[int, int],
    allowed_vehicles: Dict[int, List[int]],
    priority_by_node: Dict[int, int],
    return_after_by_node: Dict[int, bool],
    allow_drop: bool,
    time_limit_s: int,
):
    n = len(tm_min)
    m = len(techs)

    # RoutingIndexManager supports one end per vehicle -> we set end as "free end"
    manager = pywrapcp.RoutingIndexManager(n, m, start_nodes, end_free_nodes)
    routing = pywrapcp.RoutingModel(manager)

    BIGM = 10**7

    # We need per-vehicle cost to enforce:
    # if from-node is a job with return_after=True -> next MUST be that vehicle's depot node
    def make_vehicle_cost_cb(v: int):
        depot_node = depot_nodes[v]
        end_node = end_free_nodes[v]

        def cb(from_i, to_i):
            f = manager.IndexToNode(from_i)
            t = manager.IndexToNode(to_i)

            base = int(tm_min[f][t])

            if return_after_by_node.get(f, False):
                # Allowed next: depot_node only
                if t != depot_node:
                    return BIGM
                return base

            # From depot node, you can go to jobs or end (end cost already 0)
            # No extra restriction here.
            return base

        return cb

    for v in range(m):
        idx = routing.RegisterTransitCallback(make_vehicle_cost_cb(v))
        routing.SetArcCostEvaluatorOfVehicle(idx, v)

    # Time dimension: travel + service at FROM
    def time_cb(from_i, to_i):
        f = manager.IndexToNode(from_i)
        t = manager.IndexToNode(to_i)
        return int(tm_min[f][t] + service_min[f])

    time_idx = routing.RegisterTransitCallback(time_cb)
    routing.AddDimension(time_idx, 0, 10**9, True, "Time")
    time_dim = routing.GetDimensionOrDie("Time")

    # Max work time per tech
    for v, tech in enumerate(techs):
        time_dim.CumulVar(routing.End(v)).SetMax(int(tech.max_minutes))

    # Time windows
    for node, (a, b) in job_time_windows.items():
        idx = manager.NodeToIndex(int(node))
        time_dim.CumulVar(idx).SetRange(int(a), int(b))

    # SLA deadlines
    for node, dl in job_deadlines.items():
        idx = manager.NodeToIndex(int(node))
        time_dim.CumulVar(idx).SetRange(0, int(dl))

    # Skills eligibility
    for node, vehs in allowed_vehicles.items():
        idx = int(manager.NodeToIndex(int(node)))
        if routing.IsStart(idx) or routing.IsEnd(idx):
            continue
        routing.VehicleVar(idx).SetValues(sorted({int(v) for v in vehs}))

    # Optional drop
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
# Session state init
# =========================================================
if "skills_df" not in st.session_state:
    st.session_state.skills_df = pd.DataFrame([{"Skill": "elec_b1v"}, {"Skill": "travail_en_hauteur"}])

if "agencies_df" not in st.session_state:
    st.session_state.agencies_df = pd.DataFrame(
        [
            {"ID": "A1", "Agence": "Agence Paris", "Adresse": "9 rue du Saule Trapu, 91300 Massy, France"},
        ]
    )

if "techs_df" not in st.session_state:
    st.session_state.techs_df = pd.DataFrame(
        [
            {"ID": "T1", "Technicien": "Tech A", "Agence_ID": "A1", "Heures max": 7.5},
            {"ID": "T2", "Technicien": "Tech B", "Agence_ID": "A1", "Heures max": 8.0},
        ]
    )

if "jobs_df" not in st.session_state:
    st.session_state.jobs_df = pd.DataFrame(
        [
            {"ID": "J1", "Intervention": "Job 1", "Adresse": "Massy, France", "Durée (min)": 30, "HO/HNO": "HO", "SLA (h) optionnel": 4.0, "Priorité (1-5)": 5, "Retour agence après ?": False},
            {"ID": "J2", "Intervention": "Job 2", "Adresse": "Palaiseau, France", "Durée (min)": 45, "HO/HNO": "INDIFF", "SLA (h) optionnel": "", "Priorité (1-5)": 3, "Retour agence après ?": True},
        ]
    )

# =========================================================
# Sidebar
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
# Build skill list + ensure skill columns in tables
# =========================================================
skills_list = clean_skills_list(st.session_state.skills_df["Skill"].tolist() if "Skill" in st.session_state.skills_df.columns else [])
st.session_state.techs_df = ensure_skill_columns(st.session_state.techs_df, skills_list, prefix="S_")
st.session_state.jobs_df = ensure_skill_columns(st.session_state.jobs_df, skills_list, prefix="R_")

# =========================================================
# Layout
# =========================================================
left, right = st.columns([1.35, 1.65])

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
    st.subheader("1) Agences (infos)")
    agencies_df = st.data_editor(
        st.session_state.agencies_df,
        num_rows="dynamic",
        use_container_width=True,
        hide_index=True,
        column_config={
            "ID": st.column_config.TextColumn(required=True, help="ID stable ex: A1, A2..."),
            "Agence": st.column_config.TextColumn(required=True),
            "Adresse": st.column_config.TextColumn(required=True),
        },
    )
    st.session_state.agencies_df = agencies_df
    agency_ids = [safe_str(x).strip() for x in agencies_df["ID"].tolist()] if "ID" in agencies_df.columns else []

    st.divider()
    st.subheader("2) Techniciens (skills en checkbox dans le tableau)")
    techs_df = st.data_editor(
        st.session_state.techs_df,
        num_rows="dynamic",
        use_container_width=True,
        hide_index=True,
        column_config={
            "ID": st.column_config.TextColumn(required=True),
            "Technicien": st.column_config.TextColumn(required=True),
            "Agence_ID": st.column_config.SelectboxColumn(options=agency_ids, required=True),
            "Heures max": st.column_config.NumberColumn(min_value=0.0, step=0.5, required=True),
            **{f"S_{s}": st.column_config.CheckboxColumn(help=f"Skill: {s}") for s in skills_list},
        },
    )
    st.session_state.techs_df = techs_df

    st.divider()
    st.subheader("3) Interventions (Retour agence après ?)")
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
            "Retour agence après ?": st.column_config.CheckboxColumn(help="Si coché: après cette intervention, retour dépôt obligatoire."),
            **{f"R_{s}": st.column_config.CheckboxColumn(help=f"Requis: {s}") for s in skills_list},
        },
    )
    st.session_state.jobs_df = jobs_df

    st.divider()
    run = st.button("🚀 Optimiser", type="primary")


with right:
    st.subheader("Résultats")

    # parse time
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

    # basic checks
    agencies_df = st.session_state.agencies_df.copy().dropna(subset=["ID", "Agence", "Adresse"])
    techs_df = st.session_state.techs_df.copy().dropna(subset=["ID", "Technicien", "Agence_ID", "Heures max"])
    jobs_df = st.session_state.jobs_df.copy().dropna(subset=["ID", "Intervention", "Adresse", "Durée (min)", "HO/HNO", "Priorité (1-5)"])

    if agencies_df["ID"].astype(str).duplicated().any():
        st.error("IDs agences dupliqués (A1, A2...).")
        st.stop()
    if techs_df["ID"].astype(str).duplicated().any():
        st.error("IDs techniciens dupliqués (T1, T2...).")
        st.stop()
    if jobs_df["ID"].astype(str).duplicated().any():
        st.error("IDs interventions dupliqués (J1, J2...).")
        st.stop()

    if len(agencies_df) == 0:
        st.error("Ajoute au moins 1 agence.")
        st.stop()
    if len(techs_df) == 0:
        st.error("Ajoute au moins 1 technicien.")
        st.stop()
    if len(jobs_df) == 0:
        st.error("Ajoute au moins 1 intervention.")
        st.stop()

    # =========================================================
    # Geocode agencies + jobs with validation report
    # =========================================================
    agency_map: Dict[str, Agency] = {}
    bad_rows = []

    with st.spinner("Validation adresses agences (OSM)…"):
        for _, row in agencies_df.iterrows():
            aid = safe_str(row["ID"]).strip()
            aname = safe_str(row["Agence"]).strip()
            addr = safe_str(row["Adresse"]).strip()

            g = None
            try:
                g = geocode_osm(addr)
            except Exception:
                g = None

            if g is None:
                bad_rows.append({"Type": "Agence", "ID": aid, "Nom": aname, "Adresse": addr,
                                 "Erreur": "Adresse non reconnue. Ajoute n°, rue, CP, ville, France."})
                continue
            lat, lon, disp = g
            agency_map[aid] = Agency(aid, aname, addr, disp, lat, lon)
            time.sleep(0.08)

    jobs_geo = []
    with st.spinner("Validation adresses interventions (OSM)…"):
        for _, row in jobs_df.iterrows():
            jid = safe_str(row["ID"]).strip()
            jname = safe_str(row["Intervention"]).strip()
            addr = safe_str(row["Adresse"]).strip()

            g = None
            try:
                g = geocode_osm(addr)
            except Exception:
                g = None

            if g is None:
                bad_rows.append({"Type": "Intervention", "ID": jid, "Nom": jname, "Adresse": addr,
                                 "Erreur": "Adresse non reconnue. Ajoute n°, rue, CP, ville, France."})
                continue

            lat, lon, disp = g
            jobs_geo.append((jid, jname, addr, disp, lat, lon))
            time.sleep(0.08)

    st.markdown("### Validation des adresses")
    if bad_rows:
        st.error("Certaines adresses sont invalides. Corrige-les puis relance.")
        st.dataframe(pd.DataFrame(bad_rows), use_container_width=True, hide_index=True)
        st.stop()
    else:
        st.success("Toutes les adresses sont OK ✅")

    # =========================================================
    # Build Techs & Jobs objects
    # =========================================================
    techs: List[Tech] = []
    for _, row in techs_df.iterrows():
        tid = safe_str(row["ID"]).strip()
        tname = safe_str(row["Technicien"]).strip()
        aid = safe_str(row["Agence_ID"]).strip()

        if aid not in agency_map:
            st.error(f"Technicien '{tname}' référence une Agence_ID inconnue: {aid}")
            st.stop()

        max_h = float(row["Heures max"])
        tskills = skills_from_row(row, skills_list, prefix="S_")

        techs.append(Tech(
            tech_id=tid,
            name=tname,
            agency_id=aid,
            max_minutes=hours_to_minutes(max_h),
            skills=tskills
        ))

    jobs: List[Job] = []
    geo_by_id = {jid: (disp, lat, lon, addr, jname) for (jid, jname, addr, disp, lat, lon) in jobs_geo}

    for _, row in jobs_df.iterrows():
        jid = safe_str(row["ID"]).strip()
        jname = safe_str(row["Intervention"]).strip()
        addr = safe_str(row["Adresse"]).strip()
        disp, lat, lon, _, _ = geo_by_id[jid]

        dur = int(row["Durée (min)"])
        ho_mode = safe_str(row["HO/HNO"]).strip().upper()
        prio = int(row["Priorité (1-5)"])
        sla_h = to_float_or_none(row.get("SLA (h) optionnel", None))
        deadline = hours_to_minutes(sla_h) if sla_h is not None else None

        req = skills_from_row(row, skills_list, prefix="R_")
        return_after = bool(row.get("Retour agence après ?", False))

        jobs.append(Job(
            job_id=jid,
            name=jname,
            address_input=addr,
            address_resolved=disp,
            lat=lat, lon=lon,
            service_min=dur,
            required_skills=req,
            deadline_min=deadline,
            priority=prio,
            ho_mode=ho_mode,
            return_after=return_after
        ))

    # =========================================================
    # Build nodes:
    # For each tech/vehicle:
    #   start node = agency coords
    #   depot node = agency coords (return hub)
    #   end_free node = agency coords (virtual end with cost 0 from anywhere)
    # Jobs nodes after that.
    # =========================================================
    coords: List[Tuple[float, float]] = []
    service_min: List[int] = []

    start_nodes: List[int] = []
    depot_nodes: List[int] = []
    end_free_nodes: List[int] = []

    for t in techs:
        ag = agency_map[t.agency_id]

        start_nodes.append(len(coords))
        coords.append((ag.lat, ag.lon))
        service_min.append(0)

        depot_nodes.append(len(coords))
        coords.append((ag.lat, ag.lon))
        service_min.append(0)

        end_free_nodes.append(len(coords))
        coords.append((ag.lat, ag.lon))
        service_min.append(0)

    job_offset = len(coords)
    job_nodes: List[int] = []
    for j in jobs:
        job_nodes.append(len(coords))
        coords.append((j.lat, j.lon))
        service_min.append(int(j.service_min))

    with st.spinner("Calcul des temps de trajet (OSRM)…"):
        tm = osrm_table_minutes(coords)

    # Make end_free cost 0 from anywhere (finish anywhere)
    for v, end_node in enumerate(end_free_nodes):
        for f in range(len(tm)):
            tm[f][end_node] = 0
        # allow depot -> end free at 0 too
        tm[depot_nodes[v]][end_node] = 0

    # =========================================================
    # Build constraints
    # =========================================================
    job_time_windows: Dict[int, Tuple[int, int]] = {}
    job_deadlines: Dict[int, int] = {}
    allowed_vehicles: Dict[int, List[int]] = {}
    priority_by_node: Dict[int, int] = {}
    return_after_by_node: Dict[int, bool] = {}

    for node, job in zip(job_nodes, jobs):
        tw = compute_job_time_window(job.ho_mode, start_min, open_start_min, open_end_min, horizon_min)
        if tw is None:
            if allow_drop:
                job_time_windows[int(node)] = (start_min, horizon_min)
            else:
                st.error(f"'{job.name}' impossible aujourd’hui avec HO/HNO + heure de départ. Active report ou change l’heure.")
                st.stop()
        else:
            job_time_windows[int(node)] = (int(tw[0]), int(tw[1]))

        if job.deadline_min is not None:
            job_deadlines[int(node)] = int(job.deadline_min)

        priority_by_node[int(node)] = int(job.priority)
        return_after_by_node[int(node)] = bool(job.return_after)

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

        allowed_vehicles[int(node)] = ok

    # IMPORTANT:
    # If a job has return_after=True, next MUST be depot for that vehicle.
    # We implement this in the cost callback (BIGM otherwise).
    # BUT we must also tag depot nodes as "return_after=False" to not force depot->depot loops.
    for dn in depot_nodes:
        return_after_by_node[int(dn)] = False
    for sn in start_nodes:
        return_after_by_node[int(sn)] = False
    for en in end_free_nodes:
        return_after_by_node[int(en)] = False

    # =========================================================
    # Solve
    # =========================================================
    with st.spinner("Optimisation (OR-Tools)…"):
        sol = solve(
            tm_min=tm,
            service_min=service_min,
            techs=techs,
            start_nodes=start_nodes,
            end_free_nodes=end_free_nodes,
            depot_nodes=depot_nodes,
            job_nodes=job_nodes,
            job_time_windows=job_time_windows,
            job_deadlines=job_deadlines,
            allowed_vehicles=allowed_vehicles,
            priority_by_node=priority_by_node,
            return_after_by_node=return_after_by_node,
            allow_drop=allow_drop,
            time_limit_s=int(time_limit),
        )

    if sol is None:
        st.error("Aucune solution faisable. ➜ augmente heures max / assouplis SLA / ajoute un tech / active report.")
        st.stop()

    dropped = sol["dropped"]
    if dropped:
        dropped_names = []
        for n in dropped:
            dropped_names.append(jobs[int(n) - job_offset].name)
        st.warning("Interventions reportées : " + ", ".join(dropped_names))

    st.success("Solution trouvée ✅")

    # =========================================================
    # Display
    # =========================================================
    def label(node: int, v: int) -> str:
        node = int(node)
        if node == start_nodes[v]:
            ag = agency_map[techs[v].agency_id]
            return f"Départ agence ({ag.name})"
        if node == depot_nodes[v]:
            ag = agency_map[techs[v].agency_id]
            return f"Retour agence ({ag.name})"
        if node == end_free_nodes[v]:
            return "Fin (libre)"
        if node >= job_offset:
            j = jobs[node - job_offset]
            return f"{j.name}{' [retour]' if j.return_after else ''}"
        return f"Node {node}"

    st.markdown("### Tournées par technicien")
    assign_rows = []
    routes_for_map = []

    for r in sol["routes"]:
        v = r["vehicle"]
        tech_name = r["tech"]
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
                    "Technicien": tech_name,
                    "Intervention": jb.name,
                    "ID": jb.job_id,
                    "Retour après ?": "OUI" if jb.return_after else "NON",
                    "HO/HNO": jb.ho_mode,
                    "Priorité": jb.priority,
                    "SLA(h)": "" if jb.deadline_min is None else round(jb.deadline_min/60.0, 2),
                    "Skills requis": ", ".join(sorted(jb.required_skills)),
                })

        routes_for_map.append((tech_name, route_latlon))

        with st.expander(f"{tech_name} — {len(route_jobs)} interventions — fin {end_time} min (max {max_min} min)", expanded=True):
            if not route_jobs:
                st.write("Aucune intervention affectée.")
            else:
                st.dataframe(pd.DataFrame({
                    "Ordre": list(range(1, len(route_jobs)+1)),
                    "ID": [j.job_id for j in route_jobs],
                    "Intervention": [j.name for j in route_jobs],
                    "Retour après ?": ["OUI" if j.return_after else "NON" for j in route_jobs],
                    "Adresse (OSM)": [j.address_resolved for j in route_jobs],
                    "Durée (min)": [j.service_min for j in route_jobs],
                    "HO/HNO": [j.ho_mode for j in route_jobs],
                    "Priorité": [j.priority for j in route_jobs],
                    "SLA(h)": ["" if j.deadline_min is None else round(j.deadline_min/60.0, 2) for j in route_jobs],
                }), use_container_width=True, hide_index=True)

            st.caption("Chemin : " + " → ".join(label(n, v) for n in nodes_route))

    st.markdown("### Synthèse affectation")
    if assign_rows:
        st.dataframe(pd.DataFrame(assign_rows), use_container_width=True, hide_index=True)
    else:
        st.write("Aucune affectation.")

    # =========================================================
    # Map
    # =========================================================
    st.markdown("### Carte (points + tournées)")
    if not MAP_ENABLED:
        st.warning("Carte désactivée (folium/streamlit-folium non installés). Ajoute-les dans requirements.txt pour l’activer.")
    else:
        all_lat = [c[0] for c in coords]
        all_lon = [c[1] for c in coords]
        center = (sum(all_lat)/len(all_lat), sum(all_lon)/len(all_lon))
        m = folium.Map(location=center, zoom_start=11, control_scale=True)

        # agencies
        for ag in agency_map.values():
            folium.Marker(
                location=(ag.lat, ag.lon),
                tooltip=f"Agence {ag.name} ({ag.agency_id})",
                icon=folium.Icon(icon="home", prefix="fa"),
            ).add_to(m)

        # jobs
        for j in jobs:
            folium.CircleMarker(
                location=(j.lat, j.lon),
                radius=6,
                tooltip=f"{j.name} ({j.job_id}) | retour={j.return_after}",
                fill=True,
            ).add_to(m)

        # routes
        for tech_name, latlon_list in routes_for_map:
            poly = [(lat, lon) for (lat, lon) in latlon_list]
            folium.PolyLine(poly, tooltip=f"Tournée {tech_name}").add_to(m)

        st_folium(m, use_container_width=True, height=520)
