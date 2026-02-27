import time
import math
import requests
import pandas as pd
import streamlit as st
from dataclasses import dataclass
from typing import List, Optional, Tuple, Set, Dict
from datetime import datetime, time as dtime, timedelta
from zoneinfo import ZoneInfo

from ortools.constraint_solver import pywrapcp, routing_enums_pb2

import folium
from streamlit_folium import st_folium


# =========================================================
# UI
# =========================================================
st.set_page_config(page_title="Optimisation tournées multi-tech (OSM/OSRM)", layout="wide")
st.title("Optimisation tournées multi-techniciens (OSM/OSRM)")
st.caption(
    "Affectation + ordre de visite. Contraintes: heures max/tech, skills (menus), SLA, HO/HNO (horaires ouvrés), "
    "validation des adresses, carte. Sans Google: OSM (Nominatim) + OSRM."
)


# =========================================================
# Models
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
    deadline_min: Optional[int]   # None = pas de SLA
    priority: int                # 1..5
    ho_mode: str                 # "HO", "HNO", "INDIFF"


# =========================================================
# Helpers
# =========================================================
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

def safe_str(x) -> str:
    if x is None:
        return ""
    s = str(x)
    return "" if s.lower() == "nan" else s

def minutes_since_day_start(dt: datetime) -> int:
    return dt.hour * 60 + dt.minute

def parse_hhmm(s: str) -> dtime:
    # s "08:00"
    hh, mm = s.split(":")
    return dtime(hour=int(hh), minute=int(mm))


# =========================================================
# Geocoding OSM (Nominatim)
# =========================================================
NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"

@st.cache_data(show_spinner=False, ttl=24 * 3600)
def geocode_osm(address: str) -> Optional[Tuple[float, float, str]]:
    """Retourne (lat, lon, display_name) ou None."""
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
    """coords: (lat, lon) => matrice minutes."""
    if len(coords) < 2:
        return [[0]]
    coord_str = ";".join([f"{lon},{lat}" for (lat, lon) in coords])
    url = OSRM_TABLE_URL + coord_str
    r = requests.get(url, params={"annotations": "duration"}, timeout=30)
    r.raise_for_status()
    durations = r.json()["durations"]  # seconds
    return [[int(round((d or 0) / 60.0)) for d in row] for row in durations]


# =========================================================
# HO/HNO -> Time windows
# =========================================================
def compute_job_time_window(
    ho_mode: str,
    start_min: int,
    open_start_min: int,
    open_end_min: int,
    horizon_min: int,
) -> Optional[Tuple[int, int]]:
    """
    Renvoie une fenêtre [min,max] en minutes depuis start de journée.
    - start_min: heure de départ (minute du jour)
    - horizon_min: durée max de la journée de planif à partir de 00:00 (ex: 24h => 1440)

    Note: OR-Tools n'accepte qu'UNE fenêtre par job.
    Donc pour HNO on prend le segment pertinent selon l'heure actuelle.
    """
    ho_mode = ho_mode.upper().strip()

    if ho_mode == "INDIFF":
        return (start_min, horizon_min)

    if ho_mode == "HO":
        # Si on est après la fermeture -> plus faisable aujourd'hui (fenêtre vide)
        if start_min >= open_end_min:
            return None
        # Si on est avant ouverture -> on attend ouverture
        earliest = max(start_min, open_start_min)
        latest = open_end_min
        if earliest > latest:
            return None
        return (earliest, latest)

    if ho_mode == "HNO":
        # HNO = hors horaires ouvrés.
        # Cas 1: on est avant ouverture => HNO faisable maintenant jusqu'à ouverture
        if start_min < open_start_min:
            return (start_min, open_start_min)
        # Cas 2: on est pendant ouverture => prochain créneau HNO = après fermeture
        if open_start_min <= start_min < open_end_min:
            return (open_end_min, horizon_min)
        # Cas 3: on est après fermeture => HNO faisable maintenant jusqu'à horizon
        return (start_min, horizon_min)

    # fallback
    return (start_min, horizon_min)


# =========================================================
# OR-Tools Solver
# =========================================================
def solve_vrp_multi_tech(
    tm_min: List[List[int]],
    service_min: List[int],
    techs: List[Tech],
    starts: List[int],
    ends: List[int],
    job_nodes: List[int],
    job_time_windows: Dict[int, Tuple[int, int]],     # node -> (a,b)
    job_deadlines: Dict[int, int],                    # node -> deadline
    allowed_vehicles: Dict[int, List[int]],           # node -> list vehicles
    priority_by_node: Dict[int, int],
    allow_drop: bool,
    time_limit_s: int,
):
    n = len(tm_min)
    m = len(techs)

    manager = pywrapcp.RoutingIndexManager(n, m, starts, ends)
    routing = pywrapcp.RoutingModel(manager)

    # Objective: travel time only
    def travel_cb(from_i, to_i):
        f = manager.IndexToNode(from_i)
        t = manager.IndexToNode(to_i)
        return int(tm_min[f][t])

    travel_idx = routing.RegisterTransitCallback(travel_cb)
    routing.SetArcCostEvaluatorOfAllVehicles(travel_idx)

    # Time dimension: travel + service at FROM
    def time_cb(from_i, to_i):
        f = manager.IndexToNode(from_i)
        t = manager.IndexToNode(to_i)
        return int(tm_min[f][t] + service_min[f])

    time_idx = routing.RegisterTransitCallback(time_cb)
    routing.AddDimension(time_idx, 0, 10**9, True, "Time")
    time_dim = routing.GetDimensionOrDie("Time")

    # Max work time per tech (hard)
    for v, tech in enumerate(techs):
        time_dim.CumulVar(routing.End(v)).SetMax(int(tech.max_minutes))

    # Apply time windows (HO/HNO + heure actuelle)
    for node, (a, b) in job_time_windows.items():
        node = int(node)
        idx = manager.NodeToIndex(node)
        time_dim.CumulVar(idx).SetRange(int(a), int(b))

    # Apply SLA deadlines as tighter window if present
    for node, dl in job_deadlines.items():
        node = int(node)
        idx = manager.NodeToIndex(node)
        # intersect existing [0..dl]
        time_dim.CumulVar(idx).SetRange(0, int(dl))

    # Skills eligibility (robust): VehicleVar(idx).SetValues(...)
    for node, vehs in allowed_vehicles.items():
        node = int(node)
        idx = int(manager.NodeToIndex(node))
        if routing.IsStart(idx) or routing.IsEnd(idx):
            continue
        vehs_clean = sorted({int(v) for v in vehs})
        routing.VehicleVar(idx).SetValues(vehs_clean)

    # Optional drop (report) with penalty based on priority
    if allow_drop:
        for node in job_nodes:
            node = int(node)
            idx = manager.NodeToIndex(node)
            prio = int(priority_by_node.get(node, 3))
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
        routes.append(
            {"vehicle": v, "tech": tech.name, "nodes": nodes, "end_time_min": int(end_time), "max_min": int(tech.max_minutes)}
        )

    return {"routes": routes, "dropped": dropped}


# =========================================================
# Session state init
# =========================================================
if "skills_df" not in st.session_state:
    st.session_state.skills_df = pd.DataFrame([{"Skill": "elec_b1v"}, {"Skill": "travail_en_hauteur"}])

if "techs_df" not in st.session_state:
    st.session_state.techs_df = pd.DataFrame(
        [
            {"Tech": "Tech A", "Adresse départ": "", "Heures max": 7.5},
            {"Tech": "Tech B", "Adresse départ": "", "Heures max": 8.0},
        ]
    )

if "jobs_df" not in st.session_state:
    st.session_state.jobs_df = pd.DataFrame(
        [
            {"Intervention": "Job 1", "Adresse": "", "Durée (min)": 30, "HO/HNO": "HO", "SLA (h) optionnel": 4.0, "Priorité (1-5)": 5},
            {"Intervention": "Job 2", "Adresse": "", "Durée (min)": 45, "HO/HNO": "INDIFF", "SLA (h) optionnel": "", "Priorité (1-5)": 3},
        ]
    )

if "tech_skills" not in st.session_state:
    st.session_state.tech_skills = {}  # tech_name -> list[str]
if "job_skills" not in st.session_state:
    st.session_state.job_skills = {}   # job_name  -> list[str]


# =========================================================
# Controls (time / HO window / options)
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

    horizon_hours = st.slider("Horizon de planification (heures)", 4, 24, 12)
    allow_drop = st.checkbox("Autoriser le report (drop) si impossible", value=False)
    time_limit = st.slider("Temps de calcul max (s)", 5, 60, 15)

    st.caption("HO/HNO est calculé à partir de l’heure de départ ci-dessus.")


# =========================================================
# Layout
# =========================================================
left, right = st.columns([1.25, 1.75])

with left:
    st.subheader("0) Référentiel skills (menus déroulants)")
    st.write("Ajoute ici les skills autorisés. Ils seront proposés en menus déroulants, sans saisie libre.")
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
    st.subheader("1) Techniciens")
    techs_df = st.data_editor(
        st.session_state.techs_df,
        num_rows="dynamic",
        use_container_width=True,
        hide_index=True,
        column_config={
            "Tech": st.column_config.TextColumn(required=True),
            "Adresse départ": st.column_config.TextColumn(required=True),
            "Heures max": st.column_config.NumberColumn(min_value=0.0, step=0.5, required=True),
        },
    )
    st.session_state.techs_df = techs_df

    if skills_list:
        st.markdown("**Skills par technicien (menus)**")
        for _, row in techs_df.iterrows():
            tech_name = safe_str(row.get("Tech", "")).strip()
            if not tech_name:
                continue
            default = st.session_state.tech_skills.get(tech_name, [])
            chosen = st.multiselect(
                f"Skills — {tech_name}",
                options=skills_list,
                default=[s for s in default if s in skills_list],
                key=f"tech_skills__{tech_name}",
            )
            st.session_state.tech_skills[tech_name] = chosen
    else:
        st.warning("Ajoute au moins 1 skill dans le référentiel pour activer les menus.")

    st.divider()
    st.subheader("2) Interventions")
    jobs_df = st.data_editor(
        st.session_state.jobs_df,
        num_rows="dynamic",
        use_container_width=True,
        hide_index=True,
        column_config={
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
            job_name = safe_str(row.get("Intervention", "")).strip()
            if not job_name:
                continue
            default = st.session_state.job_skills.get(job_name, [])
            chosen = st.multiselect(
                f"Skills requis — {job_name}",
                options=skills_list,
                default=[s for s in default if s in skills_list],
                key=f"job_skills__{job_name}",
            )
            st.session_state.job_skills[job_name] = chosen

    st.divider()
    run = st.button("🚀 Optimiser", type="primary")


with right:
    st.subheader("Résultats")

    # Parse time params (robust)
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

    # Prepare data
    techs_df = st.session_state.techs_df.copy().dropna(subset=["Tech", "Adresse départ", "Heures max"])
    jobs_df = st.session_state.jobs_df.copy().dropna(subset=["Intervention", "Adresse", "Durée (min)", "HO/HNO", "Priorité (1-5)"])

    if len(techs_df) == 0:
        st.error("Ajoute au moins 1 technicien.")
        st.stop()
    if len(jobs_df) == 0:
        st.error("Ajoute au moins 1 intervention.")
        st.stop()

    # -----------------------------------------------------
    # 1) Validation / Geocoding with clear messages
    # -----------------------------------------------------
    tech_rows = []
    bad_tech = []
    with st.spinner("Validation adresses techniciens (géocodage OSM)…"):
        for _, row in techs_df.iterrows():
            name = safe_str(row["Tech"]).strip()
            addr = safe_str(row["Adresse départ"]).strip()
            max_h = float(row["Heures max"])
            skills = set(st.session_state.tech_skills.get(name, []))

            g = None
            try:
                g = geocode_osm(addr)
            except Exception:
                g = None

            if g is None:
                bad_tech.append({"Type": "Technicien", "Nom": name, "Adresse": addr, "Erreur": "Adresse non reconnue. Ajoute: n°, rue, CP, ville, France."})
                tech_rows.append({"Tech": name, "Adresse saisie": addr, "Statut": "❌ invalide", "Conseil": "Ajoute CP + ville + France"})
                continue

            lat, lon, disp = g
            tech_rows.append({"Tech": name, "Adresse saisie": addr, "Statut": "✅ OK", "Adresse résolue": disp})

            tech_rows[-1]["_lat"] = lat
            tech_rows[-1]["_lon"] = lon
            tech_rows[-1]["_max_min"] = hours_to_minutes(max_h)
            tech_rows[-1]["_skills"] = skills

            time.sleep(0.08)

    job_rows = []
    bad_jobs = []
    with st.spinner("Validation adresses interventions (géocodage OSM)…"):
        for _, row in jobs_df.iterrows():
            name = safe_str(row["Intervention"]).strip()
            addr = safe_str(row["Adresse"]).strip()
            dur = int(row["Durée (min)"])
            ho_mode = safe_str(row["HO/HNO"]).strip().upper()
            prio = int(row["Priorité (1-5)"])
            sla_h = to_float_or_none(row.get("SLA (h) optionnel", None))
            deadline = hours_to_minutes(sla_h) if sla_h is not None else None
            req_skills = set(st.session_state.job_skills.get(name, []))

            g = None
            try:
                g = geocode_osm(addr)
            except Exception:
                g = None

            if g is None:
                bad_jobs.append({"Type": "Intervention", "Nom": name, "Adresse": addr, "Erreur": "Adresse non reconnue. Ajoute: n°, rue, CP, ville, France."})
                job_rows.append({"Intervention": name, "Adresse saisie": addr, "Statut": "❌ invalide", "Conseil": "Ajoute CP + ville + France"})
                continue

            lat, lon, disp = g
            job_rows.append({"Intervention": name, "Adresse saisie": addr, "Statut": "✅ OK", "Adresse résolue": disp})

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
        st.error("Certaines adresses sont invalides. Corrige-les puis relance l’optimisation.")
        if bad_tech:
            st.dataframe(pd.DataFrame(bad_tech), use_container_width=True, hide_index=True)
        if bad_jobs:
            st.dataframe(pd.DataFrame(bad_jobs), use_container_width=True, hide_index=True)
        st.stop()

    # -----------------------------------------------------
    # 2) Build tech/job objects
    # -----------------------------------------------------
    techs: List[Tech] = []
    for r in tech_rows:
        techs.append(
            Tech(
                name=r["Tech"],
                address_input=r["Adresse saisie"],
                address_resolved=r["Adresse résolue"],
                lat=r["_lat"],
                lon=r["_lon"],
                max_minutes=r["_max_min"],
                skills=r["_skills"],
            )
        )

    jobs: List[Job] = []
    for r in job_rows:
        jobs.append(
            Job(
                name=r["Intervention"],
                address_input=r["Adresse saisie"],
                address_resolved=r["Adresse résolue"],
                lat=r["_lat"],
                lon=r["_lon"],
                service_min=r["_dur"],
                required_skills=r["_req_skills"],
                deadline_min=r["_deadline"],
                priority=r["_prio"],
                ho_mode=r["_ho"],
            )
        )

    # -----------------------------------------------------
    # 3) Build nodes: start per tech + end per tech + jobs
    # -----------------------------------------------------
    coords: List[Tuple[float, float]] = []
    service_min: List[int] = []
    starts: List[int] = []
    ends: List[int] = []

    # starts
    for t in techs:
        starts.append(len(coords))
        coords.append((t.lat, t.lon))
        service_min.append(0)

    # ends (duplicate)
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

    # -----------------------------------------------------
    # 4) Build constraints: skills eligibility + time windows (HO/HNO) + SLA
    # -----------------------------------------------------
    allowed_vehicles: Dict[int, List[int]] = {}
    job_time_windows: Dict[int, Tuple[int, int]] = {}
    job_deadlines: Dict[int, int] = {}
    priority_by_node: Dict[int, int] = {}

    # Time window computed using current/start time and HO hours
    for node, job in zip(job_nodes, jobs):
        # HO/HNO window
        tw = compute_job_time_window(
            ho_mode=job.ho_mode,
            start_min=start_min,
            open_start_min=open_start_min,
            open_end_min=open_end_min,
            horizon_min=horizon_min,
        )
        if tw is None:
            # not feasible today -> either drop or block
            if allow_drop:
                # we still add a wide window so it can be dropped, but not feasible window means it can never be served
                # simplest: force it to be droppable by not adding a restrictive window
                job_time_windows[int(node)] = (start_min, horizon_min)
            else:
                st.error(f"'{job.name}' est impossible à planifier aujourd’hui avec HO/HNO et l’heure de départ. Active 'report' ou change l’heure.")
                st.stop()
        else:
            job_time_windows[int(node)] = (int(tw[0]), int(tw[1]))

        # SLA deadline if provided (tighten)
        if job.deadline_min is not None:
            job_deadlines[int(node)] = int(job.deadline_min)

        # priority map
        priority_by_node[int(node)] = int(job.priority)

        # skills eligibility
        ok = []
        for v, tech in enumerate(techs):
            if job.required_skills.issubset(tech.skills):
                ok.append(v)

        if not ok:
            st.error(
                f"Aucun technicien compatible pour '{job.name}'. "
                f"Skills requis={sorted(job.required_skills)}. "
                f"➡️ Corrige via les menus skills."
            )
            st.stop()

        allowed_vehicles[int(node)] = [int(v) for v in ok]

    # -----------------------------------------------------
    # 5) Time matrix
    # -----------------------------------------------------
    with st.spinner("Calcul des temps de trajet (OSRM)…"):
        tm = osrm_table_minutes(coords)

    # -----------------------------------------------------
    # 6) Solve
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
        )

    if sol is None:
        st.error("Aucune solution faisable. ➡️ augmente heures max / assouplis SLA / ajoute un tech / active report.")
        st.stop()

    dropped = sol["dropped"]
    if dropped:
        dropped_names = [jobs[int(n) - job_offset].name for n in dropped]
        st.warning("Interventions reportées : " + ", ".join(dropped_names))

    # -----------------------------------------------------
    # 7) Display routes
    # -----------------------------------------------------
    st.success("Solution trouvée ✅")
    st.markdown("### Tournées par technicien")

    def label(node: int) -> str:
        node = int(node)
        if node < len(techs):
            return f"Départ {techs[node].name}"
        if len(techs) <= node < 2 * len(techs):
            return f"Retour {techs[node - len(techs)].name}"
        return jobs[node - job_offset].name

    assign_rows = []
    routes_for_map = []  # list of list[(lat,lon)] per tech

    for r in sol["routes"]:
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
                    "Intervention": jb.name,
                    "Technicien": tech_name,
                    "HO/HNO": jb.ho_mode,
                    "Priorité": jb.priority,
                    "SLA(h)": "" if jb.deadline_min is None else round(jb.deadline_min / 60.0, 2),
                    "Skills requis": ", ".join(sorted(jb.required_skills)),
                })

        routes_for_map.append((tech_name, route_latlon))

        with st.expander(f"{tech_name} — {len(route_jobs)} interventions — fin {end_time} min (max {max_min} min)", expanded=True):
            if not route_jobs:
                st.write("Aucune intervention affectée.")
            else:
                st.dataframe(pd.DataFrame({
                    "Ordre": list(range(1, len(route_jobs)+1)),
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
    # 8) Map
    # -----------------------------------------------------
    st.markdown("### Carte (points + tournées)")

    # Center map
    all_lat = [c[0] for c in coords]
    all_lon = [c[1] for c in coords]
    center = (sum(all_lat)/len(all_lat), sum(all_lon)/len(all_lon))

    m = folium.Map(location=center, zoom_start=11, control_scale=True)

    # Markers - tech starts
    for i, t in enumerate(techs):
        folium.Marker(
            location=(t.lat, t.lon),
            tooltip=f"Départ {t.name}",
            icon=folium.Icon(icon="home", prefix="fa"),
        ).add_to(m)

    # Markers - jobs
    for j in jobs:
        folium.CircleMarker(
            location=(j.lat, j.lon),
            radius=6,
            tooltip=f"{j.name} | {j.ho_mode} | prio {j.priority}",
            fill=True,
        ).add_to(m)

    # Draw polylines (simple connect)
    # (On ne récupère pas la géométrie route OSRM, mais c’est déjà très utile pour visualiser.)
    for tech_name, latlon_list in routes_for_map:
        # latlon_list contains (lat,lon) from coords list
        poly = [(lat, lon) for (lat, lon) in latlon_list]
        folium.PolyLine(poly, tooltip=f"Tournée {tech_name}").add_to(m)

    st_folium(m, use_container_width=True, height=520)
