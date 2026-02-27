import time
import math
import requests
import pandas as pd
import streamlit as st
from dataclasses import dataclass
from typing import List, Optional, Tuple

from ortools.constraint_solver import pywrapcp, routing_enums_pb2

# -----------------------------
# Config UI
# -----------------------------
st.set_page_config(page_title="Optimisation tournée technicien", layout="wide")
st.title("Optimisation de tournée (OSM) — avec calcul d’économies")

st.caption(
    "Saisis une agence (départ/retour) + des interventions (nom, adresse, durée). "
    "L’app compare le temps **Avant** (ordre saisi) vs **Après** (ordre optimisé) et explique le gain."
)

# -----------------------------
# Modèles
# -----------------------------
@dataclass
class Stop:
    name: str
    address_input: str
    address_resolved: str
    lat: float
    lon: float
    service_minutes: int = 0

# -----------------------------
# Géocodage OSM (Nominatim)
# -----------------------------
NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"

@st.cache_data(show_spinner=False, ttl=24 * 3600)
def geocode_address_osm(address: str) -> Optional[Tuple[float, float, str]]:
    """
    Retourne (lat, lon, display_name) via Nominatim.
    """
    if not address or not address.strip():
        return None

    headers = {
        "User-Agent": "tournee-optimizer-streamlit/1.0 (contact: internal)",
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
# Matrice de temps OSRM
# -----------------------------
OSRM_TABLE_URL = "https://router.project-osrm.org/table/v1/driving/"

@st.cache_data(show_spinner=False, ttl=6 * 3600)
def osrm_table_minutes(coords: List[Tuple[float, float]]) -> List[List[int]]:
    """
    coords: liste de (lat, lon)
    Retourne matrice [i][j] en minutes.
    """
    if len(coords) < 2:
        return [[0]]

    coord_str = ";".join([f"{lon},{lat}" for (lat, lon) in coords])
    url = OSRM_TABLE_URL + coord_str
    r = requests.get(url, params={"annotations": "duration"}, timeout=30)
    r.raise_for_status()
    durations = r.json()["durations"]  # secondes
    return [[int(round((d or 0) / 60.0)) for d in row] for row in durations]

# -----------------------------
# OR-Tools optimisation
# -----------------------------
def solve_route(
    time_matrix_minutes: List[List[int]],
    service_minutes: List[int],
    start_index: int,
    end_index: int,
    time_limit_s: int = 5,
) -> Optional[Tuple[List[int], int]]:
    """
    Minimise (trajet + service à l’arrivée).
    Retourne (route_indices, total_minutes).
    """
    n = len(time_matrix_minutes)
    if n == 0:
        return None
    if n == 1:
        return [0], service_minutes[0]

    manager = pywrapcp.RoutingIndexManager(n, 1, [start_index], [end_index])
    routing = pywrapcp.RoutingModel(manager)

    def transit(from_i: int, to_i: int) -> int:
        f = manager.IndexToNode(from_i)
        t = manager.IndexToNode(to_i)
        return time_matrix_minutes[f][t] + service_minutes[t]

    cb = routing.RegisterTransitCallback(transit)
    routing.SetArcCostEvaluatorOfAllVehicles(cb)

    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    params.time_limit.FromSeconds(time_limit_s)

    sol = routing.SolveWithParameters(params)
    if sol is None:
        return None

    route = []
    idx = routing.Start(0)
    total = 0
    while not routing.IsEnd(idx):
        node = manager.IndexToNode(idx)
        route.append(node)
        nxt = sol.Value(routing.NextVar(idx))
        nxt_node = manager.IndexToNode(nxt)
        if not routing.IsEnd(nxt):
            total += time_matrix_minutes[node][nxt_node] + service_minutes[nxt_node]
        idx = nxt
    route.append(manager.IndexToNode(idx))
    return route, total

# -----------------------------
# Calcul "Avant" (ordre saisi)
# -----------------------------
def compute_baseline_total(route_idx: List[int], tm: List[List[int]], service: List[int]) -> Tuple[int, int, int]:
    """
    Calcule:
      total = sum(trajets) + sum(service à l’arrivée) pour une route donnée.
    Retourne (total, travel_total, service_total)
    """
    travel_total = 0
    service_total = 0
    for a, b in zip(route_idx[:-1], route_idx[1:]):
        travel_total += tm[a][b]
        service_total += service[b]
    return travel_total + service_total, travel_total, service_total

def explain_savings(before_route_idx: List[int], after_route_idx: List[int], tm: List[List[int]], stops: List[Stop]) -> str:
    """
    Explication simple et concrète du gain :
    - distance "ligne droite" remplacée par trajets plus cohérents
    - moins d’allers-retours / zigzag
    On quantifie un indicateur : nombre de "grands sauts" (trajets longs) réduit.
    """
    def leg_minutes(route_idx):
        return [tm[a][b] for a, b in zip(route_idx[:-1], route_idx[1:])]

    before_legs = leg_minutes(before_route_idx)
    after_legs = leg_minutes(after_route_idx)

    if not before_legs or not after_legs:
        return "Gain expliqué : la route optimisée réduit les détours et regroupe les points proches."

    # seuil "grand saut": top 25% des trajets avant (si possible)
    sorted_before = sorted(before_legs)
    threshold = sorted_before[int(max(0, math.floor(0.75 * (len(sorted_before)-1))))]  # approx quantile 75%
    big_before = sum(1 for x in before_legs if x >= threshold)
    big_after = sum(1 for x in after_legs if x >= threshold)

    # Variation zigzag : somme des “pics” (legs très au-dessus de la médiane)
    med = sorted_before[len(sorted_before)//2]
    spikes_before = sum(max(0, x - med) for x in before_legs)
    spikes_after = sum(max(0, x - med) for x in after_legs)

    pieces = []
    pieces.append(
        "Pourquoi tu économises du temps : l’optimisation change l’ordre des visites pour limiter les détours. "
        "Au lieu de ‘zigzaguer’ entre des zones éloignées, le solveur regroupe les interventions proches géographiquement."
    )

    if big_after < big_before:
        pieces.append(
            f"Concrètement, la tournée optimisée réduit les ‘grands sauts’ (trajets longs) : "
            f"{big_before} → {big_after}. Ça évite des allers-retours inutiles."
        )
    else:
        pieces.append(
            "Concrètement, la tournée optimisée cherche à réduire les trajets les plus pénalisants en temps "
            "(ceux qui créent les plus gros détours)."
        )

    if spikes_after < spikes_before:
        pieces.append(
            f"On voit aussi moins de ‘pics de trajet’ (détours) : l’excès par rapport à un trajet typique diminue "
            f"({spikes_before} → {spikes_after} min cumulés)."
        )

    pieces.append(
        "Important : le **temps d’intervention sur site** ne change pas (tu fais les mêmes jobs). "
        "Le gain vient quasi exclusivement du **temps de déplacement**."
    )

    return "\n\n".join(pieces)

# -----------------------------
# UI
# -----------------------------
colA, colB = st.columns([1, 2])

with colA:
    st.subheader("1) Agence (départ / retour)")
    agency_name = st.text_input("Nom agence", value="Agence")
    agency_address = st.text_input("Adresse agence", value="")

    st.subheader("2) Interventions")
    st.write("Ajoute des lignes, puis remplis Nom + Adresse + Durée (min). L’ordre des lignes = l’ordre ‘Avant’.")

    if "jobs_df" not in st.session_state:
        st.session_state.jobs_df = pd.DataFrame(
            [
                {"Nom": "Intervention 1", "Adresse": "", "Durée (min)": 30},
                {"Nom": "Intervention 2", "Adresse": "", "Durée (min)": 45},
            ]
        )

    edited_df = st.data_editor(
        st.session_state.jobs_df,
        num_rows="dynamic",
        use_container_width=True,
        hide_index=True,
        column_config={
            "Nom": st.column_config.TextColumn(required=True),
            "Adresse": st.column_config.TextColumn(required=True),
            "Durée (min)": st.column_config.NumberColumn(min_value=0, step=5, required=True),
        },
    )
    st.session_state.jobs_df = edited_df

    c1, c2 = st.columns(2)
    with c1:
        return_to_agency = st.checkbox("Retour à l'agence en fin de tournée", value=True)
    with c2:
        time_limit = st.slider("Temps de calcul max (s)", 1, 15, 5)

    optimize = st.button("🚀 Optimiser + calculer économies", type="primary")

with colB:
    st.subheader("Résultat")
    if not optimize:
        st.info("Renseigne l'agence + au moins 1 intervention, puis clique sur **Optimiser + calculer économies**.")
        st.stop()

    # Validation
    jobs_df = st.session_state.jobs_df.copy().dropna(subset=["Nom", "Adresse", "Durée (min)"])
    jobs_df["Nom"] = jobs_df["Nom"].astype(str)
    jobs_df["Adresse"] = jobs_df["Adresse"].astype(str)

    if not agency_address.strip():
        st.error("Adresse agence manquante.")
        st.stop()
    if len(jobs_df) == 0:
        st.error("Ajoute au moins une intervention.")
        st.stop()

    # Géocodage agence
    with st.spinner("Géocodage agence (OSM) ..."):
        g = geocode_address_osm(agency_address.strip())
    if g is None:
        st.error("Impossible de géocoder l'adresse de l'agence (OSM). Ajoute ville + code postal + France.")
        st.stop()

    a_lat, a_lon, a_disp = g
    agency_stop = Stop(
        name=f"{agency_name} (Départ)",
        address_input=agency_address.strip(),
        address_resolved=a_disp,
        lat=a_lat,
        lon=a_lon,
        service_minutes=0,
    )

    # Géocodage interventions
    stops_jobs: List[Stop] = []
    bad = []
    with st.spinner("Géocodage interventions (OSM) ..."):
        for i, row in jobs_df.iterrows():
            name = row["Nom"].strip()
            addr = row["Adresse"].strip()
            dur = int(row["Durée (min)"])

            gg = geocode_address_osm(addr)
            if gg is None:
                bad.append((name, addr))
                continue
            lat, lon, disp = gg
            stops_jobs.append(Stop(name=name, address_input=addr, address_resolved=disp, lat=lat, lon=lon, service_minutes=dur))
            time.sleep(0.1)  # doux pour Nominatim

    if bad:
        st.warning("Certaines adresses n’ont pas été géocodées (OSM). Ajoute ville + CP + France.")
        st.write(pd.DataFrame(bad, columns=["Nom", "Adresse"]))

    if len(stops_jobs) == 0:
        st.error("Aucune intervention géocodée correctement.")
        st.stop()

    # Construire stops (avec end agence distinct si retour)
    if return_to_agency:
        agency_end = Stop(
            name=f"{agency_name} (Retour)",
            address_input=agency_address.strip(),
            address_resolved=a_disp,
            lat=a_lat,
            lon=a_lon,
            service_minutes=0,
        )
        stops_all = [agency_stop] + stops_jobs + [agency_end]
        start_index = 0
        end_index = len(stops_all) - 1
    else:
        st.warning("Sans retour agence non implémenté ici (je peux te l’ajouter). On force le retour pour le calcul.")
        agency_end = Stop(
            name=f"{agency_name} (Retour)",
            address_input=agency_address.strip(),
            address_resolved=a_disp,
            lat=a_lat,
            lon=a_lon,
            service_minutes=0,
        )
        stops_all = [agency_stop] + stops_jobs + [agency_end]
        start_index = 0
        end_index = len(stops_all) - 1

    coords = [(s.lat, s.lon) for s in stops_all]
    service = [s.service_minutes for s in stops_all]

    # Matrice OSRM
    with st.spinner("Calcul des temps de trajet (OSRM) ..."):
        tm = osrm_table_minutes(coords)

    # -----------------------------
    # AVANT (ordre saisi)
    #   start -> jobs dans l'ordre -> end
    # -----------------------------
    baseline_route_idx = [start_index] + list(range(1, 1 + len(stops_jobs))) + [end_index]
    before_total, before_travel, before_service = compute_baseline_total(baseline_route_idx, tm, service)

    # -----------------------------
    # APRÈS (optimisé)
    # -----------------------------
    with st.spinner("Optimisation (OR-Tools) ..."):
        res = solve_route(tm, service, start_index, end_index, time_limit_s=time_limit)

    if res is None:
        st.error("Aucune solution trouvée.")
        st.stop()

    after_route_idx, after_total = res
    after_total2, after_travel, after_service = compute_baseline_total(after_route_idx, tm, service)

    # (after_total et after_total2 devraient matcher ; on garde after_total2 comme contrôle)
    after_total = after_total2

    # -----------------------------
    # Gains
    # -----------------------------
    gain_total = before_total - after_total
    gain_travel = before_travel - after_travel  # c'est ça qui explique le gain
    gain_pct = (gain_total / before_total * 100.0) if before_total > 0 else 0.0

    # Affichage synthèse
    c1, c2, c3 = st.columns(3)
    c1.metric("Avant (total)", f"{before_total} min", help="Trajet + interventions, dans l’ordre saisi")
    c2.metric("Après (total)", f"{after_total} min", help="Trajet + interventions, ordre optimisé")
    c3.metric("Économie", f"{gain_total} min", f"{gain_pct:.1f} %")

    # Détails trajet vs service (pour prouver le 'pourquoi')
    st.markdown("### Détail Avant vs Après")
    detail = pd.DataFrame(
        [
            {"": "Trajet", "Avant (min)": before_travel, "Après (min)": after_travel, "Gain (min)": gain_travel},
            {"": "Interventions", "Avant (min)": before_service, "Après (min)": after_service, "Gain (min)": before_service - after_service},
            {"": "TOTAL", "Avant (min)": before_total, "Après (min)": after_total, "Gain (min)": gain_total},
        ]
    )
    st.dataframe(detail, use_container_width=True, hide_index=True)

    # Afficher les routes
    st.markdown("### Ordre des visites")

    def route_table(route_idx: List[int], label: str) -> pd.DataFrame:
        ordered = [stops_all[i] for i in route_idx]
        return pd.DataFrame({
            "Ordre": list(range(1, len(ordered) + 1)),
            "Point": [s.name for s in ordered],
            "Adresse (résolue OSM)": [s.address_resolved for s in ordered],
            "Durée intervention (min)": [s.service_minutes for s in ordered],
        })

    tab1, tab2 = st.tabs(["Avant (ordre saisi)", "Après (optimisé)"])
    with tab1:
        st.dataframe(route_table(baseline_route_idx, "Avant"), use_container_width=True, hide_index=True)
    with tab2:
        st.dataframe(route_table(after_route_idx, "Après"), use_container_width=True, hide_index=True)

    # Explication concrète du gain
    st.markdown("### Pourquoi tu économises ce temps (explication concrète)")
    explanation = explain_savings(baseline_route_idx, after_route_idx, tm, stops_all)
    st.write(explanation)

    # Message très concret métier
    if gain_total > 0:
        st.success(
            f"Conclusion : tu économises **{gain_total} min** principalement grâce à **{gain_travel} min** de trajets en moins. "
            "Le temps sur site ne bouge pas : tu fais les mêmes interventions, mais dans un ordre plus logique."
        )
    elif gain_total == 0:
        st.info("Conclusion : pas de gain mesurable sur ce jeu de données (ordre déjà proche de l’optimal).")
    else:
        st.warning("Conclusion : l’ordre saisi semble déjà meilleur que la solution trouvée (rare). Augmente le temps de calcul.")
        
