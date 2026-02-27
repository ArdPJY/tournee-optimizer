# app.py
# Streamlit app: saisie d'interventions (nom + adresse + durée) + optimisation tournée (OR-Tools)
# - Géocodage: Nominatim (OpenStreetMap) gratuit (sans clé) -> lat/lon
# - Temps de trajet: OSRM public gratuit -> matrice de durées
#
# ⚠️ Notes:
# - Nominatim impose des limites (ne pas spammer). On met du cache + un User-Agent.
# - OSRM public est pratique en POC, pas recommandé en prod (self-host si besoin).
#
# Lancer:
#   pip install -r requirements.txt
#   streamlit run app.py

import time
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
st.title("Optimisation de tournée (Technicien / Agence)")

st.caption(
    "Saisis une agence (départ/retour), puis tes interventions. "
    "Le bouton **Optimiser** calcule l'ordre optimal (temps trajet + temps d'intervention)."
)


# -----------------------------
# Modèles
# -----------------------------
@dataclass
class Stop:
    name: str
    address: str
    lat: float
    lon: float
    service_minutes: int = 0


# -----------------------------
# Géocodage (Nominatim)
# -----------------------------
NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"

@st.cache_data(show_spinner=False, ttl=24 * 3600)
def geocode_address(address: str) -> Optional[Tuple[float, float, str]]:
    """
    Retourne (lat, lon, display_name) via Nominatim.
    Cache 24h pour éviter de re-taper l'API.
    """
    if not address or not address.strip():
        return None

    headers = {
        # Important pour Nominatim
        "User-Agent": "tournee-optimizer-streamlit/1.0 (contact: internal)",
        "Accept-Language": "fr",
    }
    params = {
        "q": address,
        "format": "json",
        "limit": 1,
    }
    r = requests.get(NOMINATIM_URL, params=params, headers=headers, timeout=15)
    r.raise_for_status()
    data = r.json()
    if not data:
        return None
    lat = float(data[0]["lat"])
    lon = float(data[0]["lon"])
    disp = data[0].get("display_name", address)
    return (lat, lon, disp)


# -----------------------------
# Matrice de temps (OSRM)
# -----------------------------
OSRM_TABLE_URL = "https://router.project-osrm.org/table/v1/driving/"

@st.cache_data(show_spinner=False, ttl=6 * 3600)
def osrm_table_minutes(coords: List[Tuple[float, float]]) -> List[List[int]]:
    """
    coords: liste de (lat, lon)
    Retourne matrice [i][j] en minutes.
    Cache 6h (POC).
    """
    if len(coords) < 2:
        return [[0]]

    # OSRM attend lon,lat
    coord_str = ";".join([f"{lon},{lat}" for (lat, lon) in coords])
    url = OSRM_TABLE_URL + coord_str
    r = requests.get(url, params={"annotations": "duration"}, timeout=20)
    r.raise_for_status()
    durations = r.json()["durations"]  # secondes
    return [[int(round((d or 0) / 60.0)) for d in row] for row in durations]


# -----------------------------
# Optimisation OR-Tools
# -----------------------------
def solve_route(
    time_matrix_minutes: List[List[int]],
    service_minutes: List[int],
    start_index: int,
    end_index: int,
    time_limit_s: int = 5,
) -> Optional[Tuple[List[int], int]]:
    """
    Minimise (temps trajet + temps service à l'arrivée).
    Retour: (route_indices, total_minutes)
    """
    n = len(time_matrix_minutes)
    if n == 0:
        return None
    if n == 1:
        return ([0], service_minutes[0])

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
# UI - Saisie des données
# -----------------------------
colA, colB = st.columns([1, 2])

with colA:
    st.subheader("1) Agence (départ / retour)")
    agency_name = st.text_input("Nom agence", value="Agence")
    agency_address = st.text_input("Adresse agence", value="")

    st.subheader("2) Interventions")
    st.write("Ajoute des lignes, puis remplis Nom + Adresse + Durée (min).")

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

    options = st.columns(2)
    with options[0]:
        return_to_agency = st.checkbox("Retour à l'agence en fin de tournée", value=True)
    with options[1]:
        time_limit = st.slider("Temps de calcul max (s)", min_value=1, max_value=15, value=5)

    optimize = st.button("🚀 Optimiser la tournée", type="primary")


with colB:
    st.subheader("Résultat")
    st.write("L'ordre optimal s'affichera ici (avec temps total estimé).")

    if not optimize:
        st.info("Renseigne l'agence + au moins 1 intervention, puis clique sur **Optimiser**.")
    else:
        # Validation
        jobs_df = st.session_state.jobs_df.copy()
        jobs_df = jobs_df.dropna(subset=["Nom", "Adresse", "Durée (min)"])
        jobs_df["Nom"] = jobs_df["Nom"].astype(str)
        jobs_df["Adresse"] = jobs_df["Adresse"].astype(str)

        if not agency_address.strip():
            st.error("Adresse agence manquante.")
            st.stop()

        if len(jobs_df) == 0:
            st.error("Ajoute au moins une intervention.")
            st.stop()

        # Géocodage agence
        with st.spinner("Géocodage de l'agence..."):
            try:
                g = geocode_address(agency_address.strip())
            except Exception as e:
                st.error(f"Erreur géocodage agence: {e}")
                st.stop()

        if g is None:
            st.error("Impossible de géocoder l'adresse de l'agence.")
            st.stop()

        agency_lat, agency_lon, agency_disp = g

        # Géocodage interventions
        stops_jobs: List[Stop] = []
        bad_rows = []
        with st.spinner("Géocodage des interventions..."):
            for i, row in jobs_df.iterrows():
                name = str(row["Nom"]).strip()
                addr = str(row["Adresse"]).strip()
                dur = int(row["Durée (min)"])
                if not name or not addr:
                    bad_rows.append(i)
                    continue
                try:
                    g2 = geocode_address(addr)
                except Exception as e:
                    st.error(f"Erreur géocodage '{name}': {e}")
                    st.stop()

                if g2 is None:
                    bad_rows.append(i)
                    continue
                lat, lon, disp = g2
                stops_jobs.append(Stop(name=name, address=disp, lat=lat, lon=lon, service_minutes=dur))

                # petite pause douce (respect API) si beaucoup d'adresses non cachées
                # (le cache st.cache_data évite la plupart des appels répétés)
                time.sleep(0.1)

        if bad_rows:
            st.warning(
                f"{len(bad_rows)} ligne(s) ignorée(s) (adresse ou nom vide, ou géocodage impossible). "
                "Corrige-les si besoin."
            )

        if len(stops_jobs) == 0:
            st.error("Aucune intervention géocodée correctement.")
            st.stop()

        # Construire la liste finale des stops pour le solveur
        # Start = agence (ou position technicien si tu la remplaces plus tard)
        # End = agence si retour demandé, sinon end = dernière intervention (non géré ici)
        agency_stop = Stop(name=f"{agency_name} (Agence)", address=agency_disp, lat=agency_lat, lon=agency_lon, service_minutes=0)

        if return_to_agency:
            # stops: [agency] + jobs + [agency_end]
            # astuce : end = agence_end (nœud distinct) pour éviter doublons / ambiguïtés
            agency_end = Stop(name=f"{agency_name} (Retour)", address=agency_disp, lat=agency_lat, lon=agency_lon, service_minutes=0)
            stops_all = [agency_stop] + stops_jobs + [agency_end]
            start_index = 0
            end_index = len(stops_all) - 1
        else:
            st.warning("Mode 'sans retour agence' non activé dans ce POC (à faire si tu veux).")
            # fallback : on force le retour pour éviter un résultat trompeur
            agency_end = Stop(name=f"{agency_name} (Retour)", address=agency_disp, lat=agency_lat, lon=agency_lon, service_minutes=0)
            stops_all = [agency_stop] + stops_jobs + [agency_end]
            start_index = 0
            end_index = len(stops_all) - 1

        coords = [(s.lat, s.lon) for s in stops_all]
        service = [s.service_minutes for s in stops_all]

        # Matrice OSRM
        with st.spinner("Calcul des temps de trajet (OSRM) ..."):
            try:
                tm = osrm_table_minutes(coords)
            except Exception as e:
                st.error(f"Erreur OSRM (temps de trajet): {e}")
                st.stop()

        # Optimisation
        with st.spinner("Optimisation (OR-Tools) ..."):
            res = solve_route(tm, service, start_index=start_index, end_index=end_index, time_limit_s=time_limit)

        if res is None:
            st.error("Aucune solution trouvée (contrainte/erreur).")
            st.stop()

        route_idx, total_min = res

        # Affichage
        ordered = [stops_all[i] for i in route_idx]
        st.success(f"Tournée calculée ✅ — Temps total estimé: **{total_min} min** (trajet + interventions)")

        df_out = pd.DataFrame(
            {
                "Ordre": list(range(1, len(ordered) + 1)),
                "Point": [s.name for s in ordered],
                "Adresse (résolue)": [s.address for s in ordered],
                "Durée intervention (min)": [s.service_minutes for s in ordered],
            }
        )
        st.dataframe(df_out, use_container_width=True)

        # Détail des trajets
        st.subheader("Détail des trajets (minutes)")
        legs = []
        for a, b in zip(route_idx[:-1], route_idx[1:]):
            legs.append(
                {
                    "De": stops_all[a].name,
                    "Vers": stops_all[b].name,
                    "Trajet (min)": tm[a][b],
                    "Service à l'arrivée (min)": service[b],
                    "Total jambe (min)": tm[a][b] + service[b],
                }
            )
        st.dataframe(pd.DataFrame(legs), use_container_width=True)

        st.caption(
            "Astuce: pour une version 'temps réel trafic', on remplace OSRM par Google/Here/TomTom Distance Matrix "
            "et on recalculera à chaque update de position technicien."
        )
        
