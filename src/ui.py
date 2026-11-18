"""Streamlit evaluator for the PathSnap FastAPI route service."""

from __future__ import annotations

import os

import folium
import requests
import streamlit as st
from streamlit_folium import st_folium

DEFAULT_API_URL = "http://127.0.0.1:8000"
REQUEST_TIMEOUT_SECONDS = 120


@st.cache_data(ttl=30, show_spinner=False)
def get_reserved_trips(api_url: str) -> list[str]:
    response = requests.get(f"{api_url}/trips", timeout=10)
    response.raise_for_status()
    trip_ids = response.json()
    if not isinstance(trip_ids, list) or not all(
        isinstance(trip_id, str) for trip_id in trip_ids
    ):
        raise ValueError("The route API returned an invalid trip list.")
    return trip_ids


def build_route_map(result: dict) -> folium.Map:
    observations = result["observations"]
    gps_points = [
        [point["latitude"], point["longitude"]]
        for point in observations
    ]
    route_points: list[list[float]] = []
    for point in observations:
        geometry = point.get("geometry")
        if geometry:
            for longitude, latitude in geometry:
                coordinate = [latitude, longitude]
                if not route_points or route_points[-1] != coordinate:
                    route_points.append(coordinate)

    center = [
        sum(point[0] for point in gps_points) / len(gps_points),
        sum(point[1] for point in gps_points) / len(gps_points),
    ]
    route_map = folium.Map(location=center, zoom_start=13, control_scale=True)
    folium.PolyLine(
        gps_points,
        color="#247ba0",
        weight=3,
        opacity=0.75,
        tooltip="Original GPS observations",
    ).add_to(route_map)
    if route_points:
        folium.PolyLine(
            route_points,
            color="#e63946",
            weight=5,
            opacity=0.9,
            tooltip="Predicted road route",
        ).add_to(route_map)
    folium.CircleMarker(
        gps_points[0], radius=5, color="#2a9d8f", tooltip="Trip start"
    ).add_to(route_map)
    folium.CircleMarker(
        gps_points[-1], radius=5, color="#6a4c93", tooltip="Trip end"
    ).add_to(route_map)
    route_map.fit_bounds(gps_points)
    folium.LayerControl().add_to(route_map)
    return route_map


def main() -> None:
    st.set_page_config(page_title="PathSnap Evaluator", page_icon="🗺️", layout="wide")
    st.title("PathSnap Route Evaluator")
    st.caption(
        "Select a reserved evaluation trip to request a map-matched route from the backend."
    )
    api_url = os.environ.get("PATHSNAP_API_URL", DEFAULT_API_URL).rstrip("/")

    try:
        trip_ids = get_reserved_trips(api_url)
    except (requests.RequestException, ValueError) as error:
        st.error(
            f"Could not load trips from {api_url}. Check the backend address and "
            f"that Laptop 1 is running: {error}"
        )
        st.stop()
    if not trip_ids:
        st.error("The backend did not find any trips in its reserved evaluation set.")
        st.stop()

    trip_id = st.selectbox("Reserved evaluation trip", trip_ids)
    if st.button("Predict route", type="primary"):
        with st.spinner(f"Matching trip {trip_id}..."):
            try:
                response = requests.post(
                    f"{api_url}/predict_route",
                    json={"trip_id": trip_id},
                    timeout=REQUEST_TIMEOUT_SECONDS,
                )
                response.raise_for_status()
                result = response.json()
            except requests.RequestException as error:
                st.error(f"Route prediction request failed: {error}")
                st.stop()
            except ValueError as error:
                st.error(f"The backend returned invalid JSON: {error}")
                st.stop()

        st.success(f"Route prediction completed for trip {result['trip_id']}.")
        metric_columns = st.columns(2)
        metric_columns[0].metric("GPS observations", result["observation_count"])
        metric_columns[1].metric(
            "Matched observations", result["matched_observation_count"]
        )
        st_folium(
            build_route_map(result),
            height=620,
            use_container_width=True,
            returned_objects=[],
        )
        with st.expander("Predicted road-segment sequence"):
            st.dataframe(
                [
                    {
                        "observation": point["observation_index"],
                        "road segment": point["arc_id"],
                    }
                    for point in result["observations"]
                ],
                use_container_width=True,
                hide_index=True,
            )


if __name__ == "__main__":
    main()
