import streamlit as st
import pandas as pd


def station_editor():
    """
    A component that displays editable station metadata (lat, lon, code) inside a form,
    and lets user select which stations to keep.
    Updates `st.session_state["selected_stations_detail"]` with edited values without losing previous data.
    """

    # # Check if session state has data; if not, show info and exit
    # if "selected_stations_detail" not in st.session_state or st.session_state["selected_stations_detail"].empty:
    #     st.info("No station metadata found.")
    #     return

    # Create a copy of the session state DataFrame to avoid modifying the original
    df = st.session_state["selected_stations_detail"].copy()

    with st.popover(f"Stations to use further: {len(df)}"):
        updated_rows = []
        at_least_one_selected = False

        for idx, (_, row) in enumerate(df.iterrows()):
            col1, col2, col3, col4 = st.columns([2, 2, 2, 2])

            with col1:
                st.text_input("Station Code", value=str(row.get("USGS station code", "")), disabled=True, key=f"code_{idx}")

            with col2:
                lat = st.text_input("Latitude", value=str(row.get("Latitude", "")), key=f"lat_{idx}")

            with col3:
                lon = st.text_input("Longitude", value=str(row.get("Longitude", "")), key=f"lon_{idx}")

            with col4:
                is_valid = False
                try:
                    if lat and lon:
                        float(lat)
                        float(lon)
                        is_valid = True
                except ValueError:
                    pass
                
                st.markdown("<div style='height: 35px;'></div>", unsafe_allow_html=True)
                if is_valid:
                    selected = st.checkbox("✓", key=f"selected_{idx}", value=False)
                else:
                    st.checkbox("✓", key=f"selected_{idx}", value=False, disabled=True)
                    selected = False

            if selected:
                at_least_one_selected = True

            updated_rows.append({
                "USGS station code": row.get("USGS station code", ""),
                "lat": float(lat) if is_valid else None,
                "lon": float(lon) if is_valid else None,
                "selected": selected
            })

        if at_least_one_selected:
            if st.button("✅ Submit"):
                updated_df = pd.DataFrame(updated_rows)
                filtered_df = updated_df[updated_df["selected"]]
                final_df = filtered_df.drop(columns="selected")
                st.session_state["final_station_list"] = final_df
                st.success(f"✅ Selected {len(final_df)} station(s).")
                return final_df
        else:
            st.button("✅ Submit", disabled=True)