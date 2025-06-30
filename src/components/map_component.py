import folium
import pandas as pd

from streamlit_folium import st_folium
from src.utils.functions import read_file
import streamlit as st

def render_map(csv_path=None, default_location=[39.8283, -98.5795], zoom=4):

    m = folium.Map(location=default_location, zoom_start=zoom)

    if csv_path:
        # Read file using read_file
        data = pd.read_csv(csv_path)
        if data is not None:
            # Check for required columns
            required_columns = ['dec_lat_va', 'dec_long_va', 'station_nm']
            if not all(col in data.columns for col in required_columns):
                st.error(
                    f"CSV must contain {', '.join(required_columns)} columns")
                return st_folium(m, width=700, height=500)

            # Group stations by parameter_code
            parameter_groups = data.groupby('parameter_code')
            
            # Create a feature group for each parameter_code
            for param_code, group in parameter_groups:
                layer = folium.FeatureGroup(name=f"Parameter {param_code}", show=True)
                
                # Add markers for each station in the group
                for _, row in group.iterrows():
                    popup_text = f"""
                    <b>Station:</b> {row['station_nm']}<br>
                    <b>Site No:</b> {row['site_no']}<br>
                    <b>Parameter Code:</b> {row['parameter_code']}
                    """
                    folium.Marker(
                        location=[row['dec_lat_va'], row['dec_long_va']],
                        popup=folium.Popup(popup_text, max_width=300),
                        icon=folium.Icon(color="blue", icon="info-sign")
                    ).add_to(layer)
                
                # Add layer to map
                layer.add_to(m)
            
            # Add layer control to toggle layers
            folium.LayerControl().add_to(m)
    

    # Add drawing tools
    folium.plugins.Draw(
        export=False,
        draw_options={
            'polyline': False,
            'polygon': True,
            'circle': False,
            'rectangle': True,
            'marker': False
        }
    ).add_to(m)

    return st_folium(m, width="100%", height=500)
