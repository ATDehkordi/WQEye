# src/components/export_page_component.py
import json
import pickle
import numpy as np
import ee
import geemap.foliumap as geemap
from datetime import timedelta, datetime
import pandas as pd
import streamlit as st
import folium
from folium.plugins import Draw, MeasureControl
from streamlit_folium import st_folium
from src.services.zip_download_helper import create_zip_buffer, get_download_link
import io
import zipfile
import requests
from folium.raster_layers import ImageOverlay

from matplotlib import colormaps
from rasterio.io import MemoryFile
import rasterio
from rasterio.transform import from_bounds


from settings import SENSORS_CONFIG
from src.config.state_manager import StateManager
from src.utils.functions import uploaded_file_to_gdf
from src.utils.zip_checker import check_and_extract_zip_contents
from src.utils.log_scale_transform import ytest_to_initial_scale


CENTER_START = [39.949610, -75.150282]
ZOOM_START = 4


def get_water_mask(roi_ee, image=None, start_date="2021-01-01", end_date="2022-01-01"):
    """
    Returns either:
    - the water mask image (if image=None), or
    - the given image masked to keep only water pixels (if image is provided)

    Parameters:
    - roi_ee (ee.Geometry): ROI as ee.Geometry
    - image (ee.Image or None): if provided, will be masked with water mask
    - start_date (str): start of water dataset period
    - end_date (str): end of water dataset period

    Returns:
    - ee.Image: either water mask, or masked input image
    """
    water_mask = (
        ee.ImageCollection('JRC/GSW1_4/YearlyHistory')
        .filterDate(start_date, end_date)
        .toBands()
        .eq(3)  # class 3 = permanent water
        .clip(roi_ee)
    )

    if image:
        return image.updateMask(water_mask)
    else:
        return water_mask




def geotiff_download_link(array_2d, bounds, filename="predicted_map.tif", crs="EPSG:4326", link_text="Click to Download Predict Map"):
    """
    Creates a GeoTIFF in memory and renders a download link as HTML.

    Parameters:
    - array_2d (np.ndarray): The 2D array to convert to GeoTIFF
    - bounds (list): [minx, miny, maxx, maxy] spatial extent
    - filename (str): Name of the downloadable file
    - crs (str): Coordinate system, default EPSG:4326
    - link_text (str): Text to show for download link
    """
    try:
        height, width = array_2d.shape
        transform = from_bounds(bounds[0], bounds[1], bounds[2], bounds[3], width, height)

        with MemoryFile() as memfile:
            with memfile.open(
                driver='GTiff',
                height=height,
                width=width,
                count=1,
                dtype=array_2d.dtype,
                transform=transform,
                crs=crs
            ) as dataset:
                dataset.write(array_2d, 1)

            geotiff_bytes = memfile.read()
            download_link = get_download_link(
                zip_bytes=geotiff_bytes,
                filename=filename,
                link_text=link_text
            )
            st.markdown(download_link, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"❌ Error generating download link: {e}")



@st.cache_resource(ttl=3600, show_spinner="Create Map all layer")
def create_map_with_all_layers(_image_options, _collection, _roi_gdf, sensor, interested_date, th_cloud):
    """
    Creates and caches a geemap.Map object with all found GEE images added as layers.
    The underscore prefix on arguments tells streamlit to hash the object's contents.
    """
    # st.info(f"Creating map with {len(_image_options)} layers for the first time. This might take a moment...")
    st.info(f"Creating new map for date: {interested_date.strftime('%Y-%m-%d')} and cloud cover <= {th_cloud}%.")

    # create a map centered on the roi
    centroid = _roi_gdf.geometry.centroid.iloc[0]
    m = geemap.Map(center=(centroid.y, centroid.x), zoom=11)

    # get vis_params and roi_ee
    vis_params = SENSORS_CONFIG[sensor]['vis_params']
    roi_ee = geemap.gdf_to_ee(_roi_gdf, geodesic=False)
    
    # --- the slow loop runs only when the cache is empty ---
    is_first_layer = True
    for display_name, image_info in _image_options.items():
        filters = [ee.Filter.eq(key, value) for key, value in image_info.items()]
        image = _collection.filter(ee.Filter.And(filters)).first()
        
        m.addLayer(
            image.clip(roi_ee),
            vis_params,
            display_name,
            shown=is_first_layer
        )
        is_first_layer = False

    m.add_gdf(_roi_gdf, layer_name="ROI Boundary", style=style_roi)
    # water_mask = ee.ImageCollection('JRC/GSW1_4/YearlyHistory').filterDate('2021-01-01', '2022-01-01').toBands().eq(3).clip(roi_ee)
    water_mask = get_water_mask(roi_ee)
    m.addLayer(water_mask, {}, 'Water Mask Layer')

    return m


@st.cache_data(ttl=3600, show_spinner=False)  # Cache results for 1 hour
def find_available_images(sensor, start_date, end_date, th_cloud, _region, interested_date):
    """
    Performs the GEE search and returns a dictionary of available images.
    Results are cached to prevent re-running the search unnecessarily.
    The _region argument is used by the caching mechanism to detect changes.
    """
    # st.info("Searching for available satellite images...") # Show info message
    interested_date_ee = ee.Date(interested_date.strftime('%Y-%m-%d'))
    roi = geemap.gdf_to_ee(_region, geodesic=False)

    sensor_info = SENSORS_CONFIG[sensor]
    collection = (
        ee.ImageCollection(sensor_info['collection'])
        .filterBounds(roi)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lt(sensor_info['cloud_property'], th_cloud))
        .map(sensor_info['scale_function'])
    )

    if collection.size().getInfo() == 0:
        return {}, None  # Return an empty dict if no images are found

    properties_to_get = ['system:time_start'] + \
        sensor_info['unique_properties']
    image_info_list = collection.reduceColumns(
        reducer=ee.Reducer.toList().repeat(len(properties_to_get)),
        selectors=properties_to_get
    ).get('list').getInfo()

    image_details_list = [dict(zip(properties_to_get, item_values))
                          for item_values in zip(*image_info_list)]
    interested_timestamp = interested_date_ee.getInfo()['value']

    sorted_image_details = sorted(
        image_details_list,
        key=lambda item: abs(
            int(item['system:time_start']) - interested_timestamp)
    )

    image_options = {}
    for item in sorted_image_details:
        date_str = datetime.fromtimestamp(
            item['system:time_start'] / 1000).strftime('%Y-%m-%d')
        props_str = " | ".join(
            [f"{prop}: {item[prop]}" for prop in sensor_info['unique_properties']])
        display_name = f"{date_str} | {props_str}"
        image_options[display_name] = item

    return image_options, collection


def initialize_map():
    """
    Initializes a folium map with default settings.
    Since it has no arguments, this will run exactly once and the same
    map object will be returned every time, preserving its state (pan/zoom).
    """
    m = folium.Map(location=CENTER_START, zoom_start=ZOOM_START, scrollWheelZoom=True)

    draw = Draw(export=True,
                filename='Region.geojson',
                position='topleft',
                draw_options={'polyline': False,
                              'rectangle': True,  # keep rectangle option for flexibility in drawing
                              'polygon': {'showArea': True, 'showLength': False, 'metric': False, 'feet': False},
                              'circle': {'showArea': True, 'showLength': False, 'metric': False, 'feet': False},
                              'circlemarker': False,
                              'marker': False,
                              },
                edit_options={'poly': {'allowIntersection': False}})
    draw.add_to(m)
    MeasureControl(position='bottomleft').add_to(m)
    
    return m

style_roi = {
    "color": "#ff3939",
    "fillOpacity": 0,
    "weight": 3,
    "opacity": 1,
    "dashArray": "5, 5",
}


def render_roi_tab():
    """
    Handles all UI and logic for the 'Region Of Interest' tab.
    It returns the determined active ROI (either a GeoDataFrame or a dict).
    """

    uploaded_file = st.file_uploader(
        "Upload ROI GeoJSON file",
        type=["geojson", "json"]
    )
    # Initialize the map with current center and zoom from StateManager
    m = initialize_map()

    gdf = uploaded_file_to_gdf(uploaded_file)
    if gdf is not None and not gdf.empty:
        active_gdf = gdf
        geojson_layer = folium.GeoJson(
                data=active_gdf,
                style_function=lambda x: style_roi,
                name="region"
            )
            
        geojson_layer.add_to(m)
        
        m.fit_bounds(geojson_layer.get_bounds())

    # Display the map
    st_folium(m,
              key="folium_map_interactive", 
              height=500,
              width="100%",
              layer_control=folium.LayerControl(collapsed=True),
              returned_objects=[])
    return gdf



def render_image_selection_tab(PAGE_NAME='export'):
    """
    Handles the UI and logic for the image selection tab.
    """

    active_roi_gdf = StateManager.get_page_state(PAGE_NAME, 'region')

    if active_roi_gdf is None:
        st.warning(
                "⚠️ Region of Interest (ROI) not set.\n\n"
                "To use this tab, please first upload a valid GeoJSON file "
                "in the **'Region Of Interest'** tab."
            )
        StateManager.clear_page_state(PAGE_NAME)
        return
    
    roi_ee = geemap.gdf_to_ee(active_roi_gdf, geodesic=False)

    sensor_options = [item for item in list(SENSORS_CONFIG.keys()) if item != 'L89']
    main_container = st.container(border=False)
    with main_container:
        with st.form(key='params_form', border=False):
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                sensor = st.selectbox(
                    'sensor:',
                    options=sensor_options,
                    index=sensor_options.index(StateManager.get_page_state(PAGE_NAME, "sensor", sensor_options[0]))
                )
                StateManager.set_page_state(PAGE_NAME, 'sensor', sensor)

            with col2:
                interested_date_input = st.date_input(
                    "date of interest:",
                    # value=StateManager.get_page_state(PAGE_NAME, 'interested_date', 'today')
                )
                StateManager.set_page_state(PAGE_NAME, 'interested_date', interested_date_input)

            with col3:
                th_cloud = st.number_input('cloud cover threshold (%):',
                    min_value=0,
                    max_value=100,
                    value=StateManager.get_page_state(PAGE_NAME, 'th_cloud', 10),
                    step=1
                )
                StateManager.set_page_state(PAGE_NAME, 'th_cloud', th_cloud)

            with col4:
                output_resolution = st.number_input(
                    'output resolution (m):',
                    min_value=10,
                    value=StateManager.get_page_state(PAGE_NAME, 'output_resolution', 30),
                    step=10
                )
                StateManager.set_page_state(PAGE_NAME, 'output_resolution', output_resolution)

            submitted = st.form_submit_button("Search Images", type='primary')
        
    if submitted:
 

        start_date = (interested_date_input - timedelta(days=30)).strftime('%Y-%m-%d')
        end_date = (interested_date_input + timedelta(days=30)).strftime('%Y-%m-%d')

        with st.spinner("Searching for available satellite images..."):
            image_options, collection = find_available_images(sensor, start_date, end_date, th_cloud, active_roi_gdf, interested_date_input)

            # Store results in StateManager
            StateManager.set_page_state(PAGE_NAME, 'image_options', image_options)
            StateManager.set_page_state(PAGE_NAME, 'collection', collection)

            if not image_options:
                st.warning("No images found for the selected criteria. Try adjusting the date or cloud cover.")
            else:
                st.success(f"Found {len(image_options)} available images.")


    image_options = StateManager.get_page_state(PAGE_NAME, 'image_options')

    if image_options:
        interested_date=StateManager.get_page_state(PAGE_NAME, 'interested_date')
        collection = StateManager.get_page_state(PAGE_NAME, 'collection')
        sensor = StateManager.get_page_state(PAGE_NAME, 'sensor')
        cloud_cover= StateManager.get_page_state(PAGE_NAME, 'th_cloud')

        # simply call the cached function. this will be instant after the first run.
        m = create_map_with_all_layers(
            image_options, 
            collection, 
            active_roi_gdf, 
            sensor, 
            interested_date,
            cloud_cover
        )


        # Display the map
        st_folium(m,
                key="image_shows", 
                height=500,
                width="100%",
                layer_control=folium.LayerControl(collapsed=True),
                returned_objects=[])
            
        # --- Final Selection Section (below the map) ---
        st.markdown("#### Final Image Selection")
        st.markdown("After exploring the images on the map, please make your final selection below.")
        
        selected_display_name = st.selectbox(
            "Select the final image for prediction:",
            options=list(image_options.keys())
        )
        if selected_display_name:
            # get the final selected image object
            selected_image_info = image_options[selected_display_name]
            filters = [ee.Filter.eq(key, value) for key, value in selected_image_info.items()]
            final_selected_image = collection.filter(ee.Filter.And(filters)).first().clip(roi_ee)

            final_selected_image_masked = get_water_mask(roi_ee, image=final_selected_image)

            # store the final choice in state for tab 3
            StateManager.set_page_state(PAGE_NAME, 'final_selected_image', final_selected_image_masked)

        final_selected_image = StateManager.get_page_state(PAGE_NAME, 'final_selected_image',None)


        download_butom = st.button('Download Images', type='primary')
        if download_butom:

            with st.spinner("Preparing images for download... This may take a moment."):
                # 1. create the base name from the date and tile info
                base_name = selected_display_name.replace(" | ", "_").replace(":", "-").replace(" ", "")

                # 2. retrieve the cloud and resolution values from the state
                cloud_cover = StateManager.get_page_state(PAGE_NAME, 'th_cloud', 'nan')
                resolution = StateManager.get_page_state(PAGE_NAME, 'output_resolution', 'nan')

                safe_filename = f"{base_name}_cloud{cloud_cover}pct_res{resolution}m"

            
                try:
                    # Get URL for the clipped image
                    clipped_url = final_selected_image.getDownloadUrl({
                        'name': 'original_image',
                        'bands': SENSORS_CONFIG[sensor]['bands'],
                        'region': roi_ee.geometry(),
                        'scale': resolution,
                        'filePerBand': False
                    })
                    response_clipped = requests.get(clipped_url)
                    response_clipped.raise_for_status()

                    # water_mask = ee.ImageCollection('JRC/GSW1_4/YearlyHistory').filterDate('2021-01-01', '2022-01-01').toBands().eq(3).clip(roi_ee)
                    image_masked = get_water_mask(roi_ee, image=final_selected_image)
                    # image_masked = final_selected_image.updateMask(water_mask)
                    
                    # Get URL for the masked image
                    masked_url = image_masked.getDownloadUrl({
                        'name': 'masked_image',
                        'bands': SENSORS_CONFIG[sensor]['bands'],
                        'region': roi_ee.geometry(),
                        'scale': output_resolution,
                        'filePerBand': False
                    })
                    response_masked = requests.get(masked_url)
                    response_masked.raise_for_status()


                    with zipfile.ZipFile(io.BytesIO(response_clipped.content)) as nested_zip:
                        original_tif_content = nested_zip.read(nested_zip.namelist()[0])

                    with zipfile.ZipFile(io.BytesIO(response_masked.content)) as nested_zip:
                        masked_tif_content = nested_zip.read(nested_zip.namelist()[0])


                    files_for_zip = {
                                f"{safe_filename}_original.tif": original_tif_content,
                                f"{safe_filename}_masked.tif": masked_tif_content
                            }
                    # 3. use your generic function to create the zip buffer
                    zip_buffer = create_zip_buffer(files_for_zip)

                    download_link = get_download_link(
                        zip_buffer.getvalue(),
                        filename=f"{safe_filename}_images.zip",
                        link_text="Download Images (Original & Masked)"
                    )

                    # 5. display the link
                    st.markdown(download_link, unsafe_allow_html=True)
                except Exception as e:
                        st.error(f"Could not generate the combined download link: {e}")


def render_prediction_tab(PAGE_NAME='export'):

    # 1. check for dependencies from previous tabs.
    final_selected_image = StateManager.get_page_state(PAGE_NAME, 'final_selected_image')
    active_roi_gdf = StateManager.get_page_state(PAGE_NAME, 'region')

    if final_selected_image is None or active_roi_gdf is None:
        st.warning("⚠️ Please select a valid ROI in Tab ① and a final image in Tab ② to proceed.")
        return
    
    zip_file = st.file_uploader(
        "Upload a ZIP model package",
        type=['zip'],
        key="model_zip_upload",
        help="Please upload a ZIP file that contains the following files:\n"
            "- `scalers.pkl`: Pickled scalers and transformers used during preprocessing\n\n"
            "Make sure the filenames match exactly. The app will check their presence automatically."
    )
    if zip_file:
        # 3. process the uploaded zip file.
        required_files = {"model.pkl", "scalers.pkl", "metadata.json"}
        is_valid, contents, error_msg = check_and_extract_zip_contents(zip_file, required_files)

        if not is_valid:
            st.error(f"Invalid ZIP file: {error_msg}")
            st.stop()
        st.success("Model package loaded successfully!")
       
        try:
            model = pickle.load(contents["model.pkl"])
            scalers = pickle.load(contents["scalers.pkl"])
            metadata = json.loads(contents["metadata.json"].read().decode('utf-8'))
            features_used = metadata.get("features_used")
            if not features_used:
                st.error("Could not find 'features_used' in metadata.json.")

        except Exception as e:
            st.error(f"Error loading model artifacts: {e}")
            st.stop()
        
        # 5. prediction logic.
        if st.button("Run Prediction", type="primary"):
            with st.spinner("Applying model to image... This may take a while."):
                try:
                    roi_ee = geemap.gdf_to_ee(active_roi_gdf)
                    img_array = geemap.ee_to_numpy(final_selected_image, bands=features_used, region=roi_ee.geometry(),scale=30)

                    h, w, d = img_array.shape
                    img_reshaped = img_array.reshape(-1, d)

                    # detect masked pixels (all bands == 0)
                    masked_pixels = np.all(img_reshaped == 0, axis=1)

                    # mark them as nan
                    img_reshaped = img_reshaped.astype(np.float32)
                    img_reshaped[masked_pixels] = np.nan

                    # now separate valid pixels
                    valid_idx = ~masked_pixels
                    img_valid = img_reshaped[valid_idx]

                    # log + scale only valid
                    img_valid_log = np.log(img_valid + scalers['shift_value_X'])
                    img_valid_trans = scalers['transformerX'].transform(img_valid_log)
                    img_valid_scaled = scalers['min_max_scalerX'].transform(img_valid_trans)

                    pred_valid = model.predict(pd.DataFrame(img_valid_scaled, columns=features_used))
                    pred_original = ytest_to_initial_scale(
                        pred_valid,
                        scalers['min_max_scalerY'],
                        scalers['transformerY'],
                        scalers['shift_value_Y']
                    )

                    # fill prediction map
                    pred_full = np.full(img_reshaped.shape[0], np.nan, dtype=np.float32)
                    pred_full[valid_idx] = np.array(pred_original).flatten()

                    predicted_map_array = pred_full.reshape(h, w)
                                        
                    # store the result for display
                    StateManager.set_page_state(PAGE_NAME, 'predicted_map_array', predicted_map_array)
                except Exception as e:
                    st.error(f"An error occurred during prediction: {e}")

        # 6. display the result if it exists.
        predicted_map = StateManager.get_page_state(PAGE_NAME, 'predicted_map_array')
        if predicted_map is not None:

            with st.spinner("Creating final map visualization..."):
                min_val = np.nanmin(predicted_map)
                max_val = np.nanmax(predicted_map)
                
                if max_val == min_val:
                    normalized_map = np.zeros_like(predicted_map, dtype=np.float32)
                else:
                    normalized_map = (predicted_map - min_val) / (max_val - min_val)

                # b. Apply a colormap to convert the array to an RGBA image
                cmap = colormaps['coolwarm'] # You can change 'viridis' to 'coolwarm' or others
                rgba_image = cmap(normalized_map)
                rgba_image_uint8 = (rgba_image * 255).astype(np.uint8)

                # c. Get the geographical bounds from the original ROI
                bounds = active_roi_gdf.total_bounds
                map_bounds = [[bounds[1], bounds[0]], [bounds[3], bounds[2]]] 
                
                # d. Create a new Folium map centered on the ROI
                centroid = active_roi_gdf.geometry.centroid.iloc[0]
                m_results = folium.Map(location=[centroid.y, centroid.x], zoom_start=11)

                # e. Add the predicted image as an overlay
                ImageOverlay(
                    image=rgba_image_uint8,
                    bounds=map_bounds,
                    opacity=0.8,
                    name="Predicted Map"
                ).add_to(m_results)


                # --- add ROI layer ---
                if active_roi_gdf is not None and not active_roi_gdf.empty:
                    geojson_layer = folium.GeoJson(
                        data=active_roi_gdf,
                        style_function=lambda x: style_roi,
                        name="Study Area"
                    )
                    geojson_layer.add_to(m_results)

                # add layer control
                folium.LayerControl(collapsed=True).add_to(m_results)

                # display map in streamlit
                st_folium(
                    m_results,
                    key="prediction_map_folium",
                    height=500,
                    width="100%",
                    returned_objects=[]
                )
        predicted_map = StateManager.get_page_state(PAGE_NAME, 'predicted_map_array')
        if predicted_map is not None:
            bounds = active_roi_gdf.total_bounds
            geotiff_download_link(predicted_map, bounds)