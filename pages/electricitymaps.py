import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from datetime import timedelta

from pages.utils.streamlit_utils import *


month_to_number = {
    "January": 1, "February": 2, "March": 3, "April": 4,
    "May": 5, "June": 6, "July": 7, "August": 8,
    "September": 9, "October": 10, "November": 11, "December": 12
}


# Function to create the dashboard
def dashboard_page():
    """Creates the Streamlit dashboard using the loaded data."""
    
    st.header("Energy by Region Dashboard")

    
    st.markdown("""
        **Description**: This dashboard visualizes energy data across different regions in the US and Germany.
        You can select daily or hourly data, choose specific years or months, and filter by regions.
        Additionally, you can toggle the inclusion of the US Average data and select specific metrics to display.
    """)

    st.write("")

    with st.container(border=True):
    
        # First Select Box: Daily or Hourly data
        data_view = st.selectbox("Select Data View", options=["Hourly", "Daily"])
        
        # Second Select Box: Dynamic, based on the first selection
        if data_view == "Daily":
            selected_year = st.selectbox("Select Year", options=[2021, 2022, 2023], index=2)
        else:
            selected_year = 2023  # Hourly data is only for 2023

            month_col, date_col = st.columns(2)

            with month_col:
                selected_month = st.selectbox("Select Month", options=[
                    "January", "February", "March", "April", "May", "June", 
                    "July", "August", "September", "October", "November", "December"
                ], index=8)

                # Determine the earliest and latest possible dates based on the selected month
                month_number = month_to_number[selected_month]
                earliest_date = pd.Timestamp(f"2023-{month_number:02d}-01")
                latest_date = earliest_date + pd.offsets.MonthEnd(1)

            with date_col:
                date_range = st.date_input("Select Date Range", value=[earliest_date + timedelta(days=1), earliest_date + timedelta(days=8)], min_value=earliest_date, max_value=latest_date)


        regions = ["US-Average", "US-West", "US-Central", "US-East", "Germany", "Switzerland", "France", "Iceland", "Norway", "Sweden"]  # Adjust regions as needed
        default = ["US-West", "US-East", "Germany", "Iceland", "Norway", "Sweden"] 
        selected_regions = st.multiselect("Select Regions", options=regions, default=default)
        
        # Select Box for Metric Selection
        metric = st.selectbox("Select Metric", options=[
            "Carbon Intensity gCO₂eq/kWh (direct)",
            "Carbon Intensity gCO₂eq/kWh (LCA)",
            "Low Carbon Percentage",
            "Renewable Percentage"
        ])
        
        # Check if the user has selected all necessary filters before loading data
        if data_view and selected_regions:
            # Load the correct file based on the data view and year
            if data_view == "Daily":
                file_name = f"results/energy_data/cleaned_data/all_regions_{selected_year}_daily.parquet"
            else:
                file_name = "results/energy_data/cleaned_data/all_regions_2023_hourly.parquet"
            
            # Load the data once the file path is determined
            data = load_data(file_name)

            # Filter the data based on user selections
            if data_view == "Daily":
                data_filtered = data[data['Year'] == selected_year]
            else:
                # Convert month to number and filter by date range
                if len(date_range) == 2:
                    data_filtered = data[
                        (data['Datetime (UTC)'].dt.date >= date_range[0]) & 
                        (data['Datetime (UTC)'].dt.date <= date_range[1])
                    ]
                else:
                    month_number = month_to_number[selected_month]
                    data_filtered = data[data['Datetime (UTC)'].dt.month == month_number]
            
            data_filtered = data_filtered[data_filtered['Zone'].isin(selected_regions)]
            
            # Convert UTC to local time for each selected region
            for region in selected_regions:
                data_filtered = convert_to_local_time(data_filtered, region)
            
            # Plot the data using local time
            line_chart = alt.Chart(data_filtered).mark_line().encode(
                x='Datetime (Local):T',  # Now using Local time for the x-axis
                y=f'{metric}:Q',
                color=f'Zone:N',
            ).interactive()

            st.altair_chart(line_chart, use_container_width=True)

            st.divider()

            with st.expander("🔍 Explain Metrics", expanded=False):
                
                st.write("")
                st.write("")

                st.markdown("""
                    **Carbon Intensity gCO₂eq/kWh (direct)**: 
                    This metric measures the amount of carbon dioxide equivalent (CO₂eq) emissions directly produced per kilowatt-hour of electricity generated. It focuses on the emissions from burning fossil fuels at power plants without considering upstream or downstream emissions. A lower value indicates cleaner energy production with less environmental impact.

                    **Carbon Intensity gCO₂eq/kWh (LCA)**: 
                    The Life Cycle Assessment (LCA) carbon intensity includes not only the direct emissions but also those generated during the entire lifecycle of energy production. This includes emissions from extracting raw materials, manufacturing, transportation, and disposal. LCA provides a more holistic view of the overall environmental impact of energy generation. A lower value means fewer emissions across the entire lifecycle.

                    **Low Carbon Percentage**: 
                    This percentage represents the share of energy produced from low-carbon sources, such as nuclear, hydro, wind, and solar. These sources emit minimal or no carbon dioxide directly during operation, helping to reduce overall carbon emissions. A higher percentage reflects a greater reliance on cleaner energy.

                    **Renewable Percentage**: 
                    The renewable percentage is the proportion of energy derived from renewable sources like wind, solar, geothermal, hydro, and biomass. Renewable energy is typically considered more sustainable since these resources are naturally replenished. A higher percentage suggests a greater shift towards sustainable energy production.
                """)

                st.write("")
                st.write("")


def electricityMaps(): 
    st.title("LLM Emission - ElectricityMaps 🌍🌱")

    info, button = st.columns([8, 2], gap="large")

    with info:
        st.info("""
        This data has been sourced from the ElectricityMaps API, specifically from their historical data portal. 
        For a real-time visualization of live electricity data, use the 
        [ElectricityMaps App](https://app.electricitymaps.com/).
        """)
    
    with button:
        st.link_button("🌍 ElectricityMaps App", "https://app.electricitymaps.com/", use_container_width=True)

    

    st.divider()

    # arena_results = st.tabs([
    #     "Arena Results",
    #     ])
    

    # with arena_results:
    st.write("")
    st.write("")

    dashboard_page()



if __name__ == "__main__":
    init_session_states()
    
    st.session_state.curr_page = 'electricityMaps'
    sidebar()
    electricityMaps()