"""
🌐 Continental Analysis Page - Temperature trends across continents
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys
from pathlib import Path

try:
    from climaxtreme.dashboard.utils import DataSource, configure_sidebar, show_data_info
except ImportError:
    _src_dir = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(_src_dir))
    from climaxtreme.dashboard.utils import DataSource, configure_sidebar, show_data_info

st.set_page_config(page_title="Continental Analysis", page_icon="🌐", layout="wide")
configure_sidebar()

st.title("🌐 Continental Analysis")
st.markdown("Global temperature trends across continents over time")

data_source = DataSource()
continental_df = data_source.load_parquet('continental.parquet')

if continental_df is not None and not continental_df.empty:
    show_data_info(continental_df, "Continental Dataset")
    
    # Sort data for proper visualization
    continental_df = continental_df.sort_values(['continent', 'year'])
    
    # Tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Overview",
        "📈 Temperature Trends",
        "🔄 Comparison",
        "📉 Change Analysis"
    ])
    
    with tab1:
        st.markdown("### 📊 Continental Temperature Overview")
        
        # Year filter
        years = sorted(continental_df['year'].unique())
        min_year = int(min(years))
        max_year = int(max(years))
        
        if min_year < max_year:
            selected_year = st.slider("Select Year", min_year, max_year, max_year)
        else:
            selected_year = min_year
            st.info(f"Data only available for year {min_year}")
        
        year_data = continental_df[continental_df['year'] == selected_year]
        
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("🌍 Continents", year_data['continent'].nunique())
        with col2:
            st.metric("🌡️ Global Avg", f"{year_data['avg_temperature'].mean():.2f}°C")
        with col3:
            if len(year_data) > 0:
                hottest_idx = year_data['avg_temperature'].idxmax()
                hottest = year_data.loc[hottest_idx, 'continent']
                hottest_temp = year_data.loc[hottest_idx, 'avg_temperature']
                st.metric("🔥 Hottest", f"{hottest}", f"{hottest_temp:.1f}°C")
        with col4:
            if len(year_data) > 0:
                coldest_idx = year_data['avg_temperature'].idxmin()
                coldest = year_data.loc[coldest_idx, 'continent']
                coldest_temp = year_data.loc[coldest_idx, 'avg_temperature']
                st.metric("❄️ Coldest", f"{coldest}", f"{coldest_temp:.1f}°C")
        
        # Interactive World Map
        st.markdown(f"#### 🗺️ Interactive World Map - {selected_year}")
        
        # Map continent names to representative countries for choropleth
        # We'll use anomalies data which has country-level info with continents
        anomalies_df = data_source.load_parquet('anomalies.parquet')
        
        if anomalies_df is not None and not anomalies_df.empty:
            # Get country-level data for selected year with continent info
            year_countries = anomalies_df[anomalies_df['year'] == selected_year].copy()
            
            if len(year_countries) > 0:
                # Aggregate by country
                country_temps = year_countries.groupby(['country', 'continent']).agg({
                    'temperature': 'mean',
                    'lat_numeric': 'first',
                    'lon_numeric': 'first'
                }).reset_index()
                country_temps.columns = ['country', 'continent', 'avg_temperature', 'lat', 'lon']
                
                # Create choropleth map colored by continent temperature
                # Merge with continental averages
                continent_temps = year_data.set_index('continent')['avg_temperature'].to_dict()
                country_temps['continent_temp'] = country_temps['continent'].map(continent_temps)
                
                fig_map = px.choropleth(
                    country_temps,
                    locations="country",
                    locationmode="country names",
                    color="continent_temp",
                    hover_name="country",
                    hover_data={
                        'continent': True,
                        'avg_temperature': ':.2f',
                        'continent_temp': ':.2f'
                    },
                    color_continuous_scale="RdYlBu_r",
                    title=f"World Temperature Map by Continent - {selected_year}",
                    labels={
                        'continent_temp': 'Continental Avg (°C)',
                        'avg_temperature': 'Country Avg (°C)'
                    }
                )
                fig_map.update_geos(
                    showcoastlines=True,
                    coastlinecolor="Gray",
                    showland=True,
                    landcolor="lightgray",
                    showocean=True,
                    oceancolor="lightblue",
                    projection_type="natural earth"
                )
                fig_map.update_layout(height=500)
                st.plotly_chart(fig_map, use_container_width=True)
            else:
                st.info(f"No country-level data available for {selected_year}")
        else:
            # Fallback: show continent centroids on a scatter geo map
            continent_coords = {
                'Africa': (0, 20),
                'Asia': (35, 100),
                'Europe': (50, 10),
                'North America': (45, -100),
                'South America': (-15, -60),
                'Oceania': (-25, 135),
                'Other': (0, 0)
            }
            
            map_data = year_data.copy()
            map_data['lat'] = map_data['continent'].apply(lambda x: continent_coords.get(x, (0, 0))[0])
            map_data['lon'] = map_data['continent'].apply(lambda x: continent_coords.get(x, (0, 0))[1])
            
            fig_map = go.Figure()
            
            for _, row in map_data.iterrows():
                color_val = (row['avg_temperature'] - map_data['avg_temperature'].min()) / \
                           (map_data['avg_temperature'].max() - map_data['avg_temperature'].min())
                
                fig_map.add_trace(go.Scattergeo(
                    lat=[row['lat']],
                    lon=[row['lon']],
                    mode='markers+text',
                    marker=dict(
                        size=30,
                        color=row['avg_temperature'],
                        colorscale='RdYlBu_r',
                        cmin=map_data['avg_temperature'].min(),
                        cmax=map_data['avg_temperature'].max(),
                        showscale=True if row['continent'] == map_data.iloc[0]['continent'] else False,
                        colorbar=dict(title='Temp (°C)')
                    ),
                    text=f"{row['continent']}<br>{row['avg_temperature']:.1f}°C",
                    textposition='top center',
                    name=row['continent'],
                    hovertemplate=f"<b>{row['continent']}</b><br>Temperature: {row['avg_temperature']:.2f}°C<extra></extra>"
                ))
            
            fig_map.update_geos(
                showcoastlines=True,
                coastlinecolor="Gray",
                showland=True,
                landcolor="lightgray",
                showocean=True,
                oceancolor="lightblue",
                projection_type="natural earth"
            )
            fig_map.update_layout(
                title=f"Continental Temperature Centers - {selected_year}",
                height=500,
                showlegend=False
            )
            st.plotly_chart(fig_map, use_container_width=True)
        
        # Bar chart for selected year
        st.markdown(f"#### Continental Temperatures - {selected_year}")
        fig = px.bar(
            year_data.sort_values('avg_temperature', ascending=False),
            x='continent',
            y='avg_temperature',
            color='avg_temperature',
            color_continuous_scale='RdYlBu_r',
            title=f"Average Temperature by Continent - {selected_year}",
            labels={'avg_temperature': 'Temperature (°C)', 'continent': 'Continent'}
        )
        fig.update_layout(height=500, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        
        # Data table
        st.markdown("#### 📋 Continental Data")
        display_df = year_data[['continent', 'avg_temperature']].copy()
        display_df.columns = ['Continent', 'Avg Temperature (°C)']
        display_df = display_df.sort_values('Avg Temperature (°C)', ascending=False)
        display_df['Avg Temperature (°C)'] = display_df['Avg Temperature (°C)'].round(2)
        st.dataframe(display_df, hide_index=True, use_container_width=True)
    
    with tab2:
        st.markdown("### 📈 Continental Temperature Evolution")
        
        # Multi-select continents
        continents = sorted(continental_df['continent'].unique())
        selected_continents = st.multiselect(
            "Select Continents",
            continents,
            default=continents
        )
        
        if selected_continents:
            trend_data = continental_df[continental_df['continent'].isin(selected_continents)]
            trend_data = trend_data.sort_values(['continent', 'year'])
            
            # Line chart
            fig = px.line(
                trend_data,
                x='year',
                y='avg_temperature',
                color='continent',
                title="Temperature Trends by Continent (1751-2013)",
                labels={'year': 'Year', 'avg_temperature': 'Average Temperature (°C)'}
            )
            fig.update_layout(height=600, hovermode='x unified')
            st.plotly_chart(fig, use_container_width=True)
            
            # Add trend lines
            st.markdown("#### 📊 Trend Statistics")
            stats_data = []
            for continent in selected_continents:
                cont_data = continental_df[continental_df['continent'] == continent]
                if len(cont_data) > 1:
                    first_temp = cont_data[cont_data['year'] == cont_data['year'].min()]['avg_temperature'].values[0]
                    last_temp = cont_data[cont_data['year'] == cont_data['year'].max()]['avg_temperature'].values[0]
                    change = last_temp - first_temp
                    stats_data.append({
                        'Continent': continent,
                        'First Year': cont_data['year'].min(),
                        'Last Year': cont_data['year'].max(),
                        'Start Temp (°C)': round(first_temp, 2),
                        'End Temp (°C)': round(last_temp, 2),
                        'Change (°C)': round(change, 2)
                    })
            
            if stats_data:
                stats_df = pd.DataFrame(stats_data)
                st.dataframe(stats_df, hide_index=True, use_container_width=True)
    
    with tab3:
        st.markdown("### 🔄 Continental Temperature Comparison")
        
        # Select two years to compare
        col1, col2 = st.columns(2)
        with col1:
            year1 = st.selectbox("First Year", sorted(years), index=0)
        with col2:
            year2 = st.selectbox("Second Year", sorted(years, reverse=True), index=0)
        
        data1 = continental_df[continental_df['year'] == year1][['continent', 'avg_temperature']].copy()
        data2 = continental_df[continental_df['year'] == year2][['continent', 'avg_temperature']].copy()
        
        data1.columns = ['Continent', f'{year1}']
        data2.columns = ['Continent', f'{year2}']
        
        comparison = data1.merge(data2, on='Continent', how='outer')
        comparison['Change'] = comparison[f'{year2}'] - comparison[f'{year1}']
        
        # Grouped bar chart
        fig = go.Figure()
        fig.add_trace(go.Bar(
            name=str(year1),
            x=comparison['Continent'],
            y=comparison[f'{year1}'],
            marker_color='steelblue'
        ))
        fig.add_trace(go.Bar(
            name=str(year2),
            x=comparison['Continent'],
            y=comparison[f'{year2}'],
            marker_color='indianred'
        ))
        fig.update_layout(
            title=f"Temperature Comparison: {year1} vs {year2}",
            barmode='group',
            height=500,
            yaxis_title='Temperature (°C)'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Change visualization
        st.markdown("#### Temperature Change")
        fig_change = px.bar(
            comparison.dropna(),
            x='Continent',
            y='Change',
            color='Change',
            color_continuous_scale='RdBu_r',
            color_continuous_midpoint=0,
            title=f"Temperature Change from {year1} to {year2}"
        )
        fig_change.update_layout(height=400)
        st.plotly_chart(fig_change, use_container_width=True)
        
        # Comparison table
        st.markdown("#### 📋 Comparison Data")
        comparison_display = comparison.copy()
        comparison_display = comparison_display.round(2)
        st.dataframe(comparison_display, hide_index=True, use_container_width=True)
    
    with tab4:
        st.markdown("### 📉 Long-term Change Analysis")
        
        # Decade analysis
        continental_df['decade'] = (continental_df['year'] // 10) * 10
        decade_avg = continental_df.groupby(['decade', 'continent'])['avg_temperature'].mean().reset_index()
        decade_avg = decade_avg.sort_values(['continent', 'decade'])
        
        fig = px.line(
            decade_avg,
            x='decade',
            y='avg_temperature',
            color='continent',
            markers=True,
            title="Decadal Average Temperature by Continent",
            labels={'decade': 'Decade', 'avg_temperature': 'Temperature (°C)'}
        )
        fig.update_layout(height=500, hovermode='x unified')
        st.plotly_chart(fig, use_container_width=True)
        
        # Heatmap of temperatures
        st.markdown("#### 🗺️ Temperature Heatmap (Decade × Continent)")
        
        pivot_data = decade_avg.pivot(index='continent', columns='decade', values='avg_temperature')
        
        fig_heatmap = px.imshow(
            pivot_data,
            labels=dict(x="Decade", y="Continent", color="Temp (°C)"),
            color_continuous_scale='RdYlBu_r',
            title="Continental Temperature Heatmap by Decade"
        )
        fig_heatmap.update_layout(height=400)
        st.plotly_chart(fig_heatmap, use_container_width=True)
        
        # Global trend
        st.markdown("#### 🌍 Global Average Trend")
        global_trend = continental_df.groupby('year')['avg_temperature'].mean().reset_index()
        global_trend = global_trend.sort_values('year')
        
        fig_global = px.line(
            global_trend,
            x='year',
            y='avg_temperature',
            title="Global Average Temperature Over Time",
            labels={'year': 'Year', 'avg_temperature': 'Global Average (°C)'}
        )
        fig_global.update_traces(line_color='darkred', line_width=2)
        fig_global.update_layout(height=400)
        st.plotly_chart(fig_global, use_container_width=True)

else:
    st.error("❌ Failed to load continental.parquet")
