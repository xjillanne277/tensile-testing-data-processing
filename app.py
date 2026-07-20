import streamlit as st
import pandas as pd
import numpy as np
import scipy.integrate
import matplotlib.pyplot as plt
import plotly.express as px
import io

st.set_page_config(page_title="Instron Data Processor", layout="wide")

st.title("Instron Tensile Testing Data Processing")

st.markdown("""
Upload raw Instron CSV files to parse, clean, and visualize tensile mechanical testing data. 
Automatically zeroes baselines, calculates Energy Absorption (J), and provides downloadable cleaned CSVs.
""")

def process_file(uploaded_file):
    lines = uploaded_file.getvalue().decode('utf-8').splitlines()
    
    # Find start indices of data blocks
    data_starts = []
    for i, line in enumerate(lines):
        if 'Time,Displacement,Force' in line:
            data_starts.append(i)
            
    if not data_starts:
        raise ValueError("Could not find 'Time,Displacement,Force' data blocks in the file.")
            
    all_data = []
    summary = []
    
    for i, start_idx in enumerate(data_starts):
        specimen_id = f"Specimen_{i+1:02d}"
        
        end_idx = data_starts[i+1] if i+1 < len(data_starts) else len(lines)
        data_lines = lines[start_idx+2:end_idx]
        data_lines = [l.strip() for l in data_lines if l.strip() and not l.startswith('Results')]
        data_lines = [l for l in data_lines if len(l.split(',')) >= 8]
        
        if not data_lines:
            raise ValueError(f"No valid data rows found for {specimen_id}.")
        
        parsed_data = []
        for l in data_lines:
            parts = l.split(',')
            if len(parts) >= 8:
                row = [float(x.strip().strip('"').strip("'")) for x in parts[1:8]]
                parsed_data.append(row)
        
        if not parsed_data:
            raise ValueError(f"Failed to parse numerical data for {specimen_id}.")
            
        df = pd.DataFrame(parsed_data, dtype=float)
        df.columns = ['time_s', 'displacement_mm', 'force_n', 'tensile_strain_percent', 
                      'tensile_stress_mpa', 'tensile_displacement_mm', 'corrected_displacement_mm']
        
        # Zeroing/Baseline Calibration
        df['force_n'] = df['force_n'] - df['force_n'].iloc[0]
        df['tensile_stress_mpa'] = df['tensile_stress_mpa'] - df['tensile_stress_mpa'].iloc[0]
        
        # Noise Reduction (remove negative strain)
        df = df[df['tensile_strain_percent'] >= 0.0]
        
        # Anomaly Detection & Cleaning
        anomalies = []
        if (df['corrected_displacement_mm'].diff() < -0.05).any():
            anomalies.append("Self-Intersecting/Machine Return (Auto-Fixed)")
            
        # Truncate at max displacement to remove return loop and stop at fracture
        max_disp_idx = df['corrected_displacement_mm'].idxmax()
        df = df.loc[:max_disp_idx].copy()
        
        df['specimen_id'] = specimen_id
        
        # Calculate Energy Absorption (J)
        # Trapezoidal rule for Force (N) vs Displacement (mm) -> 1 N*mm = 0.001 J
        energy_j = scipy.integrate.trapezoid(df['force_n'], df['corrected_displacement_mm']) * 0.001
        
        # Modulus estimation
        linear_region = df[(df['tensile_strain_percent'] > 0.05) & (df['tensile_strain_percent'] < 0.25)]
        if len(linear_region) > 1:
            slope, _ = np.polyfit(linear_region['tensile_strain_percent']/100.0, linear_region['tensile_stress_mpa'], 1)
            modulus_gpa = slope / 1000.0
        else:
            modulus_gpa = np.nan
            
        peak_stress = df['tensile_stress_mpa'].max()
        strain_break = df['tensile_strain_percent'].max()
        
        all_data.append(df)
        summary.append({
            'Specimen': specimen_id,
            'Peak Stress (MPa)': peak_stress,
            'Modulus (GPa)': modulus_gpa,
            'Strain at Break (%)': strain_break,
            'Energy (J)': energy_j,
            'Anomalies': ", ".join(anomalies) if anomalies else "None"
        })
        
    return pd.concat(all_data, ignore_index=True), pd.DataFrame(summary)

col_up1, col_up2 = st.columns([3, 1])
with col_up1:
    uploaded_files = st.file_uploader("Upload Raw Instron CSV(s)", type="csv", accept_multiple_files=True)
with col_up2:
    st.write("")
    st.write("")
    use_demo = st.toggle("🚀 Use Sample Data for Demo")

if use_demo and not uploaded_files:
    try:
        with open("Instron Sample Data - jilltest-gray_1.csv", "rb") as f:
            class DummyFile:
                def __init__(self, data, name):
                    self.data = data
                    self.name = name
                def getvalue(self):
                    return self.data
            uploaded_files = [DummyFile(f.read(), "Instron Sample Data - jilltest-gray_1.csv")]
    except FileNotFoundError:
        st.error("Sample data file not found!")

if uploaded_files:
    dfs = []
    summaries = []
    
    for file in uploaded_files:
        try:
            df, summary = process_file(file)
            dfs.append(df)
            summaries.append(summary)
        except Exception as e:
            st.error(f"Error processing {file.name}: {e}")
            
    if dfs:
        final_df = pd.concat(dfs, ignore_index=True)
        final_summary = pd.concat(summaries, ignore_index=True)
        
        st.header("1. Summary Metrics")
        
        display_summary = final_summary.copy()
        for col in ['Peak Stress (MPa)', 'Modulus (GPa)', 'Strain at Break (%)', 'Energy (J)']:
            display_summary[col] = display_summary[col].round(2)
        st.dataframe(display_summary, use_container_width=True)
        
        st.subheader("Statistical Variation Analysis Among Samples")
        stats_df = pd.DataFrame()
        stats_df['Metric'] = ['Peak Stress (MPa)', 'Modulus (GPa)', 'Strain at Break (%)', 'Energy (J)']
        stats_df['Mean'] = [final_summary[c].mean() for c in stats_df['Metric']]
        stats_df['Std Dev'] = [final_summary[c].std() for c in stats_df['Metric']]
        stats_df['Coefficient of Variation (%)'] = (stats_df['Std Dev'] / stats_df['Mean']) * 100
        for col in ['Mean', 'Std Dev', 'Coefficient of Variation (%)']:
            stats_df[col] = stats_df[col].round(2)
        st.dataframe(stats_df, use_container_width=True)
        
        st.header("2. Visualizations")
        
        if 'focus_specimen' not in st.session_state:
            st.session_state.focus_specimen = "All Specimens"

        def reset_focus():
            st.session_state.focus_specimen = "All Specimens"

        focus_specimen = st.selectbox("Focus Specimen:", options=["All Specimens"] + list(final_summary['Specimen'].unique()), key='focus_specimen')
        
        plot_df = final_df
        if focus_specimen != "All Specimens":
            st.button("Reset View", on_click=reset_focus)
            plot_df = final_df[final_df['specimen_id'] == focus_specimen]
            row = final_summary[final_summary['Specimen'] == focus_specimen].iloc[0]
            st.info(f"🌟 **Specimen Spotlight: {focus_specimen}**\n\n"
                    f"Peak Stress: **{row['Peak Stress (MPa)']:.2f} MPa** | "
                    f"Young's Modulus: **{row['Modulus (GPa)']:.2f} GPa** | "
                    f"Strain at Break: **{row['Strain at Break (%)']:.2f} %** | "
                    f"Energy Absorption: **{row['Energy (J)']:.2f} J**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Stress vs. Strain")
            fig1 = px.line(plot_df, x='tensile_strain_percent', y='tensile_stress_mpa', color='specimen_id',
                           labels={'tensile_strain_percent': 'Tensile Strain (%)', 'tensile_stress_mpa': 'Tensile Stress (MPa)'},
                           title='Stress vs Strain')
            st.plotly_chart(fig1, use_container_width=True)
            
        with col2:
            st.subheader("Force vs. Displacement")
            fig2 = px.line(plot_df, x='corrected_displacement_mm', y='force_n', color='specimen_id',
                           labels={'corrected_displacement_mm': 'Corrected Displacement (mm)', 'force_n': 'Force (N)'},
                           title='Load vs Displacement')
            st.plotly_chart(fig2, use_container_width=True)
            
        st.markdown("---")
        st.subheader("Energy Absorption (Area Under Load-Displacement Curve)")
        
        # Display metric cards for energy absorption
        m_cols = st.columns(min(len(final_summary), 4) if len(final_summary) > 0 else 1)
        for i, row in final_summary.iterrows():
            m_cols[i % 4].metric(label=f"Energy ({row['Specimen']})", value=f"{row['Energy (J)']:.2f} J")
            
        fig3 = px.line(plot_df, x='corrected_displacement_mm', y='force_n', color='specimen_id',
                       labels={'corrected_displacement_mm': 'Corrected Displacement (mm)', 'force_n': 'Force (N)'},
                       title='Area Under Load-Displacement Curve')
        fig3.update_traces(fill='tozeroy', opacity=0.3)
        st.plotly_chart(fig3, use_container_width=True)
            
        st.header("3. Export Cleaned Data")
        st.markdown("Download the cleaned dataset for downstream analysis or visualization.")
        
        st.subheader("Data Previews")
        for specimen in final_df['specimen_id'].unique():
            with st.expander(f"Preview {specimen}"):
                st.dataframe(final_df[final_df['specimen_id'] == specimen].head(10))
        
        csv = final_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Cleaned Data CSV",
            data=csv,
            file_name='fea_cleaned_instron_data.csv',
            mime='text/csv',
        )
