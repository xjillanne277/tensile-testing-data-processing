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
        coupon_id = f"Coupon_{i+1:02d}"
        
        end_idx = data_starts[i+1] if i+1 < len(data_starts) else len(lines)
        data_lines = lines[start_idx+2:end_idx]
        data_lines = [l.strip() for l in data_lines if l.strip() and not l.startswith('Results')]
        data_lines = [l for l in data_lines if len(l.split(',')) >= 8]
        
        if not data_lines:
            raise ValueError(f"No valid data rows found for {coupon_id}.")
        
        parsed_data = []
        for l in data_lines:
            parts = l.split(',')
            if len(parts) >= 8:
                row = [float(x.strip().strip('"').strip("'")) for x in parts[1:8]]
                parsed_data.append(row)
        
        if not parsed_data:
            raise ValueError(f"Failed to parse numerical data for {coupon_id}.")
            
        df = pd.DataFrame(parsed_data, dtype=float)
        df.columns = ['time_s', 'displacement_mm', 'force_n', 'tensile_strain_percent', 
                      'tensile_stress_mpa', 'tensile_displacement_mm', 'corrected_displacement_mm']
        
        # Zeroing/Baseline Calibration
        df['force_n'] = df['force_n'] - df['force_n'].iloc[0]
        df['tensile_stress_mpa'] = df['tensile_stress_mpa'] - df['tensile_stress_mpa'].iloc[0]
        
        # Noise Reduction (remove negative strain)
        df = df[df['tensile_strain_percent'] >= 0.0]
        
        df['coupon_id'] = coupon_id
        
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
        
        all_data.append(df)
        summary.append({
            'Coupon': coupon_id,
            'Peak Stress (MPa)': peak_stress,
            'Modulus (GPa)': modulus_gpa,
            'Energy (J)': energy_j
        })
        
    return pd.concat(all_data, ignore_index=True), pd.DataFrame(summary)

uploaded_files = st.file_uploader("Upload Raw Instron CSV(s)", type="csv", accept_multiple_files=True)

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
        st.dataframe(final_summary)
        
        st.header("2. Visualizations")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Stress vs. Strain")
            fig1 = px.line(final_df, x='tensile_strain_percent', y='tensile_stress_mpa', color='coupon_id',
                           labels={'tensile_strain_percent': 'Tensile Strain (%)', 'tensile_stress_mpa': 'Tensile Stress (MPa)'},
                           title='Stress vs Strain')
            st.plotly_chart(fig1, use_container_width=True)
            
        with col2:
            st.subheader("Force vs. Displacement")
            fig2 = px.line(final_df, x='corrected_displacement_mm', y='force_n', color='coupon_id',
                           labels={'corrected_displacement_mm': 'Corrected Displacement (mm)', 'force_n': 'Force (N)'},
                           title='Load vs Displacement')
            st.plotly_chart(fig2, use_container_width=True)
            
        st.header("3. Export Cleaned Data")
        st.markdown("Download the cleaned dataset for downstream analysis or visualization.")
        
        st.dataframe(final_df.head(10))
        
        csv = final_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Cleaned Data CSV",
            data=csv,
            file_name='fea_cleaned_instron_data.csv',
            mime='text/csv',
        )
