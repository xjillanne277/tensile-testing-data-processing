import streamlit as st
import pandas as pd
import numpy as np
import scipy.integrate
import matplotlib.pyplot as plt
import io

st.set_page_config(page_title="Instron Data Processor", layout="wide")

st.title("Instron Mechanical Data Processing - NMT Plastics")

st.markdown("""
Upload raw Instron CSV files to parse, clean, and visualize mechanical testing data for short-fiber glass-filled plastics. 
Automatically zeroes baselines, calculates Energy Absorption (J), and provides downloadable CSVs formatted for FEA.
""")

def process_file(uploaded_file, orientation):
    lines = uploaded_file.getvalue().decode('utf-8').splitlines()
    
    # Find start indices of data blocks
    data_starts = []
    for i, line in enumerate(lines):
        if 'Time,Displacement,Force' in line:
            data_starts.append(i)
            
    all_data = []
    summary = []
    
    for i, start_idx in enumerate(data_starts):
        coupon_id = f"Coupon_{i+1:02d}"
        
        end_idx = data_starts[i+1] if i+1 < len(data_starts) else len(lines)
        data_lines = lines[start_idx+2:end_idx]
        data_lines = [l.strip() for l in data_lines if l.strip() and not l.startswith('Results')]
        data_lines = [l for l in data_lines if len(l.split(',')) >= 8]
        
        if not data_lines:
            continue
            
        df = pd.DataFrame([l.split(',')[1:8] for l in data_lines], dtype=float)
        df.columns = ['time_s', 'displacement_mm', 'force_n', 'tensile_strain_percent', 
                      'tensile_stress_mpa', 'tensile_displacement_mm', 'corrected_displacement_mm']
        
        # Zeroing/Baseline Calibration
        df['force_n'] = df['force_n'] - df['force_n'].iloc[0]
        df['tensile_stress_mpa'] = df['tensile_stress_mpa'] - df['tensile_stress_mpa'].iloc[0]
        
        # Noise Reduction (remove negative strain)
        df = df[df['tensile_strain_percent'] >= 0.0]
        
        df['orientation'] = orientation
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
            'Orientation': orientation,
            'Peak Stress (MPa)': peak_stress,
            'Modulus (GPa)': modulus_gpa,
            'Energy (J)': energy_j
        })
        
    if all_data:
        return pd.concat(all_data, ignore_index=True), pd.DataFrame(summary)
    else:
        return None, None

uploaded_files = st.file_uploader("Upload Raw Instron CSV(s)", type="csv", accept_multiple_files=True)

if uploaded_files:
    dfs = []
    summaries = []
    
    st.sidebar.header("Configuration")
    
    for file in uploaded_files:
        orientation = st.sidebar.selectbox(f"Orientation for {file.name}", 
                                           options=["0°", "30°", "45°", "60°", "90°"], 
                                           key=file.name)
        df, summary = process_file(file, orientation)
        if df is not None:
            dfs.append(df)
            summaries.append(summary)
            
    if dfs:
        final_df = pd.concat(dfs, ignore_index=True)
        final_summary = pd.concat(summaries, ignore_index=True)
        
        st.header("1. Summary Metrics")
        st.dataframe(final_summary)
        
        mean_summary = final_summary.groupby('Orientation').mean(numeric_only=True).reset_index()
        st.subheader("Mean Values by Orientation")
        st.dataframe(mean_summary)
        
        st.header("2. Visualizations")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Stress vs. Strain")
            fig1, ax1 = plt.subplots(figsize=(8, 6))
            for (ori, coupon), group in final_df.groupby(['orientation', 'coupon_id']):
                ax1.plot(group['tensile_strain_percent'], group['tensile_stress_mpa'], label=f"{ori} - {coupon}")
            
            # Too many labels might clutter, so only show unique orientations in legend if many coupons
            handles, labels = plt.gca().get_legend_handles_labels()
            by_label = dict(zip([l.split(' - ')[0] for l in labels], handles))
            ax1.legend(by_label.values(), by_label.keys())
            ax1.set_xlabel('Tensile Strain (%)')
            ax1.set_ylabel('Tensile Stress (MPa)')
            ax1.grid(True, linestyle='--', alpha=0.7)
            st.pyplot(fig1)
            
        with col2:
            st.subheader("Force vs. Displacement")
            fig2, ax2 = plt.subplots(figsize=(8, 6))
            for (ori, coupon), group in final_df.groupby(['orientation', 'coupon_id']):
                ax2.plot(group['corrected_displacement_mm'], group['force_n'], label=f"{ori} - {coupon}")
            
            handles, labels = plt.gca().get_legend_handles_labels()
            by_label = dict(zip([l.split(' - ')[0] for l in labels], handles))
            ax2.legend(by_label.values(), by_label.keys())
            ax2.set_xlabel('Corrected Displacement (mm)')
            ax2.set_ylabel('Force (N)')
            ax2.grid(True, linestyle='--', alpha=0.7)
            st.pyplot(fig2)
            
        st.header("3. FEA Export")
        st.markdown("Download the cleaned dataset grouped by orientation for use in Digimat or ANSYS/Abaqus plastic stress-strain tables.")
        
        st.dataframe(final_df.head(10))
        
        csv = final_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Cleaned Data CSV",
            data=csv,
            file_name='fea_cleaned_instron_data.csv',
            mime='text/csv',
        )
