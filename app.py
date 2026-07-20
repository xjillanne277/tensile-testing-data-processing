import streamlit as st
import pandas as pd
import numpy as np
import scipy.integrate
import plotly.express as px
import plotly.graph_objects as go
import io

st.set_page_config(page_title="Advanced Plastics Testing Processor", layout="wide")

st.title("Advanced Plastics Tensile Testing & FEA Export")
st.markdown("""
Upload raw Instron CSV files to parse, clean, and visualize mechanical testing data. 
Calculates Modulus, Yield Stress, UTS, Strain at Break, and Energy Absorption.
""")

def process_file(uploaded_file, orientation):
    lines = uploaded_file.getvalue().decode('utf-8').splitlines()
    
    data_starts = [i for i, line in enumerate(lines) if 'Time,Displacement,Force' in line]
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
        
        df['force_n'] -= df['force_n'].iloc[0]
        df['tensile_stress_mpa'] -= df['tensile_stress_mpa'].iloc[0]
        df = df[df['tensile_strain_percent'] >= 0.0]
        
        if len(df) == 0:
            raise ValueError(f"No positive strain data found for {coupon_id}.")
            
        df['orientation'] = orientation
        df['coupon_id'] = f"{coupon_id}_{orientation}"
        
        # Energy Absorption (Joules)
        energy_j = scipy.integrate.trapezoid(df['force_n'], df['corrected_displacement_mm']) * 0.001
        
        # UTS & Strain at Break
        uts_mpa = df['tensile_stress_mpa'].max()
        strain_at_break = df['tensile_strain_percent'].max()
        
        # Modulus & Yield
        linear_region = df[(df['tensile_strain_percent'] > 0.05) & (df['tensile_strain_percent'] < 0.25)]
        yield_stress = np.nan
        if len(linear_region) > 1:
            slope, intercept = np.polyfit(linear_region['tensile_strain_percent']/100.0, linear_region['tensile_stress_mpa'], 1)
            modulus_gpa = slope / 1000.0
            
            # 0.2% offset yield
            offset_stress = slope * (df['tensile_strain_percent']/100.0 - 0.002)
            diff = df['tensile_stress_mpa'] - offset_stress
            yield_points = df[diff < 0]
            if not yield_points.empty and yield_points.index[0] > linear_region.index[-1]:
                yield_stress = yield_points.iloc[0]['tensile_stress_mpa']
            else:
                yield_stress = uts_mpa
        else:
            modulus_gpa = np.nan
            yield_stress = np.nan
            
        all_data.append(df)
        summary.append({
            'Coupon': df['coupon_id'].iloc[0],
            'Orientation': orientation,
            'Yield Stress (MPa)': yield_stress,
            'UTS (MPa)': uts_mpa,
            'Modulus (GPa)': modulus_gpa,
            'Strain at Break (%)': strain_at_break,
            'Energy (J)': energy_j
        })
        
    return pd.concat(all_data, ignore_index=True), pd.DataFrame(summary)

uploaded_files = st.file_uploader("Upload Raw Instron CSV(s)", type="csv", accept_multiple_files=True)

if uploaded_files:
    dfs, summaries = [], []
    
    st.sidebar.header("Configuration")
    for file in uploaded_files:
        orientation = st.sidebar.selectbox(f"Orientation for {file.name}", options=["0°", "45°", "90°"], key=file.name)
        try:
            df, summary = process_file(file, orientation)
            dfs.append(df)
            summaries.append(summary)
        except Exception as e:
            st.error(f"Error processing {file.name}: {e}")
            
    if dfs:
        final_df = pd.concat(dfs, ignore_index=True)
        final_summary = pd.concat(summaries, ignore_index=True)
        
        st.header("1. Mechanical Metrics")
        
        cols = st.columns(min(len(final_summary), 4) if len(final_summary) > 0 else 1)
        for i, row in final_summary.iterrows():
            with cols[i % len(cols)]:
                st.metric(label=f"{row['Coupon']} Energy", value=f"{row['Energy (J)']:.2f} J", delta=f"UTS: {row['UTS (MPa)']:.1f} MPa", delta_color="off")
        
        st.dataframe(final_summary)
        
        st.header("2. Visualizations (Anisotropy & Energy)")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Stress vs. Strain (with Yield & UTS)")
            fig1 = px.line(final_df, x='tensile_strain_percent', y='tensile_stress_mpa', color='coupon_id',
                           line_dash='orientation',
                           labels={'tensile_strain_percent': 'Tensile Strain (%)', 'tensile_stress_mpa': 'Tensile Stress (MPa)'},
                           title='Stress vs Strain')
            
            for i, row in final_summary.iterrows():
                fig1.add_scatter(x=[row['Strain at Break (%)']], y=[row['UTS (MPa)']], mode='markers', 
                                 marker=dict(symbol='x', size=10, color='red'), name=f"UTS {row['Coupon']}")
            st.plotly_chart(fig1, use_container_width=True)
            
        with col2:
            st.subheader("Energy Absorption (Load vs. Displacement)")
            fig2 = px.line(final_df, x='corrected_displacement_mm', y='force_n', color='coupon_id',
                           line_dash='orientation',
                           labels={'corrected_displacement_mm': 'Displacement (mm)', 'force_n': 'Force (N)'},
                           title='Filled Area = Energy Absorption')
            
            fig2.update_traces(fill='tozeroy', opacity=0.3)
            for i, row in final_summary.iterrows():
                c_df = final_df[final_df['coupon_id'] == row['Coupon']]
                if len(c_df) > 0:
                    max_x = c_df['corrected_displacement_mm'].max()
                    max_y = c_df['force_n'].max()
                    fig2.add_annotation(x=max_x/2, y=max_y/2, text=f"{row['Energy (J)']:.2f} J", showarrow=False, font=dict(size=14), bgcolor="rgba(255,255,255,0.7)")
            
            st.plotly_chart(fig2, use_container_width=True)
            
        st.header("3. FEA Export")
        st.markdown("Download the extracted metrics summary or the full multi-curve dataset.")
        
        col_dl1, col_dl2 = st.columns(2)
        with col_dl1:
            csv_sum = final_summary.to_csv(index=False).encode('utf-8')
            st.download_button(label="Download Summary Metrics (CSV)", data=csv_sum, file_name='fea_summary_metrics.csv', mime='text/csv')
            
        with col_dl2:
            csv_full = final_df.to_csv(index=False).encode('utf-8')
            st.download_button(label="Download FEA Multi-Curve Dataset (CSV)", data=csv_full, file_name='fea_full_curves.csv', mime='text/csv')
