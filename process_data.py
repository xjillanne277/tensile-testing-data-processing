import pandas as pd
import numpy as np
import scipy.integrate

def process_instron_csv(filepath, orientation='0deg'):
    with open(filepath, 'r') as f:
        lines = f.readlines()
        
    # Find start indices of data blocks
    data_starts = []
    for i, line in enumerate(lines):
        if 'Time,Displacement,Force' in line:
            data_starts.append(i)
            
    all_data = []
    for i, start_idx in enumerate(data_starts):
        specimen_id = f"Specimen_{i+1:02d}"
        
        # Read until next start_idx or end of file, skip the units row
        end_idx = data_starts[i+1] if i+1 < len(data_starts) else len(lines)
        
        # The data starts at start_idx + 2 (skip headers and units)
        # But we also need to avoid trailing empty lines
        data_lines = lines[start_idx+2:end_idx]
        data_lines = [l.strip() for l in data_lines if l.strip() and not l.startswith('Results')]
        data_lines = [l for l in data_lines if len(l.split(',')) >= 8]
        
        if not data_lines:
            continue
            
        df = pd.DataFrame([l.split(',')[1:8] for l in data_lines], dtype=float)
        df.columns = ['time_s', 'displacement_mm', 'force_n', 'tensile_strain_percent', 
                      'tensile_stress_mpa', 'tensile_displacement_mm', 'corrected_displacement_mm']
        
        # Zeroing/Baseline Calibration
        # Find first non-negative strain/force to zero
        df['force_n'] = df['force_n'] - df['force_n'].iloc[0]
        df['tensile_stress_mpa'] = df['tensile_stress_mpa'] - df['tensile_stress_mpa'].iloc[0]
        
        # Noise Reduction (remove negative strain)
        df = df[df['tensile_strain_percent'] >= 0.0]
        
        df['orientation'] = orientation
        df['specimen_id'] = specimen_id
        
        # Calculate Energy Absorption (J)
        # force_n is in N, corrected_displacement_mm is in mm. 1 N*mm = 0.001 J
        # Trapezoidal rule
        energy_j = scipy.integrate.trapezoid(df['force_n'], df['corrected_displacement_mm']) * 0.001
        
        # Calculate Modulus (GPa) - estimate from linear region (e.g. 0.05% to 0.25% strain)
        linear_region = df[(df['tensile_strain_percent'] > 0.05) & (df['tensile_strain_percent'] < 0.25)]
        if len(linear_region) > 1:
            slope, _ = np.polyfit(linear_region['tensile_strain_percent']/100.0, linear_region['tensile_stress_mpa'], 1)
            modulus_gpa = slope / 1000.0
        else:
            modulus_gpa = np.nan
            
        peak_stress = df['tensile_stress_mpa'].max()
        
        all_data.append({
            'df': df,
            'orientation': orientation,
            'specimen_id': specimen_id,
            'energy_j': energy_j,
            'peak_stress': peak_stress,
            'modulus_gpa': modulus_gpa
        })
        
    return all_data

if __name__ == "__main__":
    filepath = "/usr/local/google/home/jiillanne/jetski-projects/Instron Sample Data - jilltest-gray_1.csv"
    results = process_instron_csv(filepath)
    
    print("--- SUMMARY ---")
    summary = []
    for r in results:
        summary.append({
            'Specimen': r['specimen_id'],
            'Peak Stress (MPa)': r['peak_stress'],
            'Modulus (GPa)': r['modulus_gpa'],
            'Energy (J)': r['energy_j']
        })
        
    sdf = pd.DataFrame(summary)
    print(f"Mean Peak Stress: {sdf['Peak Stress (MPa)'].mean():.2f}")
    print(f"Mean Modulus: {sdf['Modulus (GPa)'].mean():.2f}")
    print(f"Mean Energy: {sdf['Energy (J)'].mean():.4f}")
    
    print("\n--- PREVIEW ---")
    print(results[0]['df'].head(10).to_csv(index=False))
