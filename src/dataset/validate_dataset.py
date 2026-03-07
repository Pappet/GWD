"""
Datensatz-Validierung für Gravitationswellen Training

Dieses Script überprüft die Qualität des generierten Datensatzes und erstellt
umfassende Visualisierungen zur Analyse.

Usage:
    python validate_dataset.py [--data-folder gw_training_data]
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import DATA_DIR_TRAIN, VALIDATION_DATASET_PLOTS_DIR

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import signal
import warnings
warnings.filterwarnings('ignore')


def load_dataset_info(data_folder):
    """Lädt die Labels-CSV und gibt Statistiken zurück."""
    csv_path = os.path.join(data_folder, "labels.csv")
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Konnte {csv_path} nicht finden!")
    
    df = pd.read_csv(csv_path)
    return df


def print_statistics(df):
    """Gibt detaillierte Statistiken über den Datensatz aus."""
    print("\n" + "="*80)
    print("📊 DATENSATZ STATISTIK")
    print("="*80)
    
    # Grundlegende Statistik
    total_samples = len(df)
    signal_samples = df[df['has_signal'] == 1]
    noise_samples = df[df['has_signal'] == 0]
    
    print(f"\n1. ALLGEMEINE ÜBERSICHT:")
    print(f"   {'Gesamt Samples:':<30} {total_samples:>10,}")
    print(f"   {'Samples mit Signal:':<30} {len(signal_samples):>10,} ({len(signal_samples)/total_samples*100:>5.1f}%)")
    print(f"   {'Samples nur Rauschen:':<30} {len(noise_samples):>10,} ({len(noise_samples)/total_samples*100:>5.1f}%)")
    
    # SNR Statistik (nur Signal-Samples)
    if len(signal_samples) > 0:
        print(f"\n2. SNR VERTEILUNG (Signal-Samples):")
        print(f"   {'Minimum SNR:':<30} {signal_samples['snr'].min():>10.4f}")
        print(f"   {'Maximum SNR:':<30} {signal_samples['snr'].max():>10.4f}")
        print(f"   {'Durchschnitt SNR:':<30} {signal_samples['snr'].mean():>10.4f}")
        print(f"   {'Median SNR:':<30} {signal_samples['snr'].median():>10.4f}")
        print(f"   {'Standardabweichung:':<30} {signal_samples['snr'].std():>10.4f}")
        
        # SNR Kategorien
        print(f"\n   SNR Kategorien:")
        print(f"   {'SNR < 5 (sehr schwach):':<30} {(signal_samples['snr'] < 5).sum():>10,} ({(signal_samples['snr'] < 5).sum()/len(signal_samples)*100:>5.1f}%)")
        print(f"   {'SNR 5-10 (schwach):':<30} {((signal_samples['snr'] >= 5) & (signal_samples['snr'] < 10)).sum():>10,} ({((signal_samples['snr'] >= 5) & (signal_samples['snr'] < 10)).sum()/len(signal_samples)*100:>5.1f}%)")
        print(f"   {'SNR 10-15 (mittel):':<30} {((signal_samples['snr'] >= 10) & (signal_samples['snr'] < 15)).sum():>10,} ({((signal_samples['snr'] >= 10) & (signal_samples['snr'] < 15)).sum()/len(signal_samples)*100:>5.1f}%)")
        print(f"   {'SNR 15-20 (stark):':<30} {((signal_samples['snr'] >= 15) & (signal_samples['snr'] < 20)).sum():>10,} ({((signal_samples['snr'] >= 15) & (signal_samples['snr'] < 20)).sum()/len(signal_samples)*100:>5.1f}%)")
        print(f"   {'SNR > 20 (sehr stark):':<30} {(signal_samples['snr'] >= 20).sum():>10,} ({(signal_samples['snr'] >= 20).sum()/len(signal_samples)*100:>5.1f}%)")
        
        # Qualitäts-Check
        print(f"\n   💡 QUALITÄTS-CHECK:")
        if signal_samples['snr'].median() < 5:
            print(f"   ❌ WARNUNG: Median SNR zu niedrig! Signale schwer detektierbar.")
        elif signal_samples['snr'].median() < 8:
            print(f"   ⚠️  ACHTUNG: Median SNR niedrig. Training könnte schwierig sein.")
        elif signal_samples['snr'].median() < 12:
            print(f"   ✅ OK: Median SNR im akzeptablen Bereich.")
        else:
            print(f"   ✅ EXCELLENT: Median SNR optimal für Training!")
    
    # Physikalische Parameter (nur Signal-Samples)
    if len(signal_samples) > 0:
        print(f"\n3. PHYSIKALISCHE PARAMETER:")
        
        # Massen
        print(f"\n   Massen (Sonnenmassen):")
        print(f"   {'Mass 1 - Bereich:':<30} {signal_samples['mass1'].min():>8.1f} - {signal_samples['mass1'].max():.1f}")
        print(f"   {'Mass 1 - Durchschnitt:':<30} {signal_samples['mass1'].mean():>10.1f}")
        print(f"   {'Mass 2 - Bereich:':<30} {signal_samples['mass2'].min():>8.1f} - {signal_samples['mass2'].max():.1f}")
        print(f"   {'Mass 2 - Durchschnitt:':<30} {signal_samples['mass2'].mean():>10.1f}")
        
        # Gesamtmasse und Massenverhältnis
        total_mass = signal_samples['mass1'] + signal_samples['mass2']
        mass_ratio = signal_samples['mass1'] / signal_samples['mass2']
        print(f"   {'Gesamtmasse - Bereich:':<30} {total_mass.min():>8.1f} - {total_mass.max():.1f}")
        print(f"   {'Massenverhältnis (q):':<30} {mass_ratio.min():>8.2f} - {mass_ratio.max():.2f}")
        
        # Distanz
        print(f"\n   Distanz (Megaparsec):")
        print(f"   {'Bereich:':<30} {signal_samples['distance'].min():>8.0f} - {signal_samples['distance'].max():.0f}")
        print(f"   {'Durchschnitt:':<30} {signal_samples['distance'].mean():>10.0f}")
        print(f"   {'Median:':<30} {signal_samples['distance'].median():>10.0f}")
        
        # Merger Time
        print(f"\n   Merger Zeitpunkt (Sekunden):")
        print(f"   {'Bereich:':<30} {signal_samples['merger_time'].min():>8.2f} - {signal_samples['merger_time'].max():.2f}")
        print(f"   {'Durchschnitt:':<30} {signal_samples['merger_time'].mean():>10.2f}")


def validate_samples(df, data_folder, num_check=10):
    """Überprüft zufällige Samples auf Korrektheit."""
    print(f"\n4. SAMPLE VALIDIERUNG:")
    print(f"   Überprüfe {num_check} zufällige Samples...")
    
    issues = []
    
    # Zufällige Samples auswählen
    check_samples = df.sample(min(num_check, len(df)))
    
    for idx, row in check_samples.iterrows():
        filepath = os.path.join(data_folder, row['filename'])
        
        try:
            data = np.load(filepath)
            
            # Check 1: Korrekte Länge?
            expected_length = 16384  # 4s bei 4096 Hz
            if len(data) != expected_length:
                issues.append(f"   ⚠️  {row['filename']}: Falsche Länge ({len(data)} statt {expected_length})")
            
            # Check 2: NaN oder Inf Werte?
            if np.any(np.isnan(data)) or np.any(np.isinf(data)):
                issues.append(f"   ❌ {row['filename']}: Enthält NaN oder Inf Werte!")
            
            # Check 3: Amplitude vernünftig?
            if np.max(np.abs(data)) > 1000:
                issues.append(f"   ⚠️  {row['filename']}: Sehr hohe Amplitude ({np.max(np.abs(data)):.1f})")
            
            # Check 4: Ist es komplett Null?
            if np.all(data == 0):
                issues.append(f"   ❌ {row['filename']}: Komplett Null!")
                
        except Exception as e:
            issues.append(f"   ❌ {row['filename']}: Fehler beim Laden - {e}")
    
    if issues:
        print(f"\n   Gefundene Probleme:")
        for issue in issues:
            print(issue)
    else:
        print(f"   ✅ Alle überprüften Samples sind OK!")


def create_visualizations(df, data_folder, output_folder=VALIDATION_DATASET_PLOTS_DIR):
    """Erstellt umfassende Visualisierungen des Datensatzes."""
    
    os.makedirs(output_folder, exist_ok=True)
    
    signal_samples = df[df['has_signal'] == 1]
    noise_samples = df[df['has_signal'] == 0]
    
    print(f"\n5. VISUALISIERUNGEN:")
    print(f"   Erstelle Plots in '{output_folder}/'...")
    
    # ========== PLOT 1: SNR Verteilung ==========
    fig1, axes1 = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram
    axes1[0].hist(signal_samples['snr'], bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    axes1[0].axvline(signal_samples['snr'].median(), color='red', linestyle='--', 
                     linewidth=2, label=f'Median = {signal_samples["snr"].median():.2f}')
    axes1[0].axvline(signal_samples['snr'].mean(), color='green', linestyle='--', 
                     linewidth=2, label=f'Mean = {signal_samples["snr"].mean():.2f}')
    axes1[0].axvline(5, color='orange', linestyle=':', linewidth=2, alpha=0.5, label='SNR = 5')
    axes1[0].axvline(10, color='purple', linestyle=':', linewidth=2, alpha=0.5, label='SNR = 10')
    axes1[0].set_xlabel('Signal-to-Noise Ratio (SNR)', fontsize=12)
    axes1[0].set_ylabel('Anzahl Samples', fontsize=12)
    axes1[0].set_title('SNR Verteilung', fontsize=14, fontweight='bold')
    axes1[0].legend()
    axes1[0].grid(True, alpha=0.3)
    
    # Box Plot
    axes1[1].boxplot([signal_samples['snr']], labels=['Signal SNR'])
    axes1[1].set_ylabel('SNR', fontsize=12)
    axes1[1].set_title('SNR Box Plot', fontsize=14, fontweight='bold')
    axes1[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, '01_snr_distribution.png'), dpi=150, bbox_inches='tight')
    print(f"   ✅ 01_snr_distribution.png")
    plt.close()
    
    # ========== PLOT 2: Physikalische Parameter ==========
    fig2 = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 3, figure=fig2, hspace=0.3, wspace=0.3)
    
    # Masse 1 vs Masse 2
    ax1 = fig2.add_subplot(gs[0, 0])
    scatter = ax1.scatter(signal_samples['mass1'], signal_samples['mass2'], 
                         c=signal_samples['snr'], cmap='viridis', alpha=0.6, s=10)
    ax1.plot([0, 100], [0, 100], 'r--', alpha=0.3, label='m1 = m2')
    ax1.set_xlabel('Mass 1 [M☉]', fontsize=11)
    ax1.set_ylabel('Mass 2 [M☉]', fontsize=11)
    ax1.set_title('Massenverteilung', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    cbar1 = plt.colorbar(scatter, ax=ax1)
    cbar1.set_label('SNR', fontsize=10)
    
    # Gesamtmasse Distribution
    ax2 = fig2.add_subplot(gs[0, 1])
    total_mass = signal_samples['mass1'] + signal_samples['mass2']
    ax2.hist(total_mass, bins=40, edgecolor='black', alpha=0.7, color='coral')
    ax2.set_xlabel('Gesamtmasse [M☉]', fontsize=11)
    ax2.set_ylabel('Anzahl', fontsize=11)
    ax2.set_title('Gesamtmassen-Verteilung', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Massenverhältnis (q = m1/m2)
    ax3 = fig2.add_subplot(gs[0, 2])
    mass_ratio = signal_samples['mass1'] / signal_samples['mass2']
    ax3.hist(mass_ratio, bins=40, edgecolor='black', alpha=0.7, color='lightgreen')
    ax3.set_xlabel('Massenverhältnis q = m1/m2', fontsize=11)
    ax3.set_ylabel('Anzahl', fontsize=11)
    ax3.set_title('Massenverhältnis-Verteilung', fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # Distanz vs SNR
    ax4 = fig2.add_subplot(gs[1, 0])
    ax4.scatter(signal_samples['distance'], signal_samples['snr'], alpha=0.5, s=10, color='steelblue')
    ax4.set_xlabel('Distanz [Mpc]', fontsize=11)
    ax4.set_ylabel('SNR', fontsize=11)
    ax4.set_title('SNR vs Distanz', fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    # Distanz Distribution
    ax5 = fig2.add_subplot(gs[1, 1])
    ax5.hist(signal_samples['distance'], bins=40, edgecolor='black', alpha=0.7, color='mediumpurple')
    ax5.set_xlabel('Distanz [Mpc]', fontsize=11)
    ax5.set_ylabel('Anzahl', fontsize=11)
    ax5.set_title('Distanz-Verteilung', fontweight='bold')
    ax5.grid(True, alpha=0.3)
    
    # Merger Time Distribution
    ax6 = fig2.add_subplot(gs[1, 2])
    ax6.hist(signal_samples['merger_time'], bins=40, edgecolor='black', alpha=0.7, color='gold')
    ax6.set_xlabel('Merger Zeitpunkt [s]', fontsize=11)
    ax6.set_ylabel('Anzahl', fontsize=11)
    ax6.set_title('Merger Zeitpunkt-Verteilung', fontweight='bold')
    ax6.grid(True, alpha=0.3)
    
    plt.savefig(os.path.join(output_folder, '02_physical_parameters.png'), dpi=150, bbox_inches='tight')
    print(f"   ✅ 02_physical_parameters.png")
    plt.close()
    
    # ========== PLOT 3: Sample Visualisierung (verschiedene SNR) ==========
    # Wähle repräsentative Samples aus
    samples_to_plot = []
    
    # Nur Rauschen
    if len(noise_samples) > 0:
        samples_to_plot.append(('Nur Rauschen', noise_samples.iloc[0]))
    
    # Verschiedene SNR-Bereiche
    snr_ranges = [
        ('Sehr Schwach (SNR < 5)', signal_samples[signal_samples['snr'] < 5]),
        ('Schwach (SNR 5-10)', signal_samples[(signal_samples['snr'] >= 5) & (signal_samples['snr'] < 10)]),
        ('Mittel (SNR 10-15)', signal_samples[(signal_samples['snr'] >= 10) & (signal_samples['snr'] < 15)]),
        ('Stark (SNR 15-20)', signal_samples[(signal_samples['snr'] >= 15) & (signal_samples['snr'] < 20)]),
        ('Sehr Stark (SNR > 20)', signal_samples[signal_samples['snr'] >= 20])
    ]
    
    for label, subset in snr_ranges:
        if len(subset) > 0:
            samples_to_plot.append((label, subset.iloc[0]))
    
    n_plots = len(samples_to_plot)
    fig3, axes3 = plt.subplots(n_plots, 1, figsize=(16, 3*n_plots))
    if n_plots == 1:
        axes3 = [axes3]
    
    for ax, (label, row) in zip(axes3, samples_to_plot):
        filepath = os.path.join(data_folder, row['filename'])
        data = np.load(filepath)
        time = np.linspace(0, 4, len(data))
        
        ax.plot(time, data, linewidth=0.5, color='steelblue')
        
        # Title mit Infos
        if row['has_signal'] == 1:
            title = f"{label}: SNR={row['snr']:.2f} | M1={row['mass1']:.1f}M☉, M2={row['mass2']:.1f}M☉ | D={row['distance']:.0f}Mpc"
        else:
            title = f"{label}"
        
        ax.set_title(title, fontweight='bold', fontsize=11)
        ax.set_xlabel('Zeit [s]', fontsize=10)
        ax.set_ylabel('Strain', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Merger-Zeitpunkt markieren (falls Signal)
        if row['has_signal'] == 1 and row['merger_time'] > 0:
            ax.axvline(row['merger_time'], color='red', linestyle='--', 
                      alpha=0.7, linewidth=2, label=f"Merger @ {row['merger_time']:.2f}s")
            ax.legend(loc='upper left', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, '03_sample_timeseries.png'), dpi=150, bbox_inches='tight')
    print(f"   ✅ 03_sample_timeseries.png")
    plt.close()
    
    # ========== PLOT 4: Spektrogramme ==========
    # Wähle 3 interessante Samples für Spektrogramm
    spec_samples = []
    
    if len(noise_samples) > 0:
        spec_samples.append(('Nur Rauschen', noise_samples.iloc[0]))
    
    if len(signal_samples[signal_samples['snr'] < 10]) > 0:
        spec_samples.append(('Schwaches Signal', 
                           signal_samples[signal_samples['snr'] < 10].iloc[0]))
    
    if len(signal_samples[signal_samples['snr'] >= 15]) > 0:
        spec_samples.append(('Starkes Signal', 
                           signal_samples[signal_samples['snr'] >= 15].iloc[0]))
    
    fig4, axes4 = plt.subplots(len(spec_samples), 2, figsize=(16, 5*len(spec_samples)))
    if len(spec_samples) == 1:
        axes4 = axes4.reshape(1, -1)
    
    for i, (label, row) in enumerate(spec_samples):
        filepath = os.path.join(data_folder, row['filename'])
        data = np.load(filepath)
        fs = 4096
        
        # Zeitreihe
        time = np.linspace(0, 4, len(data))
        axes4[i, 0].plot(time, data, linewidth=0.5, color='steelblue')
        axes4[i, 0].set_xlabel('Zeit [s]', fontsize=10)
        axes4[i, 0].set_ylabel('Strain', fontsize=10)
        
        if row['has_signal'] == 1:
            title_left = f"{label} | SNR={row['snr']:.2f}"
            if row['merger_time'] > 0:
                axes4[i, 0].axvline(row['merger_time'], color='red', 
                                   linestyle='--', alpha=0.7, linewidth=2)
        else:
            title_left = label
        axes4[i, 0].set_title(title_left, fontweight='bold', fontsize=11)
        axes4[i, 0].grid(True, alpha=0.3)
        
        # Spektrogramm
        f, t, Sxx = signal.spectrogram(data, fs, nperseg=256)
        im = axes4[i, 1].pcolormesh(t, f, 10 * np.log10(Sxx + 1e-10), 
                                     shading='gouraud', cmap='viridis')
        axes4[i, 1].set_ylabel('Frequenz [Hz]', fontsize=10)
        axes4[i, 1].set_xlabel('Zeit [s]', fontsize=10)
        axes4[i, 1].set_title('Spektrogramm', fontweight='bold', fontsize=11)
        axes4[i, 1].set_ylim([0, 500])  # Fokus auf GW-relevanten Bereich
        plt.colorbar(im, ax=axes4[i, 1], label='Power [dB]')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, '04_spectrograms.png'), dpi=150, bbox_inches='tight')
    print(f"   ✅ 04_spectrograms.png")
    plt.close()
    
    # ========== PLOT 5: Korrelations-Heatmap ==========
    if len(signal_samples) > 0:
        fig5, ax5 = plt.subplots(figsize=(10, 8))
        
        # Korrelationsmatrix berechnen
        corr_cols = ['snr', 'mass1', 'mass2', 'distance', 'merger_time']
        corr_matrix = signal_samples[corr_cols].corr()
        
        im = ax5.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
        
        # Achsen beschriften
        ax5.set_xticks(range(len(corr_cols)))
        ax5.set_yticks(range(len(corr_cols)))
        ax5.set_xticklabels(corr_cols, rotation=45, ha='right')
        ax5.set_yticklabels(corr_cols)
        
        # Werte in Zellen schreiben
        for i in range(len(corr_cols)):
            for j in range(len(corr_cols)):
                text = ax5.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                              ha="center", va="center", color="black", fontsize=11)
        
        ax5.set_title('Korrelation zwischen Parametern', fontweight='bold', fontsize=14)
        plt.colorbar(im, ax=ax5, label='Korrelationskoeffizient')
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, '05_correlation_matrix.png'), dpi=150, bbox_inches='tight')
        print(f"   ✅ 05_correlation_matrix.png")
        plt.close()
    
    print(f"\n   ✅ Alle Visualisierungen erstellt in '{output_folder}/'")


def main():
    parser = argparse.ArgumentParser(description='Validiere Gravitationswellen Trainingsdatensatz')
    parser.add_argument('--data-folder', type=str, default=DATA_DIR_TRAIN,
                       help='Pfad zum Datensatz-Ordner')
    parser.add_argument('--output-folder', type=str, default=VALIDATION_DATASET_PLOTS_DIR,
                       help='Pfad für die Ausgabe-Plots')
    parser.add_argument('--no-plots', action='store_true',
                       help='Nur Statistiken, keine Plots erstellen')
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("🔍 DATENSATZ VALIDIERUNG - GRAVITATIONSWELLEN TRAINING")
    print("="*80)
    print(f"\nDatensatz-Ordner: {args.data_folder}")
    print(f"Output-Ordner: {args.output_folder}")
    
    try:
        # Labels laden
        df = load_dataset_info(args.data_folder)
        
        # Statistiken ausgeben
        print_statistics(df)
        
        # Samples validieren
        validate_samples(df, args.data_folder)
        
        # Visualisierungen erstellen
        if not args.no_plots:
            create_visualizations(df, args.data_folder, args.output_folder)
        
        # Finale Bewertung
        signal_samples = df[df['has_signal'] == 1]
        
        print("\n" + "="*80)
        print("📋 FINALE BEWERTUNG")
        print("="*80)
        
        issues = []
        warnings_list = []
        
        # Check 1: SNR
        if len(signal_samples) > 0:
            median_snr = signal_samples['snr'].median()
            if median_snr < 5:
                issues.append("❌ KRITISCH: Median SNR < 5 - Signale zu schwach!")
            elif median_snr < 8:
                warnings_list.append("⚠️  WARNUNG: Median SNR < 8 - Training könnte schwierig sein")
            else:
                print("✅ SNR: Optimal für Training")
        
        # Check 2: Balance
        balance = len(signal_samples) / len(df) * 100
        if balance < 40 or balance > 60:
            warnings_list.append(f"⚠️  WARNUNG: Unbalanciert ({balance:.1f}% Signale)")
        else:
            print(f"✅ Balance: Gut balanciert ({balance:.1f}% Signale)")
        
        # Check 3: Diversität
        if len(signal_samples) > 0:
            mass_range = signal_samples['mass1'].max() - signal_samples['mass1'].min()
            if mass_range < 20:
                warnings_list.append("⚠️  WARNUNG: Geringe Massen-Diversität")
            else:
                print("✅ Diversität: Gute Parameter-Abdeckung")
        
        # Ausgabe der Warnings und Errors
        if warnings_list:
            print("\nWarnungen:")
            for w in warnings_list:
                print(f"  {w}")
        
        if issues:
            print("\nKritische Probleme:")
            for issue in issues:
                print(f"  {issue}")
            print("\n❌ DATENSATZ NICHT GEEIGNET FÜR TRAINING!")
            return 1
        else:
            print("\n" + "="*80)
            print("✅ DATENSATZ IST BEREIT FÜR TRAINING!")
            print("="*80)
            return 0
            
    except Exception as e:
        print(f"\n❌ FEHLER: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())