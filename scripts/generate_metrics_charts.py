#!/usr/bin/env python3
"""
Script para generar gráficos de métricas del clúster Docker.

Este script lee el archivo CSV generado por monitor_cluster_metrics.ps1
y genera gráficos para incluir en el informe técnico.

Uso:
    python generate_metrics_charts.py <metrics_csv_file>
    python generate_metrics_charts.py DATA/metrics/cluster_metrics_20251206_120000.csv
"""

import sys
import os
from pathlib import Path
from datetime import datetime

# Agregar el directorio Tools/src al path
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent.parent
tools_src = project_root / "Tools" / "src"
sys.path.insert(0, str(tools_src))

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
import numpy as np


def load_metrics(csv_path: str) -> pd.DataFrame:
    """Carga el archivo CSV de métricas."""
    df = pd.read_csv(csv_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df


def create_cpu_chart(df: pd.DataFrame, output_dir: Path) -> str:
    """Genera gráfico de uso de CPU por contenedor."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    containers = df['container'].unique()
    colors = plt.cm.Set2(np.linspace(0, 1, len(containers)))
    
    for container, color in zip(containers, colors):
        container_data = df[df['container'] == container]
        # Nombre corto del contenedor
        short_name = container.replace('climaxtreme-', '')
        ax.plot(container_data['timestamp'], container_data['cpu_percent'], 
                label=short_name, color=color, linewidth=1.5)
    
    ax.set_xlabel('Tiempo', fontsize=12)
    ax.set_ylabel('CPU (%)', fontsize=12)
    ax.set_title('Uso de CPU por Contenedor durante Procesamiento', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    output_path = output_dir / 'cpu_usage_chart.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return str(output_path)


def create_memory_chart(df: pd.DataFrame, output_dir: Path) -> str:
    """Genera gráfico de uso de memoria por contenedor."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    containers = df['container'].unique()
    colors = plt.cm.Set2(np.linspace(0, 1, len(containers)))
    
    for container, color in zip(containers, colors):
        container_data = df[df['container'] == container]
        short_name = container.replace('climaxtreme-', '')
        ax.plot(container_data['timestamp'], container_data['mem_usage_mb'], 
                label=short_name, color=color, linewidth=1.5)
    
    ax.set_xlabel('Tiempo', fontsize=12)
    ax.set_ylabel('Memoria (MB)', fontsize=12)
    ax.set_title('Uso de Memoria por Contenedor durante Procesamiento', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    output_path = output_dir / 'memory_usage_chart.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return str(output_path)


def create_combined_dashboard(df: pd.DataFrame, output_dir: Path) -> str:
    """Genera un dashboard combinado con múltiples gráficos."""
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    containers = df['container'].unique()
    colors = plt.cm.Set2(np.linspace(0, 1, len(containers)))
    color_map = {c: color for c, color in zip(containers, colors)}
    
    # 1. CPU Usage (top left)
    ax1 = fig.add_subplot(gs[0, 0])
    for container in containers:
        container_data = df[df['container'] == container]
        short_name = container.replace('climaxtreme-', '')
        ax1.plot(container_data['timestamp'], container_data['cpu_percent'], 
                label=short_name, color=color_map[container], linewidth=1.5)
    ax1.set_xlabel('Tiempo')
    ax1.set_ylabel('CPU (%)')
    ax1.set_title('Uso de CPU', fontweight='bold')
    ax1.legend(loc='upper right', fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
    
    # 2. Memory Usage (top right)
    ax2 = fig.add_subplot(gs[0, 1])
    for container in containers:
        container_data = df[df['container'] == container]
        short_name = container.replace('climaxtreme-', '')
        ax2.plot(container_data['timestamp'], container_data['mem_usage_mb'], 
                label=short_name, color=color_map[container], linewidth=1.5)
    ax2.set_xlabel('Tiempo')
    ax2.set_ylabel('Memoria (MB)')
    ax2.set_title('Uso de Memoria', fontweight='bold')
    ax2.legend(loc='upper right', fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
    
    # 3. Average CPU by Container (middle left) - Bar chart
    ax3 = fig.add_subplot(gs[1, 0])
    avg_cpu = df.groupby('container')['cpu_percent'].mean()
    short_names = [c.replace('climaxtreme-', '') for c in avg_cpu.index]
    bars = ax3.bar(short_names, avg_cpu.values, color=[color_map[c] for c in avg_cpu.index])
    ax3.set_xlabel('Contenedor')
    ax3.set_ylabel('CPU Promedio (%)')
    ax3.set_title('CPU Promedio por Contenedor', fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')
    # Agregar etiquetas de valor
    for bar, val in zip(bars, avg_cpu.values):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                f'{val:.1f}%', ha='center', va='bottom', fontsize=9)
    
    # 4. Average Memory by Container (middle right) - Bar chart
    ax4 = fig.add_subplot(gs[1, 1])
    avg_mem = df.groupby('container')['mem_usage_mb'].mean()
    short_names = [c.replace('climaxtreme-', '') for c in avg_mem.index]
    bars = ax4.bar(short_names, avg_mem.values, color=[color_map[c] for c in avg_mem.index])
    ax4.set_xlabel('Contenedor')
    ax4.set_ylabel('Memoria Promedio (MB)')
    ax4.set_title('Memoria Promedio por Contenedor', fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, ha='right')
    for bar, val in zip(bars, avg_mem.values):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, 
                f'{val:.0f}MB', ha='center', va='bottom', fontsize=9)
    
    # 5. Network I/O (bottom left)
    ax5 = fig.add_subplot(gs[2, 0])
    for container in containers:
        container_data = df[df['container'] == container]
        short_name = container.replace('climaxtreme-', '')
        ax5.plot(container_data['timestamp'], container_data['net_io_rx_mb'], 
                label=f'{short_name} (RX)', color=color_map[container], linewidth=1.5, linestyle='-')
        ax5.plot(container_data['timestamp'], container_data['net_io_tx_mb'], 
                color=color_map[container], linewidth=1.5, linestyle='--', alpha=0.7)
    ax5.set_xlabel('Tiempo')
    ax5.set_ylabel('Network I/O (MB)')
    ax5.set_title('Network I/O (línea sólida=RX, punteada=TX)', fontweight='bold')
    ax5.grid(True, alpha=0.3)
    ax5.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    plt.setp(ax5.xaxis.get_majorticklabels(), rotation=45)
    
    # 6. Block I/O (bottom right)
    ax6 = fig.add_subplot(gs[2, 1])
    for container in containers:
        container_data = df[df['container'] == container]
        short_name = container.replace('climaxtreme-', '')
        ax6.plot(container_data['timestamp'], container_data['block_io_read_mb'], 
                label=f'{short_name}', color=color_map[container], linewidth=1.5, linestyle='-')
        ax6.plot(container_data['timestamp'], container_data['block_io_write_mb'], 
                color=color_map[container], linewidth=1.5, linestyle='--', alpha=0.7)
    ax6.set_xlabel('Tiempo')
    ax6.set_ylabel('Block I/O (MB)')
    ax6.set_title('Disk I/O (línea sólida=Read, punteada=Write)', fontweight='bold')
    ax6.grid(True, alpha=0.3)
    ax6.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    plt.setp(ax6.xaxis.get_majorticklabels(), rotation=45)
    
    # Título general
    fig.suptitle('Dashboard de Métricas del Clúster - ClimaXtreme', 
                fontsize=16, fontweight='bold', y=0.98)
    
    output_path = output_dir / 'metrics_dashboard.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return str(output_path)


def create_statistics_table(df: pd.DataFrame, output_dir: Path) -> str:
    """Genera una tabla de estadísticas en formato Markdown."""
    stats = []
    
    for container in df['container'].unique():
        container_data = df[df['container'] == container]
        short_name = container.replace('climaxtreme-', '')
        
        stats.append({
            'Contenedor': short_name,
            'CPU Promedio (%)': f"{container_data['cpu_percent'].mean():.2f}",
            'CPU Máximo (%)': f"{container_data['cpu_percent'].max():.2f}",
            'RAM Promedio (MB)': f"{container_data['mem_usage_mb'].mean():.0f}",
            'RAM Máximo (MB)': f"{container_data['mem_usage_mb'].max():.0f}",
            'RAM % Promedio': f"{container_data['mem_percent'].mean():.1f}",
        })
    
    stats_df = pd.DataFrame(stats)
    
    # Guardar como CSV
    csv_path = output_dir / 'statistics_summary.csv'
    stats_df.to_csv(csv_path, index=False)
    
    # Guardar como Markdown
    md_path = output_dir / 'statistics_summary.md'
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write("# Estadísticas del Clúster - ClimaXtreme\n\n")
        f.write(f"Generado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("## Resumen por Contenedor\n\n")
        f.write(stats_df.to_markdown(index=False))
        f.write("\n\n## Notas\n\n")
        f.write("- CPU: Porcentaje del total de CPUs disponibles\n")
        f.write("- RAM: Memoria utilizada en MB\n")
        f.write("- Las métricas se capturaron durante la ejecución del pipeline de procesamiento\n")
    
    return str(md_path)


def main():
    """Función principal."""
    if len(sys.argv) < 2:
        # Buscar el archivo CSV más reciente
        metrics_dir = project_root / "DATA" / "metrics"
        if metrics_dir.exists():
            csv_files = list(metrics_dir.glob("cluster_metrics_*.csv"))
            if csv_files:
                csv_path = max(csv_files, key=lambda p: p.stat().st_mtime)
                print(f"📁 Usando archivo más reciente: {csv_path}")
            else:
                print("❌ No se encontraron archivos de métricas.")
                print("   Ejecute primero: .\\scripts\\windows\\monitor_cluster_metrics.ps1")
                sys.exit(1)
        else:
            print("❌ Directorio de métricas no encontrado.")
            print("   Ejecute primero: .\\scripts\\windows\\monitor_cluster_metrics.ps1")
            sys.exit(1)
    else:
        csv_path = Path(sys.argv[1])
        if not csv_path.exists():
            print(f"❌ Archivo no encontrado: {csv_path}")
            sys.exit(1)
    
    print("=" * 60)
    print("  GENERADOR DE GRÁFICOS DE MÉTRICAS - CLIMAXTREME")
    print("=" * 60)
    print()
    
    # Cargar datos
    print("📊 Cargando datos de métricas...")
    df = load_metrics(str(csv_path))
    print(f"   - {len(df)} registros cargados")
    print(f"   - {df['container'].nunique()} contenedores")
    print(f"   - Período: {df['timestamp'].min()} a {df['timestamp'].max()}")
    print()
    
    # Crear directorio de salida
    output_dir = csv_path.parent / "charts"
    output_dir.mkdir(exist_ok=True)
    
    # Generar gráficos
    print("📈 Generando gráficos...")
    
    cpu_chart = create_cpu_chart(df, output_dir)
    print(f"   ✅ CPU Usage: {cpu_chart}")
    
    mem_chart = create_memory_chart(df, output_dir)
    print(f"   ✅ Memory Usage: {mem_chart}")
    
    dashboard = create_combined_dashboard(df, output_dir)
    print(f"   ✅ Dashboard: {dashboard}")
    
    stats_table = create_statistics_table(df, output_dir)
    print(f"   ✅ Statistics: {stats_table}")
    
    print()
    print("=" * 60)
    print("✅ Gráficos generados exitosamente!")
    print()
    print("📁 Archivos creados en:", output_dir)
    print("   - cpu_usage_chart.png")
    print("   - memory_usage_chart.png")
    print("   - metrics_dashboard.png")
    print("   - statistics_summary.csv")
    print("   - statistics_summary.md")
    print()
    print("💡 Tip: Copie metrics_dashboard.png al directorio Informe/ para")
    print("   incluirlo en el documento LaTeX.")
    print("=" * 60)


if __name__ == "__main__":
    main()
