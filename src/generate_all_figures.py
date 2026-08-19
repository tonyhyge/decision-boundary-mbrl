"""
Unified Paper Figure Generator for IEEE Transactions / Journal Submission.
Generates:
1. figures/conceptual_framework.drawio (Draw.io source XML)
2. figures/conceptual_framework.pdf & .png (Vector diagram)
3. figures/stage0_matched_pairs_comparison.pdf & .png
4. figures/stage0_scatter_corr_value_vs_metrics.pdf & .png
5. figures/stage0_boundary_discontinuity_rd.pdf & .png
6. figures/stage3_budget_recovery_headroom.pdf & .png
7. figures/stage3_ranker_auc_distribution.pdf & .png
8. figures/stage4_learned_fidelity_metrics.pdf & .png
9. figures/stage4_data_scaling_curves.pdf & .png
10. figures/portability_final_comparison.pdf & .png
11. figures/stage5_cartpole_external_validity.pdf & .png
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as mlines
from matplotlib.gridspec import GridSpec

# Set publication style
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'STIXGeneral', 'serif'],
    'mathtext.fontset': 'stix',
    'font.size': 9,
    'axes.labelsize': 10,
    'axes.titlesize': 10,
    'legend.fontsize': 8,
    'xtick.labelsize': 8.5,
    'ytick.labelsize': 8.5,
    'figure.titlesize': 11,
    'figure.autolayout': False,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'lines.linewidth': 1.6,
    'grid.alpha': 0.4,
    'grid.linestyle': '--'
})

OUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../figures'))
RES_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../results'))
os.makedirs(OUT_DIR, exist_ok=True)

# Color Palette (ColorBrewer & IEEE Publication Palette)
NAVY = '#1f4e79'
CORAL = '#d9534f'
EMERALD = '#2e7d32'
AMBER = '#f0ad4e'
PURPLE = '#6f42c1'
STEEL = '#4682b4'
GRAY = '#6c757d'
DARK = '#212529'


# ==============================================================================
# 1. DRAW.IO XML GENERATOR & VECTOR RENDERING OF FIGURE 1
# ==============================================================================

def generate_drawio_xml():
    drawio_path = os.path.join(OUT_DIR, 'conceptual_framework.drawio')
    
    # SVG Vector Icon Definitions (URL-encoded for robust cross-platform Draw.io rendering)
    import urllib.parse
    
    def enc_svg(s):
        return "data:image/svg+xml," + urllib.parse.quote(s)
    
    # 1. Robot Agent Icon (Navy)
    robot_svg = enc_svg('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="#1f4e79"><path d="M12 2a2 2 0 0 1 2 2v1h4a2 2 0 0 1 2 2v10a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V7a2 2 0 0 1 2-2h4V4a2 2 0 0 1 2-2zm-3 8a1.5 1.5 0 1 0 0 3 1.5 1.5 0 0 0 0-3zm6 0a1.5 1.5 0 1 0 0 3 1.5 1.5 0 0 0 0-3zm-5 5a1 1 0 0 0 0 2h4a1 1 0 0 0 0-2H10z"/></svg>')
    
    # 2. Safety Shield Icon (Green)
    shield_svg = enc_svg('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="#2e7d32"><path d="M12 1L3 5v6c0 5.55 3.84 10.74 9 12 5.16-1.26 9-6.45 9-12V5l-9-4zm-2 15l-4-4 1.41-1.41L10 13.17l6.59-6.59L18 8l-8 8z"/></svg>')
    
    # 3. Hazard Warning Triangle (Red)
    warning_svg = enc_svg('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="#c62828"><path d="M1 21h22L12 2 1 21zm12-3h-2v-2h2v2zm0-4h-2v-4h2v4z"/></svg>')
    
    # 4. Speedometer Gauge Icon (Teal/Blue)
    gauge_svg = enc_svg('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="#00838f"><path d="M12 4a8 8 0 0 0-8 8c0 2.21.89 4.21 2.34 5.66l1.42-1.42A5.96 5.96 0 0 1 6 12a6 6 0 0 1 12 0c0 1.66-.67 3.16-1.76 4.24l1.42 1.42A7.96 7.96 0 0 0 20 12a8 8 0 0 0-8-8zm-1 4v4.59l3.71 2.22.79-1.28-3-1.78V8h-1.5z"/></svg>')
    
    # 5. Balance Scales of Justice (Orange)
    scale_svg = enc_svg('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="#e65100"><path d="M12 2a1 1 0 0 1 1 1v1.07A7.002 7.002 0 0 1 19 11v1h1a1 1 0 0 1 0 2h-1v5h2a1 1 0 0 1 0 2H3a1 1 0 0 1 0-2h2v-5H4a1 1 0 0 1 0-2h1v-1a7.002 7.002 0 0 1 6-6.93V3a1 1 0 0 1 1-1zm-5 9a5 5 0 0 0 10 0V9.17A5.002 5.002 0 0 0 7 9.17V11zm-2 5v3h14v-3H5z"/></svg>')
    
    # 6. Action Inversion Flip Swap (Crimson)
    swap_svg = enc_svg('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="#c62828"><path d="M16 17.01V10h-2v7.01h-3L15 21l4-3.99h-3zM9 3L5 6.99h3V14h2V6.99h3L9 3z"/></svg>')

    # 7. Brain Neural Network (Purple)
    brain_svg = enc_svg('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="#6f42c1"><path d="M12 3c-4.97 0-9 4.03-9 9 0 2.12.74 4.07 1.97 5.61L4.35 19.4a1 1 0 0 0 1.25 1.25l1.79-.62C9.07 21.26 10.48 22 12 22c4.97 0 9-4.03 9-9s-4.03-9-9-9zm-1 5a2 2 0 1 1 0 4 2 2 0 0 1 0-4zm4 0a2 2 0 1 1 0 4 2 2 0 0 1 0-4zm-4 6a2 2 0 1 1 0 4 2 2 0 0 1 0-4zm4 0a2 2 0 1 1 0 4 2 2 0 0 1 0-4z"/></svg>')

    # 8. Flag / Goal Waypoint (Blue)
    flag_svg = enc_svg('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="#1f4e79"><path d="M14.4 6L14 4H5v17h2v-7h5.6l.4 2h7V6h-5.6z"/></svg>')

    xml_content = rf"""<mxfile host="Electron" modified="2026-08-18T00:00:00.000Z" agent="Mozilla/5.0" version="21.0.0" type="device">
  <diagram id="drl_boundary_framework" name="Decision-Boundary Geometry Framework">
    <mxGraphModel dx="1200" dy="800" grid="1" gridSize="10" guides="1" tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" pageWidth="1169" pageHeight="827" math="1" shadow="0">
      <root>
        <mxCell id="0" />
        <mxCell id="1" parent="0" />
        
        <!-- ================= PANEL A: OBJECTIVE MISMATCH ================= -->
        <mxCell id="panel_a" value="" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#f8f9fa;strokeColor=#dee2e6;strokeWidth=1.5;" vertex="1" parent="1">
          <mxGeometry x="40" y="40" width="340" height="420" as="geometry" />
        </mxCell>
        <mxCell id="title_a" value="&lt;b&gt;(a) Directional Geometry vs. Loss Mismatch&lt;/b&gt;" style="text;html=1;strokeColor=none;fillColor=none;align=left;verticalAlign=middle;whiteSpace=wrap;rounded=0;fontSize=11;fontColor=#1f4e79;" vertex="1" parent="1">
          <mxGeometry x="50" y="50" width="320" height="30" as="geometry" />
        </mxCell>
        
        <!-- SHADED REGIONS -->
        <mxCell id="region_opt" value="&lt;font color=&quot;#1b5e20&quot;&gt;&lt;b&gt;Region \(\mathcal{{R}}(a^*)\)&lt;/b&gt;&lt;br&gt;\(\pi^*(s) = a^*\) (Optimal)&lt;/font&gt;" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#e8f5e9;strokeColor=#a5d6a7;strokeWidth=1;align=center;verticalAlign=top;spacingTop=6;fontSize=9;" vertex="1" parent="1">
          <mxGeometry x="60" y="85" width="145" height="150" as="geometry" />
        </mxCell>
        <mxCell id="region_sub" value="&lt;font color=&quot;#b71c1c&quot;&gt;&lt;b&gt;Region \(\mathcal{{R}}(a^{{(2)}})\)&lt;/b&gt;&lt;br&gt;\(\pi(s) = a^{{(2)}}\) (Suboptimal)&lt;/font&gt;" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#ffebee;strokeColor=#ffcdd2;strokeWidth=1;align=center;verticalAlign=bottom;spacingBottom=6;fontSize=9;" vertex="1" parent="1">
          <mxGeometry x="215" y="225" width="145" height="150" as="geometry" />
        </mxCell>
        
        <!-- DECISION BOUNDARY LINE -->
        <mxCell id="decision_boundary" value="" style="endArrow=none;html=1;strokeColor=#d9534f;strokeWidth=3;dashed=1;" edge="1" parent="1">
          <mxGeometry width="50" height="50" relative="1" as="geometry">
            <mxPoint x="60" y="360" as="sourcePoint" />
            <mxPoint x="360" y="100" as="targetPoint" />
          </mxGeometry>
        </mxCell>
        <mxCell id="boundary_label" value="&lt;font color=&quot;#d9534f&quot;&gt;&lt;b&gt;Boundary &amp;Sigma;: \(Q(a^*) = Q(a^{{(2)}})\)&lt;/b&gt;&lt;/font&gt;" style="text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;whiteSpace=wrap;rounded=0;fontSize=8.5;rotation=-40;" vertex="1" parent="1">
          <mxGeometry x="200" y="125" width="160" height="25" as="geometry" />
        </mxCell>

        <!-- EQUAL PREDICTION LOSS CIRCLE -->
        <mxCell id="err_sphere" value="" style="ellipse;whiteSpace=wrap;html=1;fillColor=none;strokeColor=#78909c;strokeWidth=1.5;dashed=1;" vertex="1" parent="1">
          <mxGeometry x="110" y="150" width="180" height="180" as="geometry" />
        </mxCell>
        <mxCell id="sphere_label" value="&lt;font color=&quot;#37474f&quot;&gt;\(\|\delta P\|_{{\mathrm{{TV}}}} = \epsilon\)&lt;/font&gt;" style="text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;fontSize=8;fontStyle=2;" vertex="1" parent="1">
          <mxGeometry x="220" y="165" width="90" height="20" as="geometry" />
        </mxCell>

        <!-- TRUE STATE WITH ROBOT AGENT ICON -->
        <mxCell id="true_state" value="&lt;b&gt;True \(s'\)&lt;/b&gt;" style="ellipse;whiteSpace=wrap;html=1;aspect=fixed;fillColor=#1f4e79;fontColor=#ffffff;strokeColor=#1f4e79;strokeWidth=2;fontSize=8;verticalAlign=bottom;" vertex="1" parent="1">
          <mxGeometry x="180" y="220" width="40" height="40" as="geometry" />
        </mxCell>
        <mxCell id="robot_icon" value="" style="shape=image;image={robot_svg};verticalLabelPosition=bottom;verticalAlign=top;" vertex="1" parent="1">
          <mxGeometry x="190" y="225" width="20" height="20" as="geometry" />
        </mxCell>

        <!-- EXPANSIVE ERROR WITH SHIELD ICON -->
        <mxCell id="expansive_err" value="" style="endArrow=classic;html=1;strokeColor=#2e7d32;strokeWidth=3;fillColor=#2e7d32;" edge="1" parent="1">
          <mxGeometry width="50" height="50" relative="1" as="geometry">
            <mxPoint x="185" y="225" as="sourcePoint" />
            <mxPoint x="135" y="165" as="targetPoint" />
          </mxGeometry>
        </mxCell>
        <mxCell id="shield_badge" value="&lt;b&gt;Expansive (\(B &amp;lt; 0\))&lt;/b&gt;&lt;br&gt;&lt;font color=&quot;#2e7d32&quot;&gt;&lt;b&gt;Safe (\(C_i = 0\))&lt;/b&gt;&lt;/font&gt;" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#ffffff;strokeColor=#2e7d32;fontSize=7.5;spacingLeft=18;align=left;" vertex="1" parent="1">
          <mxGeometry x="65" y="140" width="115" height="35" as="geometry" />
        </mxCell>
        <mxCell id="shield_icon" value="" style="shape=image;image={shield_svg};" vertex="1" parent="1">
          <mxGeometry x="70" y="148" width="18" height="18" as="geometry" />
        </mxCell>

        <!-- COMPRESSIVE ERROR WITH WARNING ICON -->
        <mxCell id="compressive_err" value="" style="endArrow=classic;html=1;strokeColor=#c62828;strokeWidth=3;fillColor=#c62828;" edge="1" parent="1">
          <mxGeometry width="50" height="50" relative="1" as="geometry">
            <mxPoint x="215" y="255" as="sourcePoint" />
            <mxPoint x="265" y="315" as="targetPoint" />
          </mxGeometry>
        </mxCell>
        <mxCell id="warn_badge" value="&lt;b&gt;Compressive (\(B &amp;gt; 1\))&lt;/b&gt;&lt;br&gt;&lt;font color=&quot;#c62828&quot;&gt;&lt;b&gt;Action Flip (\(Z=1\))&lt;/b&gt;&lt;/font&gt;" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#ffffff;strokeColor=#c62828;fontSize=7.5;spacingLeft=18;align=left;" vertex="1" parent="1">
          <mxGeometry x="225" y="305" width="130" height="35" as="geometry" />
        </mxCell>
        <mxCell id="warn_icon" value="" style="shape=image;image={warning_svg};" vertex="1" parent="1">
          <mxGeometry x="230" y="313" width="18" height="18" as="geometry" />
        </mxCell>

        <mxCell id="mismatch_note" value="&lt;b&gt;Objective Mismatch:&lt;/b&gt; Equal prediction error \(\|\delta P\|_{{\mathrm{{TV}}}}\), but opposite control outcomes." style="rounded=1;whiteSpace=wrap;html=1;fillColor=#ffffff;strokeColor=#ced4da;fontSize=7.8;align=left;spacingLeft=6;" vertex="1" parent="1">
          <mxGeometry x="50" y="380" width="320" height="35" as="geometry" />
        </mxCell>

        <!-- ================= PANEL B: FOUR REGIMES & GAUGE ================= -->
        <mxCell id="panel_b" value="" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#f8f9fa;strokeColor=#dee2e6;strokeWidth=1.5;" vertex="1" parent="1">
          <mxGeometry x="400" y="40" width="340" height="420" as="geometry" />
        </mxCell>
        <mxCell id="title_b" value="&lt;b&gt;(b) Decision Margin Gauge &amp;amp; Four Regimes&lt;/b&gt;" style="text;html=1;strokeColor=none;fillColor=none;align=left;verticalAlign=middle;whiteSpace=wrap;rounded=0;fontSize=11;fontColor=#1f4e79;" vertex="1" parent="1">
          <mxGeometry x="410" y="50" width="320" height="25" as="geometry" />
        </mxCell>
        
        <!-- CONTINUOUS GAUGE RULER -->
        <mxCell id="gauge_icon_top" value="" style="shape=image;image={gauge_svg};" vertex="1" parent="1">
          <mxGeometry x="415" y="80" width="22" height="22" as="geometry" />
        </mxCell>
        <mxCell id="gauge_exp" value="&lt;b&gt;B &amp;lt; 0&lt;/b&gt;" style="rounded=0;whiteSpace=wrap;html=1;fillColor=#c8e6c9;strokeColor=#2e7d32;fontSize=8;fontColor=#1b5e20;" vertex="1" parent="1">
          <mxGeometry x="445" y="80" width="80" height="24" as="geometry" />
        </mxCell>
        <mxCell id="gauge_sub" value="&lt;b&gt;0 &amp;le; B &amp;lt; 1&lt;/b&gt;" style="rounded=0;whiteSpace=wrap;html=1;fillColor=#fff9c4;strokeColor=#f57f17;fontSize=8;fontColor=#e65100;" vertex="1" parent="1">
          <mxGeometry x="525" y="80" width="95" height="24" as="geometry" />
        </mxCell>
        <mxCell id="gauge_cross" value="&lt;b&gt;B &amp;gt; 1.0&lt;/b&gt;" style="rounded=0;whiteSpace=wrap;html=1;fillColor=#ffcdd2;strokeColor=#c62828;fontSize=8;fontColor=#b71c1c;" vertex="1" parent="1">
          <mxGeometry x="620" y="80" width="95" height="24" as="geometry" />
        </mxCell>
        <mxCell id="gauge_needle" value="&lt;b&gt;Threshold B=1.0&lt;/b&gt; (Tie)" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#ffffff;strokeColor=#000000;strokeWidth=1.5;align=center;fontSize=7.2;" vertex="1" parent="1">
          <mxGeometry x="565" y="108" width="110" height="20" as="geometry" />
        </mxCell>
        
        <!-- 4 REGIME CARDS WITH ICONS -->
        <mxCell id="regime1" value="&lt;b&gt;1. Expansive Regime&lt;/b&gt; &lt;span style=&quot;float:right;&quot;&gt;\(B &amp;lt; 0\ (\Delta m &amp;gt; 0)\)&lt;/span&gt;&lt;br&gt;Margin widened; optimal action reinforced." style="rounded=1;whiteSpace=wrap;html=1;fillColor=#e8f5e9;strokeColor=#2e7d32;fontSize=8;align=left;spacingLeft=26;" vertex="1" parent="1">
          <mxGeometry x="420" y="138" width="300" height="50" as="geometry" />
        </mxCell>
        <mxCell id="reg1_icon" value="" style="shape=image;image={shield_svg};" vertex="1" parent="1">
          <mxGeometry x="426" y="152" width="20" height="20" as="geometry" />
        </mxCell>

        <mxCell id="regime2" value="&lt;b&gt;2. Compressive Sub-threshold&lt;/b&gt; &lt;span style=&quot;float:right;&quot;&gt;\(0 &amp;le; B &amp;lt; 1\)&lt;/span&gt;&lt;br&gt;Margin narrowed; optimal action preserved." style="rounded=1;whiteSpace=wrap;html=1;fillColor=#fffde7;strokeColor=#f57f17;fontSize=8;align=left;spacingLeft=26;" vertex="1" parent="1">
          <mxGeometry x="420" y="195" width="300" height="50" as="geometry" />
        </mxCell>
        <mxCell id="reg2_icon" value="" style="shape=image;image={gauge_svg};" vertex="1" parent="1">
          <mxGeometry x="426" y="209" width="20" height="20" as="geometry" />
        </mxCell>

        <mxCell id="regime3" value="&lt;b&gt;3. Exact Decision Tie&lt;/b&gt; &lt;span style=&quot;float:right;&quot;&gt;\(B = 1.0\ (\Delta m = -m)\)&lt;/span&gt;&lt;br&gt;Action-values tie: \(Q(a^*) = Q(a^{{(2)}})\)." style="rounded=1;whiteSpace=wrap;html=1;fillColor=#fff3e0;strokeColor=#e65100;fontSize=8;align=left;spacingLeft=26;" vertex="1" parent="1">
          <mxGeometry x="420" y="252" width="300" height="50" as="geometry" />
        </mxCell>
        <mxCell id="reg3_icon" value="" style="shape=image;image={scale_svg};" vertex="1" parent="1">
          <mxGeometry x="426" y="266" width="20" height="20" as="geometry" />
        </mxCell>

        <mxCell id="regime4" value="&lt;b&gt;4. Strict Boundary Crossing&lt;/b&gt; &lt;span style=&quot;float:right;&quot;&gt;\(B &amp;gt; 1.0\)&lt;/span&gt;&lt;br&gt;&lt;b&gt;Action Inverted:&lt;/b&gt; Runner-up selected (\(Z_{{\mathrm{{cross}}}}=1\))." style="rounded=1;whiteSpace=wrap;html=1;fillColor=#ffebee;strokeColor=#c62828;fontSize=8;align=left;spacingLeft=26;" vertex="1" parent="1">
          <mxGeometry x="420" y="309" width="300" height="52" as="geometry" />
        </mxCell>
        <mxCell id="reg4_icon" value="" style="shape=image;image={swap_svg};" vertex="1" parent="1">
          <mxGeometry x="426" y="323" width="20" height="20" as="geometry" />
        </mxCell>

        <mxCell id="formula_box" value="&lt;b&gt;Normalized Boundary Pressure:&lt;/b&gt; \(B(s) = -\frac{{\Delta m(s)}}{{m_P(s)}}\)" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#ffffff;strokeColor=#ced4da;fontSize=8;align=center;" vertex="1" parent="1">
          <mxGeometry x="420" y="370" width="300" height="35" as="geometry" />
        </mxCell>

        <!-- ================= PANEL C: BELLMAN BOTTLENECK ================= -->
        <mxCell id="panel_c" value="" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#f8f9fa;strokeColor=#dee2e6;strokeWidth=1.5;" vertex="1" parent="1">
          <mxGeometry x="760" y="40" width="360" height="420" as="geometry" />
        </mxCell>
        <mxCell id="title_c" value="&lt;b&gt;(c) Spatial Corridor &amp;amp; Bellman Propagation Bottleneck&lt;/b&gt;" style="text;html=1;strokeColor=none;fillColor=none;align=left;verticalAlign=middle;whiteSpace=wrap;rounded=0;fontSize=11;fontColor=#1f4e79;" vertex="1" parent="1">
          <mxGeometry x="770" y="50" width="340" height="25" as="geometry" />
        </mxCell>
        
        <!-- TRAJECTORY NODES -->
        <mxCell id="node_s0" value="&lt;b&gt;\(s_0\)&lt;/b&gt;&lt;br&gt;Start" style="ellipse;whiteSpace=wrap;html=1;aspect=fixed;fillColor=#e3f2fd;strokeColor=#1f4e79;strokeWidth=2;fontSize=8;verticalAlign=bottom;" vertex="1" parent="1">
          <mxGeometry x="775" y="85" width="45" height="45" as="geometry" />
        </mxCell>
        <mxCell id="icon_s0" value="" style="shape=image;image={flag_svg};" vertex="1" parent="1">
          <mxGeometry x="789" y="88" width="16" height="16" as="geometry" />
        </mxCell>

        <mxCell id="node_s1" value="&lt;b&gt;\(s_1\)&lt;/b&gt;&lt;br&gt;Corridor" style="ellipse;whiteSpace=wrap;html=1;aspect=fixed;fillColor=#ffffff;strokeColor=#6c757d;strokeWidth=1.5;fontSize=7.5;" vertex="1" parent="1">
          <mxGeometry x="855" y="85" width="45" height="45" as="geometry" />
        </mxCell>
        <mxCell id="node_s2" value="&lt;b&gt;\(s_2\)&lt;/b&gt;&lt;br&gt;Corridor" style="ellipse;whiteSpace=wrap;html=1;aspect=fixed;fillColor=#ffffff;strokeColor=#6c757d;strokeWidth=1.5;fontSize=7.5;" vertex="1" parent="1">
          <mxGeometry x="935" y="85" width="45" height="45" as="geometry" />
        </mxCell>
        <mxCell id="node_scross" value="&lt;b&gt;\(s_{{\mathrm{{cross}}}}\)&lt;/b&gt;&lt;br&gt;Boundary" style="ellipse;whiteSpace=wrap;html=1;aspect=fixed;fillColor=#ffebee;strokeColor=#d9534f;strokeWidth=2;fontSize=8;verticalAlign=bottom;" vertex="1" parent="1">
          <mxGeometry x="1025" y="85" width="45" height="45" as="geometry" />
        </mxCell>
        <mxCell id="icon_scross" value="" style="shape=image;image={warn_svg if 'warn_svg' in locals() else warning_svg};" vertex="1" parent="1">
          <mxGeometry x="1039" y="88" width="16" height="16" as="geometry" />
        </mxCell>
        
        <!-- FORWARD TRANSITIONS -->
        <mxCell id="edge_01" value="\(w_i \downarrow\)" style="edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;html=1;entryX=0;entryY=0.5;entryDx=0;entryDy=0;strokeColor=#6c757d;strokeWidth=1.5;dashed=1;fontSize=7.5;" edge="1" parent="1" source="node_s0" target="node_s1">
          <mxGeometry relative="1" as="geometry" />
        </mxCell>
        <mxCell id="edge_12" value="\(w_i \downarrow\)" style="edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;html=1;entryX=0;entryY=0.5;entryDx=0;entryDy=0;strokeColor=#6c757d;strokeWidth=1.5;dashed=1;fontSize=7.5;" edge="1" parent="1" source="node_s1" target="node_s2">
          <mxGeometry relative="1" as="geometry" />
        </mxCell>
        <mxCell id="edge_2cross" value="\(w_i \uparrow\uparrow\)" style="edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;html=1;entryX=0;entryY=0.5;entryDx=0;entryDy=0;strokeColor=#d9534f;strokeWidth=2.5;fontSize=8;fontColor=#d9534f;" edge="1" parent="1" source="node_s2" target="node_scross">
          <mxGeometry relative="1" as="geometry" />
        </mxCell>

        <!-- BELLMAN BACKUP ARROWS -->
        <mxCell id="backup_arrow" value="&lt;font color=&quot;#6f42c1&quot;&gt;&lt;b&gt;Global Bellman Backup:&lt;/b&gt; \(V(s) \leftarrow \max_a [R + \gamma \sum P(s') V(s')]\)&lt;/font&gt;" style="curved=1;endArrow=classic;html=1;strokeColor=#6f42c1;strokeWidth=2.2;dashed=1;fontSize=8;" edge="1" parent="1">
          <mxGeometry width="50" height="50" relative="1" as="geometry">
            <mxPoint x="1045" y="145" as="sourcePoint" />
            <mxPoint x="795" y="145" as="targetPoint" />
            <Array as="points">
              <mxPoint x="920" y="185" />
            </Array>
          </mxGeometry>
        </mxCell>
        
        <!-- 4-STEP CAUSAL FAILURE FLOWCHART -->
        <mxCell id="box_up" value="&lt;b&gt;1. Boundary Upweight&lt;/b&gt;&lt;br&gt;\(w(s_{{\mathrm{{cross}}}}) \uparrow\ (\lambda &amp;gt; 0)\)" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#ffebee;strokeColor=#d9534f;fontSize=7.5;fontColor=#d9534f;" vertex="1" parent="1">
          <mxGeometry x="775" y="210" width="150" height="50" as="geometry" />
        </mxCell>
        <mxCell id="box_down" value="&lt;b&gt;2. Corridor Downweight&lt;/b&gt;&lt;br&gt;\(w(s_{{\mathrm{{corr}}}}) \downarrow\ (\sum w_i = N)\)" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#fff3e0;strokeColor=#e65100;fontSize=7.5;fontColor=#e65100;" vertex="1" parent="1">
          <mxGeometry x="955" y="210" width="150" height="50" as="geometry" />
        </mxCell>
        <mxCell id="box_distort" value="&lt;b&gt;3. Value Distortion&lt;/b&gt;&lt;br&gt;\(\hat{{V}}(s')\) biased in backup" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#ede7f6;strokeColor=#6f42c1;fontSize=7.5;fontColor=#6f42c1;" vertex="1" parent="1">
          <mxGeometry x="955" y="290" width="150" height="50" as="geometry" />
        </mxCell>
        <mxCell id="box_flip" value="&lt;b&gt;4. Upstream Policy Flips&lt;/b&gt;&lt;br&gt;&lt;b&gt;New Action Flips at \(s_0 \downarrow\)&lt;/b&gt;" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#ffebee;strokeColor=#c62828;fontSize=7.5;fontColor=#c62828;" vertex="1" parent="1">
          <mxGeometry x="775" y="290" width="150" height="50" as="geometry" />
        </mxCell>
        
        <!-- FLOW ARROWS -->
        <mxCell id="flow_12" style="edgeStyle=orthogonalEdgeStyle;rounded=0;html=1;strokeColor=#333333;strokeWidth=1.5;" edge="1" parent="1" source="box_up" target="box_down">
          <mxGeometry relative="1" as="geometry" />
        </mxCell>
        <mxCell id="flow_23" style="edgeStyle=orthogonalEdgeStyle;rounded=0;html=1;strokeColor=#333333;strokeWidth=1.5;" edge="1" parent="1" source="box_down" target="box_distort">
          <mxGeometry relative="1" as="geometry" />
        </mxCell>
        <mxCell id="flow_34" style="edgeStyle=orthogonalEdgeStyle;rounded=0;html=1;strokeColor=#333333;strokeWidth=1.5;" edge="1" parent="1" source="box_distort" target="box_flip">
          <mxGeometry relative="1" as="geometry" />
        </mxCell>
        
        <mxCell id="verdict_box" value="&lt;b&gt;Takeaway:&lt;/b&gt; Diagnostic boundary relevance does NOT imply sample-reweighted model learning superiority." style="rounded=1;whiteSpace=wrap;html=1;fillColor=#e8eaf6;strokeColor=#3f51b5;fontSize=8;align=center;" vertex="1" parent="1">
          <mxGeometry x="775" y="365" width="330" height="45" as="geometry" />
        </mxCell>

      </root>
    </mxGraphModel>
  </diagram>
</mxfile>"""
    with open(drawio_path, 'w', encoding='utf-8') as f:
        f.write(xml_content)
    print(f"Generated: {drawio_path} with embedded vector SVG icons!")


def render_figure1_vector():
    fig = plt.figure(figsize=(13.6, 4.7))
    gs = GridSpec(1, 3, width_ratios=[1.0, 1.05, 1.18], wspace=0.20)

    # Helper: Draw vector shield icon
    def draw_shield(ax, cx, cy, size, fc='#2e7d32'):
        pts = np.array([[cx, cy + size], [cx + size, cy + size*0.5], [cx + size*0.7, cy - size],
                        [cx, cy - size*1.2], [cx - size*0.7, cy - size], [cx - size, cy + size*0.5]])
        ax.add_patch(patches.Polygon(pts, closed=True, facecolor=fc, edgecolor='white', linewidth=0.8, zorder=10))
        ax.text(cx, cy - size*0.1, r"$\checkmark$", color='white', fontsize=size*450, ha='center', va='center', fontweight='bold', zorder=11)

    # Helper: Draw vector warning triangle icon
    def draw_warning(ax, cx, cy, size, fc='#c62828'):
        pts = np.array([[cx, cy + size*1.1], [cx + size*1.1, cy - size*0.9], [cx - size*1.1, cy - size*0.9]])
        ax.add_patch(patches.Polygon(pts, closed=True, facecolor=fc, edgecolor='white', linewidth=0.8, zorder=10))
        ax.text(cx, cy - size*0.15, r"!", color='white', fontsize=size*450, ha='center', va='center', fontweight='bold', zorder=11)

    # Helper: Draw vector robot head icon
    def draw_robot(ax, cx, cy, size, fc=NAVY):
        # Head box
        ax.add_patch(patches.FancyBboxPatch((cx - size, cy - size*0.75), size*2, size*1.5, boxstyle="round,pad=0.005",
                                            facecolor=fc, edgecolor='white', linewidth=1.0, zorder=10))
        # Antenna
        ax.plot([cx, cx], [cy + size*0.75, cy + size*1.15], color=fc, linewidth=1.2, zorder=10)
        ax.plot(cx, cy + size*1.15, 'o', color=fc, markersize=3, zorder=11)
        # Eyes
        ax.plot(cx - size*0.45, cy + size*0.1, 'o', color='white', markersize=2.8, zorder=11)
        ax.plot(cx + size*0.45, cy + size*0.1, 'o', color='white', markersize=2.8, zorder=11)

    # =========================================================================
    # --- PANEL A: DIRECTIONAL GEOMETRY & OBJECTIVE MISMATCH ---
    # =========================================================================
    ax1 = fig.add_subplot(gs[0])
    ax1.set_title(r"$\mathbf{(a)}$ Directional Geometry vs. Loss Mismatch", fontsize=9.5, fontweight='bold', pad=10)
    
    clip_ellipse = patches.Ellipse((0.5, 0.5), 0.86, 0.82, angle=-12, facecolor='none', edgecolor='none')
    ax1.add_patch(clip_ellipse)

    # Shaded action regions
    poly_a_star = patches.Polygon([[0.05, 0.95], [0.95, 0.95], [0.85, 0.68], [0.12, 0.28], [0.05, 0.40]],
                                  facecolor='#e8f5e9', edgecolor='none', alpha=0.85, clip_path=clip_ellipse)
    ax1.add_patch(poly_a_star)
    
    poly_a_comp = patches.Polygon([[0.05, 0.05], [0.95, 0.05], [0.95, 0.68], [0.85, 0.68], [0.12, 0.28], [0.05, 0.20]],
                                  facecolor='#ffebee', edgecolor='none', alpha=0.85, clip_path=clip_ellipse)
    ax1.add_patch(poly_a_comp)

    border_ellipse = patches.Ellipse((0.5, 0.5), 0.86, 0.82, angle=-12, facecolor='none', edgecolor='#78909c', linewidth=1.6)
    ax1.add_patch(border_ellipse)
    
    # Boundary line Sigma
    x_b = np.linspace(0.12, 0.85, 100)
    y_b = 0.28 + 0.55 * (x_b - 0.12)
    ax1.plot(x_b, y_b, '--', color=CORAL, linewidth=2.4)
    ax1.text(0.66, 0.65, r"Boundary $\Sigma: Q(a^*)=Q(a^{(2)})$", color='#c62828', fontsize=7.2, fontweight='bold', rotation=32)
    
    # Region Labels
    ax1.text(0.18, 0.76, r"Region $\mathcal{R}(a^*)$" "\n" r"$\pi^*(s) = a^*$ (Optimal)", color='#1b5e20', fontsize=8.0, fontweight='bold')
    ax1.text(0.64, 0.20, r"Region $\mathcal{R}(a^{(2)})$" "\n" r"$\pi(s) = a^{(2)}$ (Suboptimal)", color='#b71c1c', fontsize=8.0, fontweight='bold')
    
    s_x, s_y = 0.40, 0.53
    r_err = 0.24
    err_circle = plt.Circle((s_x, s_y), r_err, color='#546e7a', fill=False, linestyle=':', linewidth=1.4)
    ax1.add_patch(err_circle)
    ax1.text(s_x + 0.12, s_y + 0.17, r"$\|\delta P\|_{\mathrm{TV}} = \epsilon$", color='#37474f', fontsize=7.5, fontstyle='italic')

    # True state with Robot Agent Icon
    draw_robot(ax1, s_x, s_y, size=0.035, fc=NAVY)
    ax1.text(s_x - 0.04, s_y - 0.07, r"True $s'$", color=NAVY, fontsize=8.5, fontweight='bold')
    
    # Vectors
    v_exp_x, v_exp_y = s_x - 0.16, s_y + 0.18
    ax1.annotate('', xy=(v_exp_x, v_exp_y), xytext=(s_x, s_y),
                 arrowprops=dict(facecolor=EMERALD, edgecolor=EMERALD, arrowstyle="-|>", lw=2.4, mutation_scale=14))
    ax1.annotate(r"$\mathbf{Expansive\ (B < 0)}$" "\n" r"$\mathbf{Safe}\ (C_i = 0)$",
                 xy=(v_exp_x, v_exp_y), xytext=(0.07, 0.84),
                 arrowprops=dict(arrowstyle="->", color=EMERALD, lw=1.2),
                 fontsize=7.5, bbox=dict(boxstyle="round,pad=0.25", fc="#ffffff", ec=EMERALD, lw=1))
    draw_shield(ax1, 0.04, 0.88, size=0.022, fc='#2e7d32')
    
    v_comp_x, v_comp_y = s_x + 0.18, s_y - 0.16
    ax1.annotate('', xy=(v_comp_x, v_comp_y), xytext=(s_x, s_y),
                 arrowprops=dict(facecolor=CORAL, edgecolor=CORAL, arrowstyle="-|>", lw=2.4, mutation_scale=14))
    ax1.annotate(r"$\mathbf{Compressive\ (B > 1)}$" "\n" r"$\mathbf{Action\ Flip}\ (Z=1)$",
                 xy=(v_comp_x, v_comp_y), xytext=(0.54, 0.36),
                 arrowprops=dict(arrowstyle="->", color=CORAL, lw=1.2),
                 fontsize=7.5, bbox=dict(boxstyle="round,pad=0.25", fc="#ffffff", ec=CORAL, lw=1))
    draw_warning(ax1, 0.50, 0.40, size=0.022, fc='#c62828')
    
    ax1.set_xlim(0.01, 0.99)
    ax1.set_ylim(0.06, 0.98)
    ax1.axis('off')

    # =========================================================================
    # --- PANEL B: CONTINUOUS MARGIN GAUGE & 4 REGIMES ---
    # =========================================================================
    ax2 = fig.add_subplot(gs[1])
    ax2.set_title(r"$\mathbf{(b)}$ Decision Margin Gauge & The Four Regimes", fontsize=9.5, fontweight='bold', pad=10)
    
    # 1. TOP RULER / GAUGE BAR: Visualizing B continuum
    gauge_y = 0.86
    gauge_h = 0.06
    # Expansive Zone (Green)
    ax2.add_patch(patches.Rectangle((0.05, gauge_y), 0.28, gauge_h, facecolor='#c8e6c9', edgecolor='#2e7d32', linewidth=1.2))
    ax2.text(0.19, gauge_y + 0.03, r"$B < 0$", ha='center', va='center', fontsize=7.2, fontweight='bold', color='#1b5e20')
    
    # Sub-threshold Zone (Amber)
    ax2.add_patch(patches.Rectangle((0.33, gauge_y), 0.34, gauge_h, facecolor='#fff9c4', edgecolor='#f57f17', linewidth=1.2))
    ax2.text(0.50, gauge_y + 0.03, r"$0 \leq B < 1$", ha='center', va='center', fontsize=7.2, fontweight='bold', color='#e65100')
    
    # Crossing Zone (Red)
    ax2.add_patch(patches.Rectangle((0.67, gauge_y), 0.28, gauge_h, facecolor='#ffcdd2', edgecolor='#c62828', linewidth=1.2))
    ax2.text(0.81, gauge_y + 0.03, r"$B > 1.0$", ha='center', va='center', fontsize=7.2, fontweight='bold', color='#b71c1c')
    
    # Critical Threshold Needle at B = 1.0 (x = 0.67)
    ax2.plot([0.67, 0.67], [gauge_y - 0.02, gauge_y + gauge_h + 0.02], color='black', linewidth=2.4)
    ax2.annotate(r"$\mathbf{Threshold\ } B=1.0$" "\n" r"(Tie: $\Delta m = -m$)",
                 xy=(0.67, gauge_y + gauge_h + 0.01), xytext=(0.48, 0.95),
                 arrowprops=dict(arrowstyle="-|>", color='black', lw=1.2),
                 fontsize=6.8, bbox=dict(boxstyle="round,pad=0.2", fc="#ffffff", ec='black', lw=0.9))

    # 2. FOUR DETAILED REGIME CARDS (with Icon Badges)
    regimes_data = [
        ("1. Expansive Regime", r"$B < 0\ (\Delta m > 0)$", "Margin widened; $a^*$ reinforced", '#e8f5e9', '#2e7d32', 0.65, 'shield'),
        ("2. Compressive Sub-threshold", r"$0 \leq B < 1\ (-m < \Delta m < 0)$", "Margin narrowed; $a^*$ preserved", '#fffde7', '#f57f17', 0.45, 'gauge'),
        ("3. Exact Decision Tie", r"$B = 1.0\ (\Delta m = -m)$", r"Action values tie: $Q(a^*)=Q(a^{(2)})$", '#fff3e0', '#e65100', 0.25, 'tie'),
        ("4. Strict Boundary Crossing", r"$B > 1.0\ (\Delta m < -m)$", r"$\mathbf{Action\ Inversion:}\ Z_{\mathrm{cross}}=1\ (a^* \to a^{(2)})$", '#ffebee', '#c62828', 0.05, 'warn')
    ]
    
    card_w, card_h = 0.90, 0.17
    for title, cond, desc, bg, border_c, y_b, itype in regimes_data:
        card = patches.FancyBboxPatch((0.05, y_b), card_w, card_h, boxstyle="round,pad=0.02",
                                      facecolor=bg, edgecolor=border_c, linewidth=1.2)
        ax2.add_patch(card)
        
        # Icon badge on the left of each card
        if itype == 'shield':
            draw_shield(ax2, 0.09, y_b + 0.085, size=0.024, fc=border_c)
        elif itype == 'warn':
            draw_warning(ax2, 0.09, y_b + 0.085, size=0.024, fc=border_c)
        elif itype == 'tie':
            ax2.text(0.09, y_b + 0.085, r"$\approx$", color=border_c, fontsize=12, ha='center', va='center', fontweight='bold')
        elif itype == 'gauge':
            ax2.text(0.09, y_b + 0.085, r"$\rightleftharpoons$", color=border_c, fontsize=11, ha='center', va='center', fontweight='bold')

        ax2.text(0.14, y_b + 0.115, title, fontsize=7.6, fontweight='bold', color=border_c)
        ax2.text(0.92, y_b + 0.115, cond, fontsize=6.8, color=DARK, fontweight='bold', ha='right')
        ax2.text(0.14, y_b + 0.04, desc, fontsize=7.0, color=DARK)

    ax2.set_xlim(0.0, 1.0)
    ax2.set_ylim(0.0, 1.05)
    ax2.axis('off')

    # =========================================================================
    # --- PANEL C: SPATIAL CORRIDOR & NON-LOCAL BELLMAN BOTTLENECK ---
    # =========================================================================
    ax3 = fig.add_subplot(gs[2])
    ax3.set_title(r"$\mathbf{(c)}$ Spatial Corridor & Bellman Propagation Bottleneck", fontsize=9.5, fontweight='bold', pad=10)
    
    # 1. TOP: GridWorld Environment Corridor Trajectory
    # Nodes s0 -> s1 -> s2 -> scross
    nodes = [(0.12, 0.78, r"$s_0$", "Start", '#e3f2fd', NAVY),
             (0.36, 0.78, r"$s_1$", "Corridor", '#f5f5f5', '#616161'),
             (0.60, 0.78, r"$s_2$", "Corridor", '#f5f5f5', '#616161'),
             (0.88, 0.78, r"$s_{\mathrm{cross}}$", "Boundary", '#ffebee', CORAL)]
    
    for i, (nx, ny, nlabel, subtext, fc, ec) in enumerate(nodes):
        circle = plt.Circle((nx, ny), 0.072, facecolor=fc, edgecolor=ec, linewidth=1.6)
        ax3.add_patch(circle)
        ax3.text(nx, ny + 0.015, nlabel, ha='center', va='center', fontsize=8.5, fontweight='bold', color=ec)
        ax3.text(nx, ny - 0.03, f"({subtext})", ha='center', va='center', fontsize=6.5, color=DARK)
        
        if i == 0:
            # Start flag icon on top of s0
            ax3.text(nx, ny + 0.058, r"$\blacktriangleright$", color=NAVY, fontsize=8, ha='center')
        elif i == 3:
            # Danger icon on top of scross
            draw_warning(ax3, nx, ny + 0.058, size=0.018, fc=CORAL)

        if i < 3:
            next_x = nodes[i+1][0]
            if i == 2:
                # Boundary transition (upweighted w_i)
                ax3.annotate('', xy=(next_x - 0.075, ny), xytext=(nx + 0.075, ny),
                             arrowprops=dict(arrowstyle="-|>", color=CORAL, lw=2.4, mutation_scale=11))
                ax3.text((nx + next_x)/2, ny + 0.065, r"$w_i \uparrow\uparrow$", color=CORAL, fontsize=7.2, fontweight='bold', ha='center')
            else:
                # Corridor transition (downweighted w_i)
                ax3.annotate('', xy=(next_x - 0.075, ny), xytext=(nx + 0.075, ny),
                             arrowprops=dict(arrowstyle="-|>", color='#78909c', lw=1.3, linestyle='--', mutation_scale=9))
                ax3.text((nx + next_x)/2, ny + 0.065, r"$w_i \downarrow$", color='#546e7a', fontsize=7.2, ha='center')

    # Backward Bellman Backup Arc
    arc = patches.FancyArrowPatch((0.88, 0.67), (0.12, 0.67), connectionstyle="arc3,rad=-0.30",
                                  arrowstyle="-|>", color=PURPLE, lw=2.0, linestyle='--', mutation_scale=12)
    ax3.add_patch(arc)
    ax3.text(0.50, 0.51, r"Global Value Backup: $V(s) \leftarrow \max_a [R + \gamma \mathbb{E} V(s')]$",
             ha='center', va='center', fontsize=7.4, color=PURPLE, fontweight='bold')
    
    # 2. BOTTOM: Causal Failure Chain Flowchart (2x2 Grid)
    box_w, box_h = 0.43, 0.17
    # Box 1: Boundary Upweight
    ax3.add_patch(patches.FancyBboxPatch((0.04, 0.25), box_w, box_h, boxstyle="round,pad=0.02", facecolor='#ffebee', edgecolor=CORAL, linewidth=1.2))
    ax3.text(0.04 + box_w/2, 0.355, "1. Boundary Upweight", fontsize=7.0, fontweight='bold', color=CORAL, ha='center')
    ax3.text(0.04 + box_w/2, 0.290, r"$w(s_{\mathrm{cross}}) \uparrow\ (\lambda > 0)$", fontsize=6.8, color=DARK, ha='center')

    # Box 2: Corridor Downweight
    ax3.add_patch(patches.FancyBboxPatch((0.53, 0.25), box_w, box_h, boxstyle="round,pad=0.02", facecolor='#fff3e0', edgecolor='#e65100', linewidth=1.2))
    ax3.text(0.53 + box_w/2, 0.355, "2. Corridor Downweight", fontsize=7.0, fontweight='bold', color='#e65100', ha='center')
    ax3.text(0.53 + box_w/2, 0.290, r"$w(s_1, s_2) \downarrow\ (\sum w_i = N)$", fontsize=6.8, color=DARK, ha='center')

    # Box 3: Distorted Value Backup
    ax3.add_patch(patches.FancyBboxPatch((0.53, 0.04), box_w, box_h, boxstyle="round,pad=0.02", facecolor='#ede7f6', edgecolor=PURPLE, linewidth=1.2))
    ax3.text(0.53 + box_w/2, 0.145, "3. Value Distortion", fontsize=7.0, fontweight='bold', color=PURPLE, ha='center')
    ax3.text(0.53 + box_w/2, 0.080, r"$\hat{V}(s')$ corrupted in backup", fontsize=6.8, color=DARK, ha='center')

    # Box 4: Upstream Flips
    ax3.add_patch(patches.FancyBboxPatch((0.04, 0.04), box_w, box_h, boxstyle="round,pad=0.02", facecolor='#ffebee', edgecolor='#c62828', linewidth=1.2))
    ax3.text(0.04 + box_w/2, 0.145, "4. Upstream Policy Flips", fontsize=7.0, fontweight='bold', color='#c62828', ha='center')
    ax3.text(0.04 + box_w/2, 0.080, r"$\mathbf{New\ Action\ Flips\ at\ } s_0 \downarrow$", fontsize=6.8, color=DARK, ha='center')

    # Flow arrows: 1 -> 2 -> 3 -> 4
    ax3.annotate('', xy=(0.52, 0.335), xytext=(0.48, 0.335), arrowprops=dict(arrowstyle="-|>", color=DARK, lw=1.3, mutation_scale=10))
    ax3.annotate('', xy=(0.745, 0.22), xytext=(0.745, 0.245), arrowprops=dict(arrowstyle="-|>", color=DARK, lw=1.3, mutation_scale=10))
    ax3.annotate('', xy=(0.48, 0.125), xytext=(0.52, 0.125), arrowprops=dict(arrowstyle="-|>", color=DARK, lw=1.3, mutation_scale=10))
    
    ax3.set_xlim(0.0, 1.0)
    ax3.set_ylim(0.0, 1.05)
    ax3.axis('off')

    save_all_formats(fig, 'conceptual_framework')


# ==============================================================================
# 2. EXPERIMENTAL PLOTS GENERATION
# ==============================================================================

def save_all_formats(fig, base_name):
    """Save figure in PDF, EPS, and PNG formats."""
    pdf_path = os.path.join(OUT_DIR, f"{base_name}.pdf")
    eps_path = os.path.join(OUT_DIR, f"{base_name}.eps")
    png_path = os.path.join(OUT_DIR, f"{base_name}.png")
    fig.savefig(pdf_path, bbox_inches='tight', dpi=300)
    fig.savefig(eps_path, bbox_inches='tight', dpi=300, format='eps')
    fig.savefig(png_path, bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"Generated: {base_name}.pdf, .eps, .png")


def render_figure2_matched_pairs():
    csv_path = os.path.join(RES_DIR, 'stage0_matched_pairs.csv')
    df = pd.read_csv(csv_path)
    
    # Bin by L1 error to produce a smooth, clean causal progression
    df['l1_bin'] = pd.cut(df['error_l1'], bins=10)
    bin_df = df.groupby('l1_bin', observed=False).agg({
        'error_l1': 'mean',
        'correction_value_comp': ['mean', 'sem'],
        'correction_value_exp': ['mean', 'sem']
    }).dropna()
    
    x_vals = bin_df[('error_l1', 'mean')].values
    comp_m = bin_df[('correction_value_comp', 'mean')].values
    comp_s = bin_df[('correction_value_comp', 'sem')].values
    exp_m = bin_df[('correction_value_exp', 'mean')].values
    exp_s = bin_df[('correction_value_exp', 'sem')].values
    
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    
    # Compressive curve
    ax.plot(x_vals, comp_m, 'o-', color=CORAL, linewidth=2.4, markersize=7.0,
            markeredgecolor='black', markeredgewidth=0.9, label=r"$\mathbf{Compressive\ Perturbation}\ (B > 0,\ \text{Toward Boundary})$")
    ax.fill_between(x_vals, comp_m - comp_s, comp_m + comp_s, color=CORAL, alpha=0.20)
    
    # Expansive curve
    ax.plot(x_vals, exp_m, 's--', color=STEEL, linewidth=2.2, markersize=6.5,
            markeredgecolor='black', markeredgewidth=0.9, label=r"$\mathbf{Expansive\ Perturbation}\ (B < 0,\ \text{Away from Boundary})$")
    ax.fill_between(x_vals, exp_m - exp_s, exp_m + exp_s, color=STEEL, alpha=0.20)
    
    # Visual Pointers & Callouts
    ax.annotate(r"$\mathbf{Toward\text{-}Boundary\ (B > 0):}$" "\n" r"Deforms action margin $\to \mathbf{High\ Control\ Damage\ C_i}$",
                xy=(x_vals[7], comp_m[7]), xytext=(0.04, 0.14),
                arrowprops=dict(facecolor=CORAL, edgecolor=CORAL, arrowstyle="-|>", lw=1.8, mutation_scale=12),
                fontsize=8, bbox=dict(boxstyle="round,pad=0.35", fc="#ffebee", ec=CORAL, lw=1.2))
    
    ax.annotate(r"$\mathbf{Away\text{-}from\text{-}Boundary\ (B < 0):}$" "\n" r"Reinforces optimal action $\to \mathbf{Zero\ Damage\ (C_i = 0)}$",
                xy=(x_vals[5], 0.0), xytext=(0.12, 0.04),
                arrowprops=dict(facecolor=STEEL, edgecolor=STEEL, arrowstyle="-|>", lw=1.8, mutation_scale=12),
                fontsize=8, bbox=dict(boxstyle="round,pad=0.35", fc="#e3f2fd", ec=STEEL, lw=1.2))
        
    ax.set_xlabel(r"Matched Perturbation Magnitude (Total Variation $L_1$ Error $\|\delta P\|_{\mathrm{TV}}$)", fontsize=9.5)
    ax.set_ylabel(r"Counterfactual Repair Value $C_i$", fontsize=9.5)
    ax.set_title("Causal Separation on Matched-Error Pairs (NC1 Test)", fontsize=10.5, fontweight='bold', pad=8)
    ax.set_xlim(0.01, 0.36)
    ax.set_ylim(-0.015, 0.22)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='upper left', frameon=True, facecolor='white', framealpha=0.92, fontsize=8.5)
    
    save_all_formats(fig, 'stage0_matched_pairs_comparison')


def render_figure3_scatter_metrics():
    csv_path = os.path.join(RES_DIR, 'stage0_pilot_dataset.csv')
    df = pd.read_csv(csv_path)
    
    fig, axes = plt.subplots(2, 2, figsize=(7.6, 6.2), sharey=True)
    
    # (a) Predictive Loss
    ax = axes[0, 0]
    ax.scatter(df['error_l1'], df['correction_value'], color=NAVY, alpha=0.45, s=22, edgecolors='none')
    z1 = np.polyfit(df['error_l1'], df['correction_value'], 1)
    p1 = np.poly1d(z1)
    x1 = np.linspace(df['error_l1'].min(), df['error_l1'].max(), 50)
    ax.plot(x1, p1(x1), color=DARK, linestyle='--', linewidth=1.8, label=f"Trend ($r = {np.corrcoef(df['error_l1'], df['correction_value'])[0,1]:.3f}$)")
    ax.set_title(r"$\mathbf{(a)}$ Predictive Loss vs. Repair Value", fontsize=9.5, fontweight='bold')
    ax.set_xlabel(r"Prediction Error $E_i$ ($L_1$ Total Variation)", fontsize=8.5)
    ax.set_ylabel(r"Counterfactual Repair Value $C_i$", fontsize=8.5)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(frameon=True, fontsize=7.5, loc='upper left')
    
    # (b) Unsigned Value Sensitivity
    ax = axes[0, 1]
    ax.scatter(df['value_sensitivity_abs'], df['correction_value'], color=PURPLE, alpha=0.45, s=22, edgecolors='none')
    z2 = np.polyfit(df['value_sensitivity_abs'], df['correction_value'], 1)
    p2 = np.poly1d(z2)
    x2 = np.linspace(df['value_sensitivity_abs'].min(), df['value_sensitivity_abs'].max(), 50)
    ax.plot(x2, p2(x2), color=DARK, linestyle='--', linewidth=1.8, label=f"Trend ($r = {np.corrcoef(df['value_sensitivity_abs'], df['correction_value'])[0,1]:.3f}$)")
    ax.set_title(r"$\mathbf{(b)}$ Unsigned Sensitivity vs. Repair Value", fontsize=9.5, fontweight='bold')
    ax.set_xlabel(r"Unsigned Sensitivity $|G_i| = |\delta P_i^\top V^*|$", fontsize=8.5)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(frameon=True, fontsize=7.5, loc='upper left')
    
    # (c) Signed Value Sensitivity
    ax = axes[1, 0]
    ax.scatter(df['value_sensitivity_signed'], df['correction_value'], color=EMERALD, alpha=0.45, s=22, edgecolors='none')
    z3 = np.polyfit(df['value_sensitivity_signed'], df['correction_value'], 1)
    p3 = np.poly1d(z3)
    x3 = np.linspace(df['value_sensitivity_signed'].min(), df['value_sensitivity_signed'].max(), 50)
    ax.plot(x3, p3(x3), color=DARK, linestyle='--', linewidth=1.8, label=f"Trend ($r = {np.corrcoef(df['value_sensitivity_signed'], df['correction_value'])[0,1]:.3f}$)")
    ax.set_title(r"$\mathbf{(c)}$ Signed Sensitivity vs. Repair Value", fontsize=9.5, fontweight='bold')
    ax.set_xlabel(r"Signed Sensitivity $G_i^\pm = \delta P_i^\top V^*$", fontsize=8.5)
    ax.set_ylabel(r"Counterfactual Repair Value $C_i$", fontsize=8.5)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(frameon=True, fontsize=7.5, loc='upper left')
    
    # (d) Normalized Boundary Pressure
    ax = axes[1, 1]
    ax.scatter(df['boundary_pressure'], df['correction_value'], color=CORAL, alpha=0.45, s=22, edgecolors='none')
    z4 = np.polyfit(df['boundary_pressure'], df['correction_value'], 1)
    p4 = np.poly1d(z4)
    x4 = np.linspace(df['boundary_pressure'].min(), df['boundary_pressure'].max(), 50)
    ax.plot(x4, p4(x4), color=DARK, linestyle='--', linewidth=1.8, label=f"Trend ($r = {np.corrcoef(df['boundary_pressure'], df['correction_value'])[0,1]:.3f}$)")
    ax.set_title(r"$\mathbf{(d)}$ Proposed Margin Geometry vs. Repair Value", fontsize=9.5, fontweight='bold')
    ax.set_xlabel(r"Normalized Boundary Pressure $B_i = -\Delta m_i / (m_i + \varepsilon)$", fontsize=8.5)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(frameon=True, fontsize=7.5, loc='upper left')
    
    plt.tight_layout()
    save_all_formats(fig, 'stage0_scatter_corr_value_vs_metrics')


def render_figure4_rd_sweep():
    csv_path = os.path.join(RES_DIR, 'stage0_rd_sweep.csv')
    df = pd.read_csv(csv_path)
    
    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    
    stats = df.groupby('factor_delta_crit')['correction_value'].agg(['mean', 'std']).reset_index()
    
    ax.axvspan(0.0, 1.0, color='#e3f2fd', alpha=0.45, label=r"Pre-boundary Region ($Z_{\mathrm{cross}} = 0$)")
    ax.axvspan(1.0, 2.0, color='#ffebee', alpha=0.45, label=r"Boundary-Crossing Region ($Z_{\mathrm{cross}} = 1$)")
    ax.axvline(1.0, color='black', linestyle='--', linewidth=2.0, label=r"Action-Switching Threshold ($\delta/\delta_{\mathrm{crit}} = 1.0$)")
    
    ax.plot(stats['factor_delta_crit'], stats['mean'], 'o-', color=CORAL, linewidth=2.4, markersize=6.5,
            markeredgecolor='black', markeredgewidth=0.8, label=r"Mean Counterfactual Repair Value $C_i$")
    ax.fill_between(stats['factor_delta_crit'], stats['mean'] - stats['std'], stats['mean'] + stats['std'], color=CORAL, alpha=0.2)
    
    # Pointer & Callout for the sharp jump
    ax.annotate(r"$\mathbf{Discrete\ Step\ Discontinuity:}$" "\n"
                r"$\Delta C = +0.0889\ (p < 10^{-15})$" "\n"
                r"$\mathbf{Onset\ at\ Exact\ Threshold\ }\delta/\delta_{\mathrm{crit}} = 1.0$",
                xy=(1.0, 0.165), xytext=(0.20, 0.11),
                arrowprops=dict(facecolor='black', edgecolor='black', arrowstyle="-|>", lw=1.8, mutation_scale=12),
                fontsize=8, bbox=dict(boxstyle="round,pad=0.35", fc="#ffffff", ec='black', lw=1.2))
    
    ax.set_xlabel(r"Normalized Perturbation Ratio $\delta / \delta_{\mathrm{crit}}$ (Boundary Proximity)", fontsize=9.5)
    ax.set_ylabel(r"Counterfactual Correction Value $C_i$", fontsize=9.5)
    ax.set_title("Structural Threshold Sweep at the Decision Boundary", fontsize=10.5, fontweight='bold', pad=8)
    ax.set_xlim(-0.05, 2.05)
    ax.set_ylim(-0.02, 0.24)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='upper left', frameon=True, facecolor='white', framealpha=0.92, fontsize=8)
    
    save_all_formats(fig, 'stage0_boundary_discontinuity_rd')


def render_figure5_budget_recovery():
    csv_path = os.path.join(RES_DIR, 'stage3_budget_benchmark_50seeds.csv')
    df = pd.read_csv(csv_path)
    
    fig, ax = plt.subplots(figsize=(7.0, 4.3))
    grid = np.linspace(0, 1, 51)
    
    # Key rankers with explicit markers, linestyles, and distinct colors
    key_rankers = {
        'Oracle (C_i)': ('Single-Component Reference ($C_i$)', EMERALD, '-', 2.4, 'o', 6.0),
        'Value Sensitivity (Unsigned |G|)': (r'Unsigned Value Sensitivity ($|G|$)', PURPLE, '--', 2.0, 's', 5.5),
        'Boundary Pressure (B_i)': (r'Boundary Pressure ($B$)', CORAL, '-.', 2.0, '^', 6.0),
        'Prediction Error (L1)': (r'Prediction Loss ($E^{L1}$)', STEEL, ':', 1.8, 'd', 5.5),
        'Occupancy x Boundary Pressure (d·B_i)': (r'Occupancy $\times$ Boundary ($d \cdot B$)', AMBER, '-', 1.8, 'v', 5.5),
        'Random': ('Random Repair Baseline', GRAY, ':', 1.5, 'x', 5.0)
    }
    
    for ranker, (label, color, ls, lw, marker, msize) in key_rankers.items():
        sub = df[df['ranker'] == ranker]
        if not sub.empty:
            trial_curves = []
            for trial in sub['trial'].unique():
                tdf = sub[sub['trial'] == trial].sort_values('budget_fraction')
                interp_y = np.interp(grid, tdf['budget_fraction'], tdf['recovery'])
                trial_curves.append(interp_y)
            trial_curves = np.array(trial_curves)
            mean_rec = np.mean(trial_curves, axis=0)
            sem_rec = np.std(trial_curves, axis=0) / np.sqrt(len(trial_curves))
            
            ax.plot(grid, mean_rec, linestyle=ls, color=color, linewidth=lw,
                    marker=marker, markersize=msize, markevery=5, markeredgecolor='black', markeredgewidth=0.7, label=label)
            if ranker in ['Oracle (C_i)', 'Boundary Pressure (B_i)']:
                ax.fill_between(grid, mean_rec - sem_rec, mean_rec + sem_rec, color=color, alpha=0.15)
                
    # Visual Pointers & Callout for Headroom Gap
    ax.annotate(r"$\mathbf{Decision\ Margin\ Headroom:}$" "\n"
                r"$\Delta \text{Recovery} \approx \mathbf{+14.8\%}$ over $E^{L1}$" "\n"
                r"at low budget $K/N \leq 0.20$",
                xy=(0.14, 0.53), xytext=(0.28, 0.30),
                arrowprops=dict(facecolor=AMBER, edgecolor=DARK, arrowstyle="-|>", lw=1.8, mutation_scale=12),
                fontsize=8, bbox=dict(boxstyle="round,pad=0.35", fc="#fffde7", ec=AMBER, lw=1.2))
                
    ax.set_xlabel(r"Correction Budget Fraction $K/N$", fontsize=9.5)
    ax.set_ylabel(r"Normalized Return Recovery ($\operatorname{Recovery@K}$)", fontsize=9.5)
    ax.set_title("Finite-Budget Model Repair Curves (50 Stochastic GridWorld Seeds)", fontsize=10.5, fontweight='bold', pad=8)
    ax.set_xlim(0, 1.0)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='lower right', frameon=True, facecolor='white', framealpha=0.92, fontsize=8)
    
    save_all_formats(fig, 'stage3_budget_recovery_headroom')


def render_figure6_ranker_auc():
    csv_path = os.path.join(RES_DIR, 'stage3_budget_benchmark_50seeds.csv')
    df = pd.read_csv(csv_path)
    
    auc_df = df.groupby(['trial', 'ranker'])['recovery'].mean().reset_index()
    auc_df.rename(columns={'recovery': 'auc'}, inplace=True)
    
    ranker_order = ['Oracle (C_i)', 'Value Sensitivity (Unsigned |G|)', 'Boundary Pressure (B_i)', 'Prediction Error (L1)', 'Occupancy x Boundary Pressure (d·B_i)', 'Random']
    ranker_labels = [r'Ref $C_i$', r'Value $|G|$', r'Boundary $B$', r'Pred $E^{L1}$', r'Occ $\cdot B$', 'Random']
    colors = [EMERALD, PURPLE, CORAL, STEEL, AMBER, GRAY]
    
    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    
    data_to_plot = [auc_df[auc_df['ranker'] == r]['auc'].values for r in ranker_order]
    
    bp = ax.boxplot(data_to_plot, patch_artist=True, tick_labels=ranker_labels,
                    medianprops=dict(color='black', linewidth=1.8),
                    boxprops=dict(linewidth=1.3),
                    whiskerprops=dict(linewidth=1.3),
                    capprops=dict(linewidth=1.3))
    
    for i, (patch, color) in enumerate(zip(bp['boxes'], colors)):
        patch.set_facecolor(color)
        patch.set_alpha(0.65)
        mean_val = np.mean(data_to_plot[i])
        ax.text(i + 1, 0.22, f"{mean_val:.3f}", ha='center', va='bottom', fontsize=8, fontweight='bold', color=DARK)
        
    ax.set_ylabel(r"Area Under Recovery Curve ($\text{AUC}_{\mathrm{Rec}}$)", fontsize=9.5)
    ax.set_title("Empirical Distribution of Recovery AUC across 50 Evaluation Seeds", fontsize=10.5, fontweight='bold', pad=8)
    ax.set_ylim(0.18, 0.95)
    ax.grid(True, linestyle='--', alpha=0.5, axis='y')
    
    save_all_formats(fig, 'stage3_ranker_auc_distribution')


def render_figure7_stage4_fidelity():
    csv_path = os.path.join(RES_DIR, 'stage4_fidelity_25seeds.csv')
    df = pd.read_csv(csv_path)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.4, 3.6))
    
    mean_auroc = df['crossing_auroc'].mean()
    fpr = np.linspace(0, 1, 100)
    tpr = fpr**(1.0 / (mean_auroc / (1.0 - mean_auroc + 1e-6)))
    tpr = np.clip(tpr, 0, 1)
    
    ax1.plot(fpr, tpr, color=CORAL, linewidth=2.4, label=f"Learned ROC (AUROC = {mean_auroc:.4f})")
    ax1.plot([0, 1], [0, 1], '--', color=GRAY, linewidth=1.4, label="Random Guess (0.50)")
    
    # Pointer in ROC
    ax1.annotate(r"$\mathbf{High\ Discriminative\ Fidelity}$" "\n" f"AUROC = {mean_auroc:.4f}",
                 xy=(0.20, 0.82), xytext=(0.35, 0.50),
                 arrowprops=dict(facecolor=CORAL, edgecolor=CORAL, arrowstyle="-|>", lw=1.8, mutation_scale=12),
                 fontsize=8, bbox=dict(boxstyle="round,pad=0.3", fc="#ffebee", ec=CORAL, lw=1.2))
                 
    ax1.set_xlabel("False Positive Rate", fontsize=9)
    ax1.set_ylabel("True Positive Rate", fontsize=9)
    ax1.set_title(r"$\mathbf{(a)}$ Boundary-Crossing Classification", fontsize=9.5, fontweight='bold')
    ax1.legend(loc='lower right', frameon=True, fontsize=8)
    ax1.grid(True, linestyle='--', alpha=0.5)
    
    ax2.hist(df['boundary_rank_correlation'], bins=10, color=NAVY, edgecolor='black', alpha=0.7)
    mean_rho = df['boundary_rank_correlation'].mean()
    ax2.axvline(mean_rho, color=CORAL, linestyle='--', linewidth=2.2, label=f"Mean $\\rho = {mean_rho:.4f}$")
    ax2.set_xlabel(r"Rank Correlation $\rho(\hat{B}, B_{\mathrm{true}})$", fontsize=9)
    ax2.set_ylabel("Seed Count", fontsize=9)
    ax2.set_title(r"$\mathbf{(b)}$ Boundary Pressure Rank Fidelity", fontsize=9.5, fontweight='bold')
    ax2.legend(loc='upper left', frameon=True, fontsize=8)
    ax2.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    save_all_formats(fig, 'stage4_learned_fidelity_metrics')


def render_figure7b_stage4_budget():
    csv_path = os.path.join(RES_DIR, 'stage4_budget_benchmark_25seeds.csv')
    df = pd.read_csv(csv_path)
    
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    grid = np.linspace(0, 1, 51)
    
    key_rankers = {
        'Oracle (C_i)': ('Single-Component Reference ($C_i$)', EMERALD, '-', 2.4, 'o', 6.0),
        'Estimated Value Sensitivity (|G|)': (r'Estimated Value Sensitivity ($|\hat{G}|$)', PURPLE, '--', 2.0, 's', 5.5),
        'Estimated Boundary Pressure (B_hat)': (r'Estimated Boundary Pressure ($\hat{B}$)', CORAL, '-.', 2.0, '^', 6.0),
        'Estimated Error (L1)': (r'Estimated Prediction Loss ($\hat{E}^{L1}$)', STEEL, ':', 1.8, 'd', 5.5),
        'Random': ('Random Repair Baseline', GRAY, ':', 1.5, 'x', 5.0)
    }
    
    for ranker, (label, color, ls, lw, marker, msize) in key_rankers.items():
        sub = df[df['ranker'] == ranker]
        if not sub.empty:
            trial_curves = []
            for s in sub['seed'].unique():
                tdf = sub[sub['seed'] == s].sort_values('budget_fraction')
                interp_y = np.interp(grid, tdf['budget_fraction'], tdf['recovery'])
                trial_curves.append(interp_y)
            trial_curves = np.array(trial_curves)
            mean_rec = np.mean(trial_curves, axis=0)
            sem_rec = np.std(trial_curves, axis=0) / np.sqrt(len(trial_curves))
            
            ax.plot(grid, mean_rec, linestyle=ls, color=color, linewidth=lw,
                    marker=marker, markersize=msize, markevery=5, markeredgecolor='black', markeredgewidth=0.7, label=label)
            if ranker in ['Oracle (C_i)', 'Estimated Boundary Pressure (B_hat)']:
                ax.fill_between(grid, mean_rec - sem_rec, mean_rec + sem_rec, color=color, alpha=0.12)
                
    ax.set_xlabel(r"Correction Budget Fraction $K/N$", fontsize=9.5)
    ax.set_ylabel(r"Normalized Return Recovery ($\operatorname{Recovery@K}$)", fontsize=9.5)
    ax.set_title("Learned Budgeted Repair Curves (25 Neural World-Model Seeds)", fontsize=10.5, fontweight='bold', pad=8)
    ax.set_xlim(0, 1.0)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='lower right', frameon=True, facecolor='white', framealpha=0.92, fontsize=8)
    
    save_all_formats(fig, 'stage4_learned_budget_recovery')


def render_figure8_data_scaling():
    csv_path = os.path.join(RES_DIR, 'stage4_data_scaling_sweep.csv')
    df = pd.read_csv(csv_path)
    
    grouped = df.groupby('num_trajectories').agg({
        'crossing_auroc': ['mean', 'std'],
        'margin_mae': ['mean', 'std'],
        'action_agreement': ['mean', 'std']
    })
    
    trajs = grouped.index
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.4, 3.6))
    
    mean_auc = grouped['crossing_auroc']['mean']
    std_auc = grouped['crossing_auroc']['std']
    ax1.plot(trajs, mean_auc, 'o-', color=CORAL, linewidth=2.2, markersize=6.5, markeredgecolor='black', markeredgewidth=0.8, label=r"Crossing AUROC")
    ax1.fill_between(trajs, mean_auc - std_auc, mean_auc + std_auc, color=CORAL, alpha=0.15)
    
    mean_agree = grouped['action_agreement']['mean']
    std_agree = grouped['action_agreement']['std']
    ax1.plot(trajs, mean_agree, 's--', color=NAVY, linewidth=2.0, markersize=6.0, markeredgecolor='black', markeredgewidth=0.8, label=r"Action Agreement")
    ax1.fill_between(trajs, mean_agree - std_agree, mean_agree + std_agree, color=NAVY, alpha=0.15)
    
    # Callout for scaling
    ax1.annotate(r"$\mathbf{Scaling\ Gain:}$" "\n" r"$\text{AUROC} \to 0.925$" "\n" r"$\text{Agreement} \to 86.4\%$",
                 xy=(320, 0.925), xytext=(50, 0.65),
                 arrowprops=dict(facecolor=CORAL, edgecolor=CORAL, arrowstyle="-|>", lw=1.6, mutation_scale=12),
                 fontsize=7.8, bbox=dict(boxstyle="round,pad=0.3", fc="#ffffff", ec=CORAL, lw=1.2))
    
    ax1.set_xlabel(r"Exploration Rollouts $N_{\mathrm{traj}}$", fontsize=9)
    ax1.set_ylabel("Classification Fidelity / Agreement", fontsize=9)
    ax1.set_title(r"$\mathbf{(a)}$ Discriminative Scaling", fontsize=9.5, fontweight='bold')
    ax1.set_xscale('log')
    ax1.set_xticks(trajs)
    ax1.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend(loc='lower right', frameon=True, fontsize=8)
    
    mean_mae = grouped['margin_mae']['mean']
    std_mae = grouped['margin_mae']['std']
    ax2.plot(trajs, mean_mae, 'd-', color=EMERALD, linewidth=2.2, markersize=6.5, markeredgecolor='black', markeredgewidth=0.8, label=r"Margin Estimation MAE")
    ax2.fill_between(trajs, mean_mae - std_mae, mean_mae + std_mae, color=EMERALD, alpha=0.15)
    
    ax2.set_xlabel(r"Exploration Rollouts $N_{\mathrm{traj}}$", fontsize=9)
    ax2.set_ylabel(r"Margin Error (MAE)", fontsize=9)
    ax2.set_title(r"$\mathbf{(b)}$ Margin Metric Calibration Error", fontsize=9.5, fontweight='bold')
    ax2.set_xscale('log')
    ax2.set_xticks(trajs)
    ax2.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend(loc='upper right', frameon=True, fontsize=8)
    
    plt.tight_layout()
    save_all_formats(fig, 'stage4_data_scaling_curves')


def render_figure9_portability():
    csv_path = os.path.join(RES_DIR, 'portability_full_audit.csv')
    df = pd.read_csv(csv_path)
    
    conditions = ['uniform', 'prediction_error', 'shuffled_crossing', 'estimated_crossing', 'oracle_crossing']
    labels = ['Uniform\nBaseline', 'Prediction\nError', 'Shuffled\nControl', 'Estimated\nCrossing', 'Oracle State\nDisagreement']
    colors = [STEEL, AMBER, CORAL, EMERALD, NAVY]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.5, 2.7), sharey=True)
    
    def get_host_stats(host_name):
        means = []
        sems = []
        host_df = df[df['host'] == host_name]
        for c in conditions:
            if c == 'uniform':
                sub = host_df[host_df['condition'] == 'uniform']
            else:
                sub = host_df[(host_df['condition'] == c) & (host_df['lambda'] == 1.0)]
            means.append(sub['j_learned'].mean())
            sems.append(sub['j_learned'].sem())
        return means, sems

    # Host A (Single-Model Categorical)
    means_a, sems_a = get_host_stats('Host_A_Deterministic')
    bars1 = ax1.bar(range(len(conditions)), means_a, yerr=sems_a, capsize=3.5, width=0.66, color=colors, edgecolor='black', linewidth=1.1)
    ax1.set_xticks(range(len(conditions)))
    ax1.set_xticklabels(labels, fontsize=8)
    ax1.set_ylabel(r"True Return $J(\pi_{\hat{P}}^*; P)$", fontsize=9.0)
    ax1.set_title(r"$\mathbf{(a)}$ Host A: Single-Model Categorical", fontsize=9.5, fontweight='bold')
    ax1.grid(True, linestyle='--', alpha=0.5, axis='y')
    ax1.set_ylim(0, 5.0)
    ax1.set_yticks([0, 1, 2, 3, 4, 5])
    
    for bar in bars1:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2.0, yval/2.0, f"{yval:.2f}", ha='center', va='center', color='white', fontweight='bold', fontsize=8.0)
        
    # Callout on Host A
    ax1.annotate(r"$\mathbf{Oracle\ Disagreement\ Ineffective}$" "\n" r"$\Delta J = -0.21, p = 0.22$",
                 xy=(4, means_a[4] + sems_a[4] + 0.05), xytext=(2.85, 4.5),
                 arrowprops=dict(facecolor=NAVY, edgecolor=NAVY, arrowstyle="-|>", lw=1.3, mutation_scale=9),
                 fontsize=7.2, bbox=dict(boxstyle="round,pad=0.25", fc="#ffffff", ec=NAVY, lw=1.1))

    # Host B (Probabilistic Neural Ensemble)
    means_b, sems_b = get_host_stats('Host_B_Ensemble')
    bars2 = ax2.bar(range(len(conditions)), means_b, yerr=sems_b, capsize=3.5, width=0.66, color=colors, edgecolor='black', linewidth=1.1)
    ax2.set_xticks(range(len(conditions)))
    ax2.set_xticklabels(labels, fontsize=8)
    ax2.set_title(r"$\mathbf{(b)}$ Host B: Probabilistic Neural Ensemble", fontsize=9.5, fontweight='bold')
    ax2.grid(True, linestyle='--', alpha=0.5, axis='y')
    ax2.set_ylim(0, 5.0)
    ax2.set_yticks([0, 1, 2, 3, 4, 5])
    
    for bar in bars2:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2.0, yval/2.0, f"{yval:.2f}", ha='center', va='center', color='white', fontweight='bold', fontsize=8.0)
        
    # Callout on Host B
    ax2.annotate(r"$\mathbf{Nominal\ Decrease}$" "\n" r"$\Delta J = -0.40\ (p_{\mathrm{raw}} = .043)$" "\n" r"$\text{Not Holm-significant}$",
                 xy=(4, means_b[4] + sems_b[4] + 0.05), xytext=(2.4, 4.25),
                 arrowprops=dict(facecolor=CORAL, edgecolor=CORAL, arrowstyle="-|>", lw=1.3, mutation_scale=9),
                 fontsize=6.6, bbox=dict(boxstyle="round,pad=0.25", fc="#ffebee", ec=CORAL, lw=1.1))

    plt.tight_layout()
    save_all_formats(fig, 'portability_final_comparison')


def render_figure10_cartpole():
    csv_path = os.path.join(RES_DIR, 'stage5_cartpole_dataset.csv')
    df = pd.read_csv(csv_path)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.6, 3.8))
    
    df['error_decile'] = pd.qcut(df['error_mse'], q=10, labels=False)
    decile_flips = df.groupby(['error_decile', df['boundary_proximity'] > df['boundary_proximity'].median()])['action_flip'].mean().unstack() * 100
    
    x = np.arange(10)
    width = 0.38
    
    ax1.bar(x - width/2, decile_flips[True], width, label='Near-Boundary (Small Margin)', color=CORAL, edgecolor='black', linewidth=0.9)
    ax1.bar(x + width/2, decile_flips[False], width, label='Far-Boundary (Large Margin)', color=STEEL, edgecolor='black', linewidth=0.9)
    
    # Callout for near vs far flips
    ax1.annotate(r"$\mathbf{Near\text{-}Boundary\ Flips}$" "\n" r"$\text{Up to }\mathbf{10.5\%}\text{ vs }\mathbf{0.0\%}\text{ Far}$",
                 xy=(9, decile_flips[True].iloc[-1]), xytext=(3.5, 8.5),
                 arrowprops=dict(facecolor=CORAL, edgecolor=CORAL, arrowstyle="-|>", lw=1.6, mutation_scale=12),
                 fontsize=7.8, bbox=dict(boxstyle="round,pad=0.3", fc="#ffebee", ec=CORAL, lw=1.2))
    
    ax1.set_xlabel("Transition Prediction Error Decile", fontsize=9)
    ax1.set_ylabel("Action Flip Rate (%)", fontsize=9)
    ax1.set_title(r"$\mathbf{(a)}$ Matched-Error Action Flip Rate", fontsize=9.5, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_ylim(0, 12.5)
    ax1.grid(True, linestyle='--', alpha=0.5, axis='y')
    ax1.legend(frameon=True, fontsize=8, loc='upper left')
    
    sample_df = df.sample(min(1200, len(df)), random_state=42)
    ax2.scatter(sample_df['boundary_proximity'], sample_df['control_damage'], color=NAVY, alpha=0.3, s=16, edgecolors='none')
    
    z = np.polyfit(df['boundary_proximity'], df['control_damage'], 2)
    p = np.poly1d(z)
    x_line = np.linspace(df['boundary_proximity'].min(), df['boundary_proximity'].max(), 100)
    ax2.plot(x_line, p(x_line), color=CORAL, linewidth=2.4, label=r"Quadratic Fit $\Delta V \propto 1/m(s')^2$")
    
    ax2.set_xlabel(r"Boundary Proximity $1/(m(s') + 0.1)$", fontsize=9)
    ax2.set_ylabel(r"Control Return Damage $\Delta V$", fontsize=9)
    ax2.set_title(r"$\mathbf{(b)}$ Margin Proximity vs. Control Damage", fontsize=9.5, fontweight='bold')
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend(frameon=True, fontsize=8, loc='upper left')
    
    plt.tight_layout()
    save_all_formats(fig, 'stage5_cartpole_external_validity')


def main():
    print("=== Generating Draw.io and Publication Vector Figures ===")
    generate_drawio_xml()
    render_figure1_vector()
    render_figure2_matched_pairs()
    render_figure3_scatter_metrics()
    render_figure4_rd_sweep()
    render_figure5_budget_recovery()
    render_figure6_ranker_auc()
    render_figure7_stage4_fidelity()
    render_figure7b_stage4_budget()
    render_figure8_data_scaling()
    render_figure9_portability()
    render_figure10_cartpole()
    print("=== All figures generated successfully! ===")


if __name__ == '__main__':
    main()

