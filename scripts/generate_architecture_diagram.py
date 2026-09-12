"""
Generate a professional, high-resolution architecture diagram for DECIVERSE Decision Agent.
"""
from PIL import Image, ImageDraw, ImageFont
import os

WIDTH = 1400
HEIGHT = 1600
BG_COLOR = (15, 23, 42)      # #0f172a slate-900
PANEL_COLOR = (30, 41, 59)   # #1e293b slate-800
BORDER_COLOR = (56, 189, 248) # #38bdf8 sky-400
TEXT_MAIN = (248, 250, 252)  # #f8fafc slate-50
TEXT_MUTED = (148, 163, 184) # #94a3b8 slate-400
ACCENT_PURPLE = (168, 85, 247) # #a855f7
ACCENT_PINK = (236, 72, 153)  # #ec4899
ACCENT_TEAL = (45, 212, 191)  # #2dd4bf
ACCENT_GREEN = (34, 197, 94)  # #22c55e
ACCENT_RED = (239, 68, 68)    # #ef4444
ACCENT_ROSE = (244, 63, 94)   # #f43f5e
ACCENT_ORANGE = (251, 146, 60) # #fb923c

img = Image.new("RGB", (WIDTH, HEIGHT), BG_COLOR)
draw = ImageDraw.Draw(img)

# Try loading standard windows font or fallback to default
try:
    font_title = ImageFont.truetype("arial.ttf", 36)
    font_subtitle = ImageFont.truetype("arial.ttf", 20)
    font_box_title = ImageFont.truetype("arial.ttf", 22)
    font_box_desc = ImageFont.truetype("arial.ttf", 15)
    font_arrow = ImageFont.truetype("arial.ttf", 16)
except Exception:
    font_title = ImageFont.load_default()
    font_subtitle = font_title
    font_box_title = font_title
    font_box_desc = font_title
    font_arrow = font_title

# Helper functions
def draw_card(x, y, w, h, title, subtitle="", border=BORDER_COLOR, bg=PANEL_COLOR):
    # Rounded box
    draw.rounded_rectangle([x, y, x + w, y + h], radius=16, fill=bg, outline=border, width=3)
    # Title
    draw.text((x + w // 2, y + 22), title, fill=TEXT_MAIN, font=font_box_title, anchor="mm")
    if subtitle:
        draw.text((x + w // 2, y + 50), subtitle, fill=TEXT_MUTED, font=font_box_desc, anchor="mm")

def draw_v_arrow(x, y1, y2, color=BORDER_COLOR, label=""):
    draw.line([(x, y1), (x, y2)], fill=color, width=3)
    draw.polygon([(x - 8, y2 - 10), (x + 8, y2 - 10), (x, y2)], fill=color)
    if label:
        draw.text((x + 15, (y1 + y2) // 2), label, fill=color, font=font_arrow, anchor="lm")

def draw_h_arrow(x1, y1, x2, y2, color=BORDER_COLOR):
    draw.line([(x1, y1), (x2, y2)], fill=color, width=3)
    if x2 > x1:
        draw.polygon([(x2 - 10, y2 - 8), (x2 - 10, y2 + 8), (x2, y2)], fill=color)
    else:
        draw.polygon([(x2 + 10, y2 - 8), (x2 + 10, y2 + 8), (x2, y2)], fill=color)

# Header
draw.text((WIDTH // 2, 55), "DECIVERSE Decision Agent — Architecture", fill=TEXT_MAIN, font=font_title, anchor="mm")
draw.text((WIDTH // 2, 95), "Tech Zephyr 4.0 — IIT Bhubaneswar Agentic AI Hackathon", fill=TEXT_MUTED, font=font_subtitle, anchor="mm")

# 1. User Goal
draw_card(350, 140, 700, 75, "USER GOAL & CONSTRAINTS", "Policy Objective + Target Rules (e.g. polarization<0.35, adoption>0.50)", border=BORDER_COLOR, bg=(3, 105, 161))
draw_v_arrow(700, 215, 270, color=BORDER_COLOR)

# 2. Autonomous Controller
draw_card(300, 270, 800, 80, "AUTONOMOUS CONTROLLER", "Closed-Loop Orchestrator, Persistent DecisionState, Iteration Budgets", border=(129, 140, 248), bg=(30, 27, 75))

# Branches to Planner & Replanner
draw.line([(700, 350), (700, 385)], fill=(129, 140, 248), width=3)
draw.line([(380, 385), (1020, 385)], fill=(129, 140, 248), width=3)
draw_v_arrow(380, 385, 420, color=ACCENT_PURPLE)
draw_v_arrow(1020, 385, 420, color=ACCENT_PINK)

# 3. Planner & Replanner
draw_card(180, 420, 400, 80, "ACTION UTILITY PLANNER", "7-Factor Continuous Utility Scoring Formula", border=ACCENT_PURPLE, bg=PANEL_COLOR)
draw_card(820, 420, 400, 80, "DYNAMIC REPLANNER", "Constraint Deficit Diagnosis & Policy Mutations", border=ACCENT_PINK, bg=PANEL_COLOR)

# Merge down to Tool Registry
draw.line([(380, 500), (380, 545)], fill=ACCENT_PURPLE, width=3)
draw.line([(1020, 500), (1020, 545)], fill=ACCENT_PINK, width=3)
draw.line([(380, 545), (1020, 545)], fill=ACCENT_TEAL, width=3)
draw_v_arrow(700, 545, 580, color=ACCENT_TEAL)

# 4. Agent Tool Registry
draw_card(320, 580, 760, 75, "AGENT TOOL REGISTRY", "Execution Routing, Parameter Validation, Failure Trapping", border=ACCENT_TEAL, bg=(15, 118, 110))

# 3-way Tool Dispatch
draw.line([(700, 655), (700, 690)], fill=ACCENT_TEAL, width=3)
draw.line([(250, 690), (1150, 690)], fill=ACCENT_TEAL, width=3)
draw_v_arrow(250, 690, 725, color=BORDER_COLOR)
draw_v_arrow(700, 690, 725, color=BORDER_COLOR)
draw_v_arrow(1150, 690, 725, color=BORDER_COLOR)

# 5. Tools
draw_card(100, 725, 300, 75, "CONTEXT TOOLS", "Ontology & Digital Twins", border=BORDER_COLOR, bg=PANEL_COLOR)
draw_card(550, 725, 300, 75, "SCENARIO TOOLS", "Formulation & Policy Mutations", border=BORDER_COLOR, bg=PANEL_COLOR)
draw_card(1000, 725, 300, 75, "SIMULATION TOOLS", "Subprocess Runner & IPC", border=BORDER_COLOR, bg=PANEL_COLOR)

# Simulation Engine Flow
draw_v_arrow(1150, 800, 845, color=ACCENT_ROSE)
draw_card(950, 845, 400, 75, "DECIVERSE / OASIS SOCIETY", "Multi-Agent Social Network Interaction Engine", border=ACCENT_ROSE, bg=(131, 24, 67))

draw_v_arrow(1150, 920, 965, color=ACCENT_ORANGE)
draw_card(950, 965, 400, 75, "EMERGENT BEHAVIOURS", "Actions, Sentiment, Stances, Polarization", border=ACCENT_ORANGE, bg=PANEL_COLOR)

# To Social Impact Model
draw.line([(1150, 1040), (1150, 1075)], fill=ACCENT_ORANGE, width=3)
draw.line([(700, 1075), (1150, 1075)], fill=ACCENT_ORANGE, width=3)
draw_v_arrow(700, 1075, 1110, color=(165, 180, 252))

# 6. Social Impact Model
draw_card(350, 1110, 700, 75, "SOCIAL IMPACT MODEL", "Multidimensional Consequence Scores (Acceptance, Conflict, Equity, Adoption)", border=(165, 180, 252), bg=(49, 46, 129))

# To Verifier
draw_v_arrow(700, 1185, 1225, color=ACCENT_GREEN)

# 7. Verifier
draw_card(420, 1225, 560, 75, "INTEGRATED VERIFIER", "Strict Mathematical Inequality & Constraint Reconciliation", border=ACCENT_GREEN, bg=(19, 78, 74))

# Verification Outcomes
draw.line([(700, 1300), (700, 1340)], fill=BORDER_COLOR, width=3)
draw.line([(380, 1340), (1020, 1340)], fill=BORDER_COLOR, width=3)
draw_v_arrow(380, 1340, 1380, color=ACCENT_GREEN, label="  ALL PASS")
draw_v_arrow(1020, 1340, 1380, color=ACCENT_RED, label="  VIOLATION")

# 8. Outcomes
draw_card(180, 1380, 400, 85, "VERIFIED: FINAL RESULT", "Optimal Policy + 7 Audit Artifacts Exported", border=ACCENT_GREEN, bg=(6, 78, 59))
draw_card(820, 1380, 400, 85, "FAILED / UNCERTAIN", "Prescribe Root-Deficit Policy Mutations", border=ACCENT_RED, bg=(127, 29, 29))

# Dynamic Replanning Loop back up to Replanner
draw.line([(1220, 1422), (1340, 1422)], fill=ACCENT_PINK, width=3)
draw.line([(1340, 1422), (1340, 460)], fill=ACCENT_PINK, width=3)
draw_h_arrow(1340, 460, 1220, 460, color=ACCENT_PINK)
draw.text((1350, 940), "DYNAMIC REPLANNING FEEDBACK LOOP", fill=ACCENT_PINK, font=font_arrow, anchor="mm", rotation=270 if hasattr(draw, 'rotation') else None)

img.save("architecture.png", "PNG")
print("Saved architecture.png (1400x1600) successfully!")
