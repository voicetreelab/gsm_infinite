import matplotlib.pyplot as plt
import numpy as np
import svgwrite

# Create an SVG drawing
dwg = svgwrite.Drawing('static/images/GSMmouse.svg', profile='tiny')

# Scaling factor to match TikZ proportions
scale = 100

# Define colors
pink = "rgb(255,182,193)"
red = "rgb(255,0,0)"
light_red = "rgb(255,102,102)"
black = "rgb(0,0,0)"
white = "rgb(255,255,255)"

# Body (ellipse)
dwg.add(dwg.ellipse(center=(0, 0), r=(1.5*scale, 1.2*scale), stroke=black, fill="none", stroke_width=2))

# Ears
dwg.add(dwg.circle(center=(-1.2*scale, 1.1*scale), r=0.6*scale, stroke=black, fill="none", stroke_width=2))
dwg.add(dwg.circle(center=(1.2*scale, 1.1*scale), r=0.6*scale, stroke=black, fill="none", stroke_width=2))

# Inner ears (pink rings)
dwg.add(dwg.circle(center=(-1.2*scale, 1.1*scale), r=0.4*scale, stroke=pink, fill="none", stroke_width=5))
dwg.add(dwg.circle(center=(1.2*scale, 1.1*scale), r=0.4*scale, stroke=pink, fill="none", stroke_width=5))

# Eyes: White background
dwg.add(dwg.circle(center=(-0.5*scale, 0.3*scale), r=0.25*scale, stroke=black, fill=white, stroke_width=2))
dwg.add(dwg.circle(center=(0.5*scale, 0.3*scale), r=0.25*scale, stroke=black, fill=white, stroke_width=2))

# Pupils
dwg.add(dwg.circle(center=(-0.5*scale, 0.3*scale), r=0.1*scale, stroke=black, fill=black, stroke_width=1))
dwg.add(dwg.circle(center=(0.5*scale, 0.3*scale), r=0.1*scale, stroke=black, fill=black, stroke_width=1))

# Eye highlights
dwg.add(dwg.circle(center=(-0.52*scale, 0.33*scale), r=0.03*scale, stroke=white, fill=white, stroke_width=1))
dwg.add(dwg.circle(center=(0.48*scale, 0.33*scale), r=0.03*scale, stroke=white, fill=white, stroke_width=1))

# Infinity-symbol nose
dwg.add(dwg.circle(center=(-0.2*scale, -0.1*scale), r=0.2*scale, stroke=red, fill="none", stroke_width=3))
dwg.add(dwg.circle(center=(0.2*scale, -0.1*scale), r=0.2*scale, stroke=red, fill="none", stroke_width=3))

# Whiskers
dwg.add(dwg.line(start=(-0.1*scale, -0.1*scale), end=(-1.0*scale, -0.3*scale), stroke=black, stroke_width=2))
dwg.add(dwg.line(start=(-0.1*scale, -0.2*scale), end=(-1.0*scale, -0.5*scale), stroke=black, stroke_width=2))
dwg.add(dwg.line(start=(0.1*scale, -0.1*scale), end=(1.0*scale, -0.3*scale), stroke=black, stroke_width=2))
dwg.add(dwg.line(start=(0.1*scale, -0.2*scale), end=(1.0*scale, -0.5*scale), stroke=black, stroke_width=2))

# Smile (Bezier curve approximation)
dwg.add(dwg.path(d="M-60,-20 Q-20,-60 0,-40 T60,-20", stroke=light_red, fill="none", stroke_width=3))

# Tongue
dwg.add(dwg.ellipse(center=(0, -0.4*scale), r=(0.2*scale, 0.1*scale), stroke="none", fill=light_red))

# Save the SVG file
dwg.save()

# Provide download link
"static/images/GSMmouse.svg" 
