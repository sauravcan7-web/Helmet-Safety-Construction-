# Construction Safety Helmet Monitor

Real-time AI-powered helmet detection system for construction sites.  
Detects workers, counts helmets / no-helmets, raises visual alarm on violations.  
Built with YOLOv8 + Streamlit for easy demo and deployment.



## Features
- Upload video or use demo clip → live helmet detection
- Counts: total workers, helmets, workers without helmet
- Red boxes + "NO HELMET" labels on violators
- Clear violation status (red/green) with alarm indicator
- Looping video mode simulates 24/7 monitoring
- Adjustable confidence threshold via slider
- Clean, browser-based interface (no local install needed)

## Demo Video Used
Pexels free stock footage (royalty-free):  
Example clips from construction sites with workers ± helmets.

## Tech Stack
- **YOLOv8** (Ultralytics) – object detection  
- **Streamlit** – web app interface  
- **OpenCV** – video processing

## Quick Start (Local)
1. Clone the repo
   ```bash
   git clone https://github.com/YOUR_USERNAME/construction-safety-monitor.git
   cd construction-safety-monitor
