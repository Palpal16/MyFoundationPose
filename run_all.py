import os
import subprocess
from pathlib import Path

methods = ['sam3d', 'fp', 'attach', 'any6d']
videos = ['SM1', 'SB13', 'AP14', 'MPM10']

for video in videos:
    for method in methods:
        '''subprocess.run([
            "python", "run_demo.py",
            "--video_id", video,
            "--method", method
        ])'''
        subprocess.run([
            "python", "make_metrics.py",
            "--video_id", video,
            "--method", method
        ])
subprocess.run(["python", "analyze_results.py"])