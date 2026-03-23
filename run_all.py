import os
import subprocess
from pathlib import Path

methods = ['fp', 'attach', 'sam3d', 'ua', 'any6d']
videos = ['AP10','AP11', 'AP12', 'AP13', 'AP14', 'MPM10', 'MPM11', 'MPM12', 'MPM13', 'MPM14', 'SB11', 'SB13', 'SM1']

for video in videos:
    for method in methods:
        print(f"\n{'='*60}")
        print(f'Running {video} with {method}')
        print(f"{'='*60}\n")
        if method == 'attach':
            subprocess.run([
                "python", "run_attachment.py",
                "--video_id", video
            ])
        else:
            subprocess.run([
                "python", "run_demo.py",
                "--video_id", video,
                "--method", method
            ])
        subprocess.run([
            "python", "make_metrics.py",
            "--video_id", video,
            "--method", method
        ])