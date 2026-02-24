"""为 scenarios/ 下的每个场景文件在 saved_fig/ 中创建对应子目录。"""

import os

script_dir = os.path.dirname(os.path.abspath(__file__))
scenarios_dir = os.path.join(script_dir, "scenarios")
saved_fig_dir = os.path.join(script_dir, "saved_fig")

os.makedirs(saved_fig_dir, exist_ok=True)

created = 0
for f in sorted(os.listdir(scenarios_dir)):
    if f.endswith(".xml"):
        benchmark_id = f.replace(".xml", "")
        target = os.path.join(saved_fig_dir, benchmark_id)
        if not os.path.exists(target):
            os.makedirs(target)
            created += 1
            print(f"Created: {target}")
        else:
            print(f"Exists:  {target}")

print(f"\nDone. {created} new directories created.")
