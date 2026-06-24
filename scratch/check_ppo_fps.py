from datetime import datetime

t_start = datetime.strptime("12/06/2026 01:57:46", "%d/%m/%Y %H:%M:%S")
t_end = datetime.strptime("12/06/2026 08:32:36", "%d/%m/%Y %H:%M:%S")
duration_sec = (t_end - t_start).total_seconds()
steps = 27998208 # 28M

fps = steps / duration_sec
print(f"PPO Total Sec: {duration_sec}")
print(f"PPO FPS: {fps}")
print(f"Time to 16M (90%): {16000000 / fps / 3600} hours")

sac_sec = 10382
sac_steps = 2000000
sac_fps = sac_steps / sac_sec
print(f"SAC Total Sec: {sac_sec}")
print(f"SAC FPS: {sac_fps}")
print(f"Time to 2M (90%): {2000000 / sac_fps / 3600} hours")

