$PPO_PATH = "ur3_push_data\rl\MujocoUR3PushEnv_ppo_teste_fps_mesa_20260625_223152"
$SAC_PATH = "ur3_push_data\rl\MujocoUR3PushEnv_sac_teste_fps_mesa_20260626_015332"

Write-Host "Gerando video para o PPO..." -ForegroundColor Cyan
.\.venv\Scripts\python.exe -m ur3_push_rl_sb3.evaluate_warp_video --evalPath $PPO_PATH --eVideoName "video_ppo_otimizado" --numStackedObs 4 --eNumEvalEpisodes 8

Write-Host "Gerando video para o SAC..." -ForegroundColor Yellow
.\.venv\Scripts\python.exe -m ur3_push_rl_sb3.evaluate_warp_video --evalPath $SAC_PATH --eVideoName "video_sac_otimizado" --numStackedObs 4 --eNumEvalEpisodes 8

Write-Host "Videos gerados com sucesso na pasta videos/" -ForegroundColor Green
