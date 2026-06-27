Write-Host "==============================================" -ForegroundColor Cyan
Write-Host " INICIANDO TREINAMENTO PPO (30M + STACKED OBS) " -ForegroundColor Cyan
Write-Host "==============================================" -ForegroundColor Cyan

python -m ur3_push_rl_sb3.train --algorithm ppo --useWarp 1 --numTrain 4096 --batchSize 1024 --nEpochs 20 --totalLearningTimesteps 30000000 --normalizeReward 1 --ppolr 0.0005 --entCoef 0.01 --evalFreq 244 --nEvalEpisodes 50 --nSteps 128 --torquePenaltyScale 0.005 --manipulabilityRewardScale 0.01 --actionRatePenaltyScale 0.001 --successBonus 2.0 --earlyTerminationOnSuccess 1 --randomizeInitialJoints 1 --actionScalingFactor 0.035 --maxEpisodeSteps 200 --numStackedObs 4 --commentLogPath "ppo_definitivo_30M_STACKED"

if ($LASTEXITCODE -ne 0) {
    Write-Host "Erro no treinamento do PPO. O script foi abortado antes de iniciar o SAC." -ForegroundColor Red
    exit $LASTEXITCODE
}

Write-Host "`n==============================================" -ForegroundColor Yellow
Write-Host " INICIANDO TREINAMENTO SAC (10M + WARP GPU)    " -ForegroundColor Yellow
Write-Host "==============================================" -ForegroundColor Yellow

python -m ur3_push_rl_sb3.train --algorithm sac --useWarp 1 --numTrain 128 --totalLearningTimesteps 10000000 --normalizeReward 1 --sacLr 0.0003 --sacBufferSize 1000000 --sacBatchSize 512 --sacTau 0.005 --sacGamma 0.99 --sacLearningStarts 10000 --sacTrainFreq 1 --sacGradientSteps 32 --evalFreq 3906 --nEvalEpisodes 50 --torquePenaltyScale 0.005 --manipulabilityRewardScale 0.01 --actionRatePenaltyScale 0.001 --successBonus 2.0 --earlyTerminationOnSuccess 1 --randomizeInitialJoints 1 --actionScalingFactor 0.035 --maxEpisodeSteps 200 --numStackedObs 4 --commentLogPath "sac_definitivo_10M_Warp_GPU"

if ($LASTEXITCODE -ne 0) {
    Write-Host "Erro no treinamento do SAC." -ForegroundColor Red
    exit $LASTEXITCODE
}

Write-Host "`n==============================================" -ForegroundColor Green
Write-Host " TODOS OS TREINAMENTOS FORAM CONCLUIDOS!      " -ForegroundColor Green
Write-Host "==============================================" -ForegroundColor Green
