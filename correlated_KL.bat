@echo off
setlocal enabledelayedexpansion

rem Define seeds
set seeds=1
set gloabl_attention_over_all_lag = 0 1

for %%K in (0 1) do (
    for %%G in (%gloabl_attention_over_all_lag%) do (
        for %%S in (%seeds%) do (
            echo Running with correlated_KL=%%K, seed=%%S, gloabl_attention_over_all_lag=%%G
            python10 "C:\PostDoc Research\Projects\AERCA\main.py" --correlated_KL=%%K --seed=%%S --dataset_name "msds" --gloabl_attention_over_all_lag=%%G
        )
    )
    
)

endlocal
pause