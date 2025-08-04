@echo off
setlocal enabledelayedexpansion

rem Define seeds
set seeds=1 2 3

for %%K in (0 1) do (
    for %%S in (%seeds%) do (
        echo Running with correlated_KL=%%K, seed=%%S
        python10 "C:\PostDoc Research\Projects\AERCA\main.py" --correlated_KL=%%K --seed=%%S --dataset_name "msds" 
    )
)

endlocal
pause