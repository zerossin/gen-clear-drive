@echo off
set "DIR=C:\Users\korea\Documents\GitHub\gen-clear-drive\datasets\yolo_bdd100k\clear_synth_night\labels\test"

pushd "%DIR%"
for %%F in (*_fake_B*.txt) do (
    set "name=%%~nF"
    setlocal enabledelayedexpansion
    ren "%%F" "!name:_fake_B=!.txt"
    endlocal
)
popd
echo All _fake_B removed.
