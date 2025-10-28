@echo off
set "DIR=C:\Users\korea\Documents\GitHub\gen-clear-drive\datasets\yolo_bdd100k\clear_synth_night\labels\test"

pushd "%DIR%"
for %%F in (*.txt) do (
    ren "%%F" "%%~nF_fake_B.txt"
)
popd
echo Done!
