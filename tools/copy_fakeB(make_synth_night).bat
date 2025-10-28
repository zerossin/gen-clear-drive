@echo off
set SRC=C:\Users\korea\Documents\GitHub\gen-clear-drive\pytorch-CycleGAN-and-pix2pix\results\clear_d2n_256_e200_k10k\test_latest\images
set DST=C:\Users\korea\Documents\GitHub\gen-clear-drive\datasets\yolo_bdd100k\clear_synth_night\images\test

if not exist "%DST%" mkdir "%DST%"

for %%F in ("%SRC%\*_fake_B.png") do (
    copy /Y "%%F" "%DST%"
)
echo Done!