#!/bin/bash
REM for %%i in (*.jpg) do (echo %%i)

set a=12001                                # define an incremental variable
setlocal EnableDelayedExpansion   
for %%i in (*.jpg) do (
echo %%i
set /A a+=0
ren "%%i" "!a!.jpg"
set /A a+=1
)


