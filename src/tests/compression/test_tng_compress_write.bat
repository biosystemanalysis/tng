@echo off
setlocal enableextensions enabledelayedexpansion
SET /A I=0
:start
SET /A I+=1
test_tng_compress_gen%I%
IF "%I%" == "64" (
  GOTO end
) ELSE (
  GOTO start
)
:end
endlocal
