# Python / Conda environment

Run `conda activate diffusion` before running Python commands.

## PowerShell initialization

Use the helper below for regular Python commands so activation can run inside
the sandbox without requesting additional permissions.

The agent runs PowerShell 7. The existing Conda initialization is in
`C:\Users\26525\Documents\WindowsPowerShell\profile.ps1`, which is not loaded
by the current PowerShell 7 session. Without initialization, `conda` resolves
to `conda.bat` rather than the PowerShell activation function.

From the `diffusion_mattergen` project directory, dot-source the sandbox-compatible
activation helper in the same shell invocation that runs Python; activation
does not persist across separate tool calls:

```powershell
. './scripts/activate_diffusion.ps1'
if ($env:CONDA_DEFAULT_ENV -ne 'diffusion') {
    throw 'The diffusion conda environment is not active.'
}
python -B -c "import sys; print(sys.executable)"
```

The expected Python executable is:
`D:\Anaconda\anaconda3\envs\diffusion\python.exe`.
Do not continue with Python if activation fails or the executable is incorrect.
If you deactivate the environment, dot-source the helper again before the next
activation.

## Known sandbox startup failure

The installed `conda.exe` lacks PE relocation information and can fail to
start in the agent sandbox with `无效的系统 DLL 重定位` (invalid system DLL
relocation). The activation helper uses the relocatable setuptools launcher
already installed in the same Conda distribution with its existing Conda
entry point. It creates a temporary launcher for the current PowerShell
session and calls `conda activate diffusion` before Python is run.

If the helper itself fails because of the sandbox, follow the applicable
approval policy for any necessary rerun outside the sandbox. Keep activation
and the Python command together. This file does not grant approval to bypass
the sandbox or change system configuration.

Do not diagnose this known startup error as a broken `diffusion` environment,
reinstall packages, or modify PowerShell profiles or system DLLs merely to
work around it. Distinguish missing PowerShell initialization from executable
startup failure when reporting problems.


## Task Details
We want to reproduce the results of MatterGen, so in EVERY part of this project,
follow the methods and conventions that MatterGen used in their work. However, 
you don't have to completely copy their source code.