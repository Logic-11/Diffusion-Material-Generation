# Dot-source this file before running Python in the diffusion environment.
$codexCondaRoot = 'D:\Anaconda\anaconda3'
$codexCondaLauncher = Join-Path $codexCondaRoot 'Lib\site-packages\setuptools\cli-64.exe'
$codexCondaSourceScript = Join-Path $codexCondaRoot 'Scripts\conda-script.py'
$codexCondaTempDir = Join-Path ([IO.Path]::GetTempPath()) ('codex-diffusion-conda-' + [Guid]::NewGuid().ToString('N'))
$codexCondaTempExe = Join-Path $codexCondaTempDir 'conda.exe'
$codexCondaTempScript = Join-Path $codexCondaTempDir 'conda-script.py'
$codexCondaPreviousExe = $env:CONDA_EXE
$codexCondaPreviousRootExe = $env:_CONDA_EXE
$codexCondaSetupComplete = $false

try {
    New-Item -ItemType Directory -Path $codexCondaTempDir -ErrorAction Stop | Out-Null
    Copy-Item -LiteralPath $codexCondaLauncher -Destination $codexCondaTempExe -ErrorAction Stop
    $codexCondaSource = [IO.File]::ReadAllText($codexCondaSourceScript)
    $codexCondaShebang = '#!D:\Anaconda\anaconda3\python.exe' + "`r`n"
    [IO.File]::WriteAllText($codexCondaTempScript, $codexCondaShebang + $codexCondaSource, [Text.UTF8Encoding]::new($false))

    . (Join-Path $codexCondaRoot 'shell\condabin\conda-hook.ps1')
    $env:CONDA_EXE = $codexCondaTempExe
    $env:_CONDA_EXE = $codexCondaTempExe
    conda activate diffusion

    if ($LASTEXITCODE -ne 0) {
        throw "Conda activation failed with exit code $LASTEXITCODE."
    }
    if ($env:CONDA_DEFAULT_ENV -ne 'diffusion') {
        throw 'The diffusion conda environment is not active.'
    }
    $codexCondaPython = Get-Command python -ErrorAction Stop | Select-Object -First 1
    if ($codexCondaPython.CommandType -ne 'Application' -or $codexCondaPython.Source -ne 'D:\Anaconda\anaconda3\envs\diffusion\python.exe') {
        throw "Unexpected Python executable: $($codexCondaPython.Source)"
    }

    # Activation restores Conda's original launcher; restore the compatible
    # launcher for follow-up conda commands in the active environment.
    $env:CONDA_EXE = $codexCondaTempExe
    $env:_CONDA_EXE = $codexCondaTempExe
    $codexCondaCleanup = @($codexCondaTempExe, $codexCondaTempScript, $codexCondaTempDir)
    Register-EngineEvent -SourceIdentifier PowerShell.Exiting -MessageData $codexCondaCleanup -Action {
        foreach ($codexCondaPath in $event.MessageData) {
            Remove-Item -LiteralPath $codexCondaPath -Force -ErrorAction SilentlyContinue
        }
    } | Out-Null
    $codexCondaSetupComplete = $true
}
finally {
    if (-not $codexCondaSetupComplete) {
        $env:CONDA_EXE = $codexCondaPreviousExe
        $env:_CONDA_EXE = $codexCondaPreviousRootExe
        foreach ($codexCondaPath in @($codexCondaTempExe, $codexCondaTempScript, $codexCondaTempDir)) {
            Remove-Item -LiteralPath $codexCondaPath -Force -ErrorAction SilentlyContinue
        }
    }
}
