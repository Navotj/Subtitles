# Puts a "D&D Subtitles" icon on the desktop that starts the app (no CLI window).
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$desktop = [Environment]::GetFolderPath('Desktop')
$ws = New-Object -ComObject WScript.Shell
$lnk = $ws.CreateShortcut((Join-Path $desktop 'D&D Subtitles.lnk'))
$lnk.TargetPath = Join-Path $here '.venv\Scripts\pythonw.exe'
$lnk.Arguments = '"' + (Join-Path $here 'app.py') + '"'
$lnk.WorkingDirectory = $here
$lnk.IconLocation = (Join-Path $here 'app.ico') + ',0'
$lnk.Description = 'Live D&D subtitles'
$lnk.Save()
Write-Host "Shortcut created on $desktop"
