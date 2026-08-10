# Register the Sports board auto-update on Windows Task Scheduler.
#
#   .\setup_schedule.ps1            # install
#   .\setup_schedule.ps1 -Remove    # uninstall
#
# Two tasks, because the chain has two halves with very different costs:
#
#   TaraSportsLight  every 30 min   settle finished games, re-price, publish
#   TaraSportsFull   daily 07:30    refresh sources + re-predict, then the above
#
# Both run whether or not you are logged in is NOT set on purpose: these run as
# the current user in the user's session so they inherit the same environment
# the scripts were developed against, including the venv and any git credentials.
#
# The light task publishes only when the board actually changed, so a quiet
# afternoon produces no commits and no Vercel rebuilds.

param([switch]$Remove)

$py      = "C:\Users\Gohan\.venvs\sports-model\Scripts\python.exe"
$script  = "C:\Users\Gohan\OneDrive\Documents\Gopal\sports-model\src\auto_update.py"
$workdir = "C:\Users\Gohan\OneDrive\Documents\Gopal\sports-model"
$light   = "TaraSportsLight"
$full    = "TaraSportsFull"

if ($Remove) {
    foreach ($n in @($light, $full)) {
        try {
            Unregister-ScheduledTask -TaskName $n -Confirm:$false -ErrorAction Stop
            Write-Output "removed $n"
        } catch { Write-Output "$n was not registered" }
    }
    return
}

if (-not (Test-Path $py))     { throw "python not found at $py" }
if (-not (Test-Path $script)) { throw "auto_update.py not found at $script" }

# Idempotent: re-running replaces rather than duplicating.
foreach ($n in @($light, $full)) {
    try { Unregister-ScheduledTask -TaskName $n -Confirm:$false -ErrorAction Stop } catch {}
}

$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -MultipleInstances IgnoreNew `
    -ExecutionTimeLimit (New-TimeSpan -Hours 2)

# ── light: every 30 minutes, indefinitely ──
$a1 = New-ScheduledTaskAction -Execute $py `
        -Argument "`"$script`" --mode light" -WorkingDirectory $workdir
$t1 = New-ScheduledTaskTrigger -Once -At (Get-Date).Date.AddMinutes(5) `
        -RepetitionInterval (New-TimeSpan -Minutes 30)
Register-ScheduledTask -TaskName $light -Action $a1 -Trigger $t1 `
    -Settings $settings `
    -Description "Settle finished games, re-price the Sports board, publish if changed." | Out-Null
Write-Output "registered $light  (every 30 min)"

# ── full: once daily, before the US slate ──
$a2 = New-ScheduledTaskAction -Execute $py `
        -Argument "`"$script`" --mode full" -WorkingDirectory $workdir
$t2 = New-ScheduledTaskTrigger -Daily -At 7:30am
Register-ScheduledTask -TaskName $full -Action $a2 -Trigger $t2 `
    -Settings $settings `
    -Description "Refresh all sources, re-predict every sport, then publish." | Out-Null
Write-Output "registered $full   (daily 07:30)"

Write-Output ""
Write-Output "check status : Get-ScheduledTask TaraSports* | Select TaskName,State"
Write-Output "run now      : Start-ScheduledTask -TaskName $light"
Write-Output "see the log  : Get-Content '$workdir\reports\auto_update.log' -Tail 30"
Write-Output "remove       : .\setup_schedule.ps1 -Remove"
