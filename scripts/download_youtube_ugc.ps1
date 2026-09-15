param(
    [string]$TargetRoot = (Join-Path $PSScriptRoot "..\data\YouTube-UGC"),
    [ValidateSet("original_videos_h264", "original_videos", "vp9_compressed_videos", "previews")]
    [string]$Variant = "original_videos_h264",
    [switch]$ManifestOnly,
    [string]$ManifestPath = (Join-Path $PSScriptRoot "..\test_youtube_ugc.txt")
)

$ErrorActionPreference = "Stop"

$Bucket = "gs://ugc-dataset"
$DestName = if ($Variant -eq "original_videos_h264") { "videos" } else { $Variant }
$Dest = Join-Path $TargetRoot $DestName

New-Item -ItemType Directory -Force -Path $Dest | Out-Null

$gcloud = Get-Command gcloud -ErrorAction SilentlyContinue
$gsutil = Get-Command gsutil -ErrorAction SilentlyContinue

if (-not $gcloud -and -not $gsutil) {
    throw "Neither gcloud nor gsutil was found. Install Google Cloud CLI, reopen PowerShell, then rerun this script."
}

if ($ManifestOnly) {
    if (-not (Test-Path -LiteralPath $ManifestPath)) {
        throw "Manifest file not found: $ManifestPath"
    }

    $files = Get-Content -LiteralPath $ManifestPath |
        ForEach-Object { ($_ -split ",", 2)[0].Trim() } |
        Where-Object { $_ } |
        Sort-Object -Unique

    Write-Host "Downloading $($files.Count) files from $Bucket/$Variant to $Dest"

    foreach ($name in $files) {
        $src = "$Bucket/$Variant/$name"
        $dst = Join-Path $Dest $name

        if (Test-Path -LiteralPath $dst) {
            Write-Host "skip existing $name"
            continue
        }

        if ($gcloud) {
            & gcloud storage cp $src $Dest
        }
        else {
            & gsutil cp $src $Dest
        }
    }
}
else {
    Write-Host "Syncing $Bucket/$Variant to $Dest"

    if ($gcloud) {
        & gcloud storage rsync -r "$Bucket/$Variant" $Dest
    }
    else {
        & gsutil -m rsync -r "$Bucket/$Variant" $Dest
    }
}

Write-Host "Done. Video directory: $Dest"
