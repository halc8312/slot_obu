# GitHub Release Update Instructions

## Background
The dynamic machine update system has been successfully implemented and pushed to GitHub. The machine_master.csv file now includes:
- 2025-compliant machine list (5号機 removed, 6号機/スマスロ updated)
- `is_active` flag for tracking operational machines
- `last_updated` timestamp for version control

## Update Steps

### Method 1: Using GitHub Web Interface (Recommended)
1. Go to: https://github.com/halc8312/slot_obu/releases/tag/quantile-model-v1
2. Click "Edit" (pencil icon) on the release
3. Under "Assets", find the existing `machine_master.csv`
4. Delete the old file by clicking the trash icon
5. Upload the new file from: `temp_slot_obu/model_files_for_upload/machine_master.csv`
6. Click "Update release"

### Method 2: Using GitHub CLI (if available)
```bash
cd temp_slot_obu
gh release upload quantile-model-v1 model_files_for_upload/machine_master.csv --clobber
```

### Method 3: Using curl API
```bash
# First, get your GitHub token from https://github.com/settings/tokens
# Then run:
cd temp_slot_obu
curl -X POST \
  -H "Authorization: token YOUR_GITHUB_TOKEN" \
  -H "Content-Type: text/csv" \
  --data-binary @model_files_for_upload/machine_master.csv \
  "https://uploads.github.com/repos/halc8312/slot_obu/releases/assets/YOUR_ASSET_ID"
```

## What Changed
- Machine 419: Changed from "凱旋" (5号機) to "サンダーVリベンジ" (6号機)
- Added `is_active` column: Tracks which machines are currently operational
- Added `last_updated` column: Shows when each machine was last verified
- All machines updated to 2025-07-18 status

## Verification
After updating, the GitHub Actions workflow should automatically use the new machine_master.csv for predictions, correctly showing actual machine names instead of generic ones.

## Important Notes
- The file in `model_files_for_upload/machine_master.csv` is the correct one to upload
- Ensure the encoding is UTF-8 with BOM (already set correctly)
- The file contains all 640 machines with their current 2025 status