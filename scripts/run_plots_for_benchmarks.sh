#!/bin/bash
# Run plot_multimodel_summary.py for all latest benchmark output directories
#
# Usage: ./scripts/run_plots_for_benchmarks.sh [options]
# Or: bash scripts/run_plots_for_benchmarks.sh [options]
#
# Options:
#   --dry-run         Show which directories would be processed without running plots
#   --filter PATTERN  Only process directories matching PATTERN (e.g., "frangieh21")
#   --force           Run plot_multimodel_summary even if additional_results/ already exists

set -e  # Exit on error

# Parse arguments
DRY_RUN=false
FILTER_PATTERN=""
FORCE_RERUN=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --filter)
            FILTER_PATTERN="$2"
            shift 2
            ;;
        --force)
            FORCE_RERUN=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --dry-run         Show which directories would be processed without running plots"
            echo "  --filter PATTERN  Only process directories matching PATTERN (e.g., 'frangieh21')"
            echo "  --force           Regenerate plots even when additional_results/ already exists"
            echo "  --help, -h        Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
OUTPUTS_DIR="$PROJECT_ROOT/outputs"
PLOT_SCRIPT="$SCRIPT_DIR/plot_multimodel_summary.py"

echo "=============================================="
if [ "$DRY_RUN" = true ]; then
    echo "DRY RUN - Listing benchmark directories"
else
    echo "Running plots for all benchmark directories"
fi
echo "=============================================="
echo ""

# Check if outputs directory exists
if [ ! -d "$OUTPUTS_DIR" ]; then
    echo "ERROR: outputs directory not found at $OUTPUTS_DIR"
    exit 1
fi

# Find all directories with "benchmark" in the name (including those starting with _)
if [ -n "$FILTER_PATTERN" ]; then
    benchmark_dirs=$(find "$OUTPUTS_DIR" -maxdepth 1 -type d -name "*benchmark*" | grep "$FILTER_PATTERN" | sort)
    echo "Filter pattern: $FILTER_PATTERN"
else
    benchmark_dirs=$(find "$OUTPUTS_DIR" -maxdepth 1 -type d -name "*benchmark*" | sort)
fi

if [ -z "$benchmark_dirs" ]; then
    echo "No benchmark directories found in $OUTPUTS_DIR"
    exit 0
fi

echo "Found benchmark directories:"
echo "$benchmark_dirs"
echo ""

# Process each benchmark directory
for benchmark_dir in $benchmark_dirs; do
    benchmark_name=$(basename "$benchmark_dir")
    echo "----------------------------------------------"
    echo "Processing: $benchmark_name"
    
    # Find all timestamp subdirectories (format: YYYY-MM-DD_HH-MM-SS)
    timestamp_dirs=$(find "$benchmark_dir" -maxdepth 1 -type d -regextype sed -regex ".*/[0-9]\{4\}-[0-9]\{2\}-[0-9]\{2\}_[0-9]\{2\}-[0-9]\{2\}-[0-9]\{2\}" | sort)
    
    if [ -z "$timestamp_dirs" ]; then
        echo "  No timestamp directories found, skipping..."
        continue
    fi
    
    # Get the latest (last in sorted list)
    latest_dir=$(echo "$timestamp_dirs" | tail -1)
    timestamp=$(basename "$latest_dir")
    
    echo "  Latest timestamp: $timestamp"
    
    # Check for detailed_metrics.csv
    csv_file="$latest_dir/detailed_metrics.csv"
    if [ ! -f "$csv_file" ]; then
        echo "  WARNING: detailed_metrics.csv not found at $csv_file"
        continue
    fi
    
    echo "  Found: detailed_metrics.csv"

    # Skip if plots already generated (unless --force)
    additional_results_dir="$latest_dir/additional_results"
    if [ -d "$additional_results_dir" ] && [ "$FORCE_RERUN" != true ]; then
        echo "  SKIPPED: additional_results already exists (use --force to regenerate)"
        continue
    fi
    if [ -d "$additional_results_dir" ] && [ "$FORCE_RERUN" = true ]; then
        echo "  --force: rerunning despite existing additional_results/"
    fi
    
    # Run the plotting script (or skip if dry-run)
    if [ "$DRY_RUN" = true ]; then
        echo "  [DRY RUN] Would run: uv run python $PLOT_SCRIPT $csv_file"
    else
        echo "  Running plot_multimodel_summary.py..."
        cd "$PROJECT_ROOT"
        if uv run python "$PLOT_SCRIPT" "$csv_file"; then
            echo "  ✓ Plots generated successfully"
        else
            echo "  ✗ Error generating plots"
        fi
    fi
    echo ""
done

echo "=============================================="
echo "All benchmark directories processed"
echo "=============================================="
