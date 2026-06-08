#!/bin/bash
set -a
source /projects/workershield-v1/.env
set +a
export PYTHONPATH=/projects/workershield-v1
echo ""
echo "========================================="
echo "  WorkerShield — starting on port 7860"
echo "  Open: http://192.168.100.10:7860"
echo "========================================="
echo ""
cd /projects/workershield-v1
python3 ui/app.py
