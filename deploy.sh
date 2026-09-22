#!/bin/bash
set -e
cd /opt/prometheus
git pull
source venv/bin/activate
pip install -q -r requirements.txt
systemctl restart prometheus-api.service prometheus-bots.service
echo "Deployed $(git log --oneline -1)"
