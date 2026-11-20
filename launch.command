#!/bin/zsh
cd "$(dirname "$0")"
if [[ ! -x .venv/bin/python ]]; then
  echo "SignSpeak is not set up yet. Follow the 'Mac setup' steps in README.md first."
  read -k 1 "?Press any key to close..."
  exit 1
fi
source .venv/bin/activate
python -m streamlit run app.py
