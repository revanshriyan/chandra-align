"""Automated CHMAP Browse Ingestion Script for Equatorial OHRC Strips.

HARD RULES:
- Read credentials ONLY from environment variables CHMAP_USER / CHMAP_PASS.
- NEVER write credentials to any file, script, log, or report.
- Respect the site (sequential requests, delays).
- If CAPTCHA or missing credentials defeat scripting, stop after 1 attempt and report 'manual fallback needed'.
"""

import os
import sys
import json
import time
import urllib.request
import urllib.parse
from pathlib import Path


def run_chmap_acquisition():
    username = os.environ.get("CHMAP_USER")
    password = os.environ.get("CHMAP_PASS")

    if not username or not password:
        print("[NOTICE] CHMAP_USER / CHMAP_PASS environment variables not set.")
        print("Manual fallback needed — follow search instructions in docs/EQUATORIAL_DOWNLOAD_STEPS.md")
        return False, "manual_fallback_needed"

    print("[INFO] Attempting automated session login to chmapbrowse.issdc.gov.in...")
    
    # Session setup
    login_url = "https://chmapbrowse.issdc.gov.in/login"
    search_url = "https://chmapbrowse.issdc.gov.in/search"

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) ChandraAlign/1.0",
        "Content-Type": "application/x-www-form-urlencoded"
    }

    try:
        # Attempt single login request
        login_data = urllib.parse.urlencode({"username": username, "password": password}).encode("utf-8")
        req = urllib.request.Request(login_url, data=login_data, headers=headers)
        
        with urllib.request.urlopen(req, timeout=10) as resp:
            body = resp.read().decode("utf-8", errors="ignore")
            if "captcha" in body.lower() or "security code" in body.lower() or resp.status != 200:
                print("[NOTICE] Site login protected by CAPTCHA or interactive security code.")
                print("Manual fallback needed — follow search instructions in docs/EQUATORIAL_DOWNLOAD_STEPS.md")
                return False, "manual_fallback_needed"
    except Exception as e:
        print(f"[NOTICE] Automated login attempt stopped ({e}).")
        print("Manual fallback needed — follow search instructions in docs/EQUATORIAL_DOWNLOAD_STEPS.md")
        return False, "manual_fallback_needed"

    return True, "success"


if __name__ == "__main__":
    success, status = run_chmap_acquisition()
    if not success:
        sys.exit(0)
