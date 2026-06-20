import os
import re
import urllib.request
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

def get_file_id(url):
    match = re.search(r'/d/([^/]+)', url)
    return match.group(1) if match else None

def download_file(args):
    url, dest_path = args
    if os.path.exists(dest_path):
        return True
    file_id = get_file_id(url)
    if not file_id:
        return False
    download_url = f"https://docs.google.com/uc?export=download&id={file_id}"
    try:
        urllib.request.urlretrieve(download_url, dest_path)
        return True
    except Exception as e:
        print(f"\nError downloading {url}: {e}")
        return False

def main():
    print("Reading leaderboard and logs...")
    lead = pd.read_excel('GDGoC AI Challenge 2026 - Leaderboard.xlsx', sheet_name='Leaderboard')
    logs = pd.read_excel('GDGoC AI Challenge 2026 - Leaderboard.xlsx', sheet_name='Logs')

    # Top teams (ranks 1 to 10)
    top_teams = lead[lead['Rank'] <= 10]
    top_subs = set(top_teams['Submission ID'].unique())
    print("Top Submission IDs:", top_subs)

    # Filter matches containing top teams
    def contains_top_sub(sub_str):
        if not isinstance(sub_str, str):
            return False
        subs = [s.strip() for s in sub_str.split(',')]
        return any(s in top_subs for s in subs)

    logs['has_top_team'] = logs['Submission IDs'].apply(contains_top_sub)
    matching_logs = logs[logs['has_top_team']].copy()
    print(f"Total matches containing top teams: {len(matching_logs)}")

    # We want to download matches. Let's sample or take the most recent ones.
    # Sorting by 'Created At' to get the latest games first (latest submission versions)
    matching_logs['Created At'] = pd.to_datetime(matching_logs['Created At'])
    matching_logs = matching_logs.sort_values(by='Created At', ascending=False)

    os.makedirs('expert_matches', exist_ok=True)
    
    download_tasks = []
    for idx, row in matching_logs.iterrows():
        match_id = row['Match ID']
        url = row['JSON Drive URL']
        dest = f"expert_matches/{match_id}.json"
        download_tasks.append((url, dest))
        if len(download_tasks) >= 300: # Download 300 matches for a rich dataset
            break

    print(f"Downloading {len(download_tasks)} match JSONs to expert_matches/ ...")
    
    success_count = 0
    with ThreadPoolExecutor(max_workers=10) as executor:
        results = list(tqdm(executor.map(download_file, download_tasks), total=len(download_tasks)))
        success_count = sum(1 for r in results if r)
        
    print(f"Successfully downloaded {success_count}/{len(download_tasks)} matches.")

if __name__ == "__main__":
    main()
