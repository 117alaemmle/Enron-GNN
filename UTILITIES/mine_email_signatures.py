import pandas as pd
import email
import re
from collections import defaultdict
from pathlib import Path


SCRIPT_DIR = Path(__file__).parent
file_path = SCRIPT_DIR.parent / 'Kaggle_DataSet' / 'emails.csv'

print("1. Loading the Enron corpus for text mining...")
# Let's scan a good chunk of emails to get a solid tally
#df = pd.read_csv("../Kaggle_DataSet/emails.csv")
df = pd.read_csv(file_path)

# This dictionary will keep a scorecard. 
# Example: {'jeff@enron.com': {'ceo': 5, 'president': 2}}
title_scorecard = defaultdict(lambda: defaultdict(int))

# The job titles we are hunting for
target_titles = [
    "president", "ceo", "chief executive officer", "vice president", "vp",
    "managing director", "director", "manager", "trader", "analyst", 
    "assistant", "admin", "counsel", "attorney"
]

print("2. Scanning outgoing emails for signatures...")
for raw_message in df['message']:
    msg = email.message_from_string(raw_message)
    sender = msg.get('From')
    
    if sender:
        sender = sender.strip().lower()
        # We only want to classify internal Enron employees
        if "@enron.com" not in sender:
            continue 
            
        # Extract the actual text body of the email
        if msg.is_multipart():
            body = "".join([part.get_payload() for part in msg.get_payload() if part.get_content_type() == 'text/plain'])
        else:
            body = msg.get_payload()
            
        if isinstance(body, str):
            # Split the email into individual lines
            lines = body.split('\n')
            
            # Grab the last 15 lines (the "signature block")
            signature_block = " ".join(lines[-15:]).lower()
            
            # Search the signature block for our target titles
            for title in target_titles:
                # We use regex \b to ensure we match whole words 
                # (so "admin" doesn't accidentally match "administration")
                if re.search(r'\b' + title + r'\b', signature_block):
                    title_scorecard[sender][title] += 1

print("3. Tallying the results to find the most likely job titles...\n")
print(f"{'Email Address':<30} | {'Predicted Title from Text':<20} | {'Confidence (Mentions)'}")
print("-" * 75)

# Filter and display the results
discovered_dictionary = {}
for sender, tallies in title_scorecard.items():
    if tallies: # If we found at least one title
        # Find the title with the highest count for this person
        best_title = max(tallies, key=tallies.get)
        mentions = tallies[best_title]
        
        # Only print people where we found the title multiple times (filters out noise/forwards)
        if mentions > 5:
            print(f"{sender:<30} | {best_title:<20} | {mentions}")
            discovered_dictionary[sender] = best_title

print(f"\nSuccessfully auto-mined {len(discovered_dictionary)} highly confident job titles!")