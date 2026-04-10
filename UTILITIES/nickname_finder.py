#Search through our notebook-lm role label names in the corpus to see if any emails are being skipped due to nicknames.
import pandas as pd
import email

print("1. Loading Enron Corpus for Nickname Diagnostic...")
df = pd.read_csv('./Kaggle_DataSet/emails.csv')

edges_text = []
for raw_message in df['message']:
    msg = email.message_from_string(raw_message)
    sender = msg.get('From')
    receivers_raw = msg.get('To')
    
    if sender and receivers_raw:
        sender = sender.strip().lower()
        receivers = [r.strip().lower() for r in receivers_raw.replace('\n', '').split(',')]
        for r in receivers:
            if r: edges_text.append((sender, r))

# Get all unique email addresses and extract just the part before the '@'
unique_emails = set([src for src, dst in edges_text] + [dst for src, dst in edges_text])
enron_prefixes = set([e.split('@')[0] for e in unique_emails])

# Paste your FULL dictionary here (I've included a sample based on your previous list)
role_labels = {
    # --- 0: EXECUTIVES & BOARD MEMBERS (The Hubs) --- #"greg.whalley": 0 is our test node
    "kenneth.lay": 0, "jeff.skilling": 0,  
    "mark.frevert": 0, "cliff.baxter": 0, "mick.seidl": 0, "andrew.fastow": 0, 
    "richard.causey": 0, "richard.buy": 0, "john.lavorato": 0, "louise.kitchen": 0,
    "kenneth.rice": 0, "kevin.hannon": 0, "joe.hirko": 0, "lou.pai": 0, 
    "dave.delainey": 0, "rebecca.mark": 0, "joe.sutton": 0, "rebecca.mcdonald": 0, 
    "john.sherriff": 0, "stanley.horton": 0, 
    "jeffrey.sherrick": 0, "forresthoglund": 0, "william.powers": 0,
    "wgramm": 0, "john.urquhart": 0,

    # --- 1: MANAGERS, DIRECTORS & VPs (The Bridges) --- "timothy.belden": 1,
    "jeffrey.mcmahon": 1, "ben.glisan": 1, "raymond.bowen": 1, "michael.kopper": 1, 
    "bill.gathmann": 1, "lfastow": 1, "bill.brown": 1, "james.timmins": 1, 
    "joe.deffner": 1, "kathy.lynn": 1, "shirley.hudler": 1, "tim.despain": 1, 
    "mike.jakubik": 1, "kelly.boots": 1, "woytek.david": 1, "wes.colwell": 1, 
    "ryan.siurek": 1, "rodney.faldyn": 1, "bob.butts": 1,  
    "mike.mcconnell": 1, "jeffrey.shankman": 1, "john.forney": 1, "don.black": 1, 
    "david.cox": 1, "jim.fallon": 1, "kevin.howard": 1, 
    "michael.krautz": 1, "larry.lawyer": 1, "tom.white": 1, "jeremy.blachman": 1, 
    "margaret.ceconi": 1, "jimmie.williams": 1, "javier.li": 1, "james.c.alexander": 1, "bob.kelly": 1, "michael.brown": 1, "amanda.martin": 1, 
    "frank.stabler": 1, "jere.overdyke": 1, "colin.skellett": 1, "mark.palmer": 1, 
    "mark.koenig": 1, "kean.steve": 1, "cindy.olson": 1, 
    "rebecca.carter": 1, "beth.tilney": 1, "mary.joyce": 1,
    "david.oxley": 1, "robert.jones": 1, "danny.mccarty": 1, "john.esslinger": 1, 
    "john.harding": 1, "vince.kaminski": 1, "sherron.watkins": 1, "greg.piper": 1, 
    "allan.sommer": 1, "mark.muller": 1, "gene.humphrey": 1,
    "alan.quaintance": 1, "clint.walden": 1,

    # --- 2: SPECIALISTS, TRADERS & LEGAL (The Cliques) --- "john.arnold": 2,
    "angela.schwarz": 2, "james.derrick": 2, 
    "mark.haedicke": 2, "jordan.mintz": 2, "kristina.mordaunt": 2, "richard.sanders": 2, 
    "rob.walls": 2, "rex.rogers": 2, "stuart.zisman": 2, "christian.yoder": 2, 
    "sharon.butcher": 2, "carol.st.clair": 2, "scott.sefton": 2,

    # --- 3: GENERAL STAFF, ANALYSTS & ACCOUNTANTS (The Leaves) ---
    "kevin.jordan": 3, "wanda.curry": 3, "kent.castleman": 3, 
    "charles.weiss": 3, "trushar.patel": 3, "stinson.gibner": 3, 
    "vasant.shanbhogue": 3, "rakesh.bharati": 3, "kevin.kindall": 3, "mark.lay": 3, 
    "keith.power": 3, "steve.pearlman": 3, "mary.clark": 3, "joannie.williamson": 3, 
    "ding.yuan": 3, "s..galvan": 3, "ron.baker": 3,
    "george.mckean": 3, "sheila.knudsen": 3, 
    "cheryl.lipshutz": 3,

    # --- 4: ADMINS & SUPPORT (The Broadcast Stars) --- "rosalee.fleming": 4,
    "sherri.sera": 4, "kim.garcia": 4
}

missing_names = []
found_names = 0

for name in role_labels.keys():
    if name in enron_prefixes:
        found_names += 1
    else:
        missing_names.append(name)

print(f"\nDiagnostic Results: Found {found_names} | Missing {len(missing_names)} out of {len(role_labels)} total.")
print("=" * 80)

if missing_names:
    print(f"{'MISSING DICTIONARY NAME': <25} | {'SUGGESTED NICKNAMES FOUND IN CORPUS'}")
    print("-" * 80)
    for missing in missing_names:
        # Extract the last name (e.g., "belden" from "timothy.belden")
        last_name = missing.split('.')[-1]
        
        # Find all prefixes in the corpus that contain this last name
        suggestions = [p for p in enron_prefixes if last_name in p and len(p) < 20]
        
        # Format the suggestions so it doesn't flood the screen
        if not suggestions:
            suggest_str = "No obvious matches found."
        else:
            suggest_str = ", ".join(suggestions[:5])
            if len(suggestions) > 5:
                suggest_str += f" (+{len(suggestions)-5} more)"
                
        print(f"{missing: <25} | {suggest_str}")
else:
    print("Perfect! Every single name in your dictionary exists in the dataset.")