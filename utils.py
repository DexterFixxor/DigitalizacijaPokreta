import os

def ucitaj_fajlove(folder, sort=True, ekstenzije=None):
    all_files = list()
    for f in os.listdir(folder):
        if ekstenzije == None or (os.path.splitext(f)[-1] in ekstenzije):
            all_files.append(os.path.join(folder, f))
    if sort:
        all_files.sort()
    return all_files