from datasets import load_dataset, load_from_disk
import os

def print_tree(root: str, prefix: str = "") -> None:
    entries = sorted(os.listdir(root))
    for i, name in enumerate(entries):
        path = os.path.join(root, name)
        is_last = (i == len(entries) - 1)
        branch = "└── " if is_last else "├── "
        print(prefix + branch + name)
        if os.path.isdir(path):
            extension = "    " if is_last else "│   "
            print_tree(path, prefix + extension)
            

print("Import Complete.")





# Save to disk in a folder “/home/youruser/hub_data/librispeech”
root_dir = "hub_data/librispeech"
clean_dir = os.path.join(root_dir, "clean")
other_dir = os.path.join(root_dir, "other")

os.makedirs(clean_dir, exist_ok=True)
os.makedirs(other_dir, exist_ok=True)

print("Setup Complete.")





print("Downloading LibriSpeech ASR 'clean' configuration (train.100, train.360, validation, test)...")
ds_clean = load_dataset("openslr/librispeech_asr", "clean")

print("Downloading LibriSpeech ASR 'other' configuration (train.500, validation, test)...")
ds_other = load_dataset("openslr/librispeech_asr", "other")

print(f"Saving 'clean' to: {clean_dir}")
ds_clean.save_to_disk(clean_dir)

print(f"Saving 'other' to: {other_dir}")
ds_other.save_to_disk(other_dir)

print("\nSaved datasets. Split keys:")
print("clean splits:", list(ds_clean.keys()))
print("other splits:", list(ds_other.keys()))

print("\nExample split lengths (number of rows):")
for k in ds_clean.keys():
    print(f"  clean/{k}: {len(ds_clean[k])}")
for k in ds_other.keys():
    print(f"  other/{k}: {len(ds_other[k])}")

print("\nReloading from disk to verify...")
ds_clean_re = load_from_disk(clean_dir)
ds_other_re = load_from_disk(other_dir)

print("Reload OK. Reloaded split keys:")
print("clean:", list(ds_clean_re.keys()))
print("other:", list(ds_other_re.keys()))

print("\nFolder structure under ./hub_data:")
root_to_print = "./hub_data"
print(root_to_print)
print_tree(root_to_print)














