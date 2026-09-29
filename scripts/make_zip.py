import zipfile
import os

project_dir = r"c:\Users\akber ali\aiml"
zip_path = os.path.join(project_dir, "aiml_2.zip")

# Exclude these
exclude_names = {'__pycache__', 'aiml_project.zip', 'aiml_2.zip', '.gemini'}

def should_exclude(path):
    parts = path.split(os.sep)
    for part in parts:
        if part in exclude_names:
            return True
    return False

print("Creating zip...")
count = 0
with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED, allowZip64=True) as zf:
    for root, dirs, files in os.walk(project_dir):
        # Skip excluded directories
        dirs[:] = [d for d in dirs if d not in exclude_names]
        
        for file in files:
            if file in exclude_names:
                continue
            full_path = os.path.join(root, file)
            arcname = os.path.relpath(full_path, project_dir)
            zf.write(full_path, arcname)
            count += 1
            if count % 100 == 0:
                print(f"  {count} files added...")

print(f"\nDone! Total {count} files added.")
size_mb = os.path.getsize(zip_path) / (1024*1024)
print(f"Zip file: {zip_path}")
print(f"Size: {size_mb:.1f} MB")
