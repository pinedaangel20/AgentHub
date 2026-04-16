# utils/output_formatter.py
import os
import zipfile

def generate_submission_file(fraud_ids: list, filename="output.txt"):
    """Writes the fraudulent Transaction IDs to a text file, one per line."""
    with open(filename, 'w', encoding='ascii') as f:
        for t_id in fraud_ids:
            f.write(f"{t_id}\n")
    print(f"CHECK: Output file '{filename}' generated with {len(fraud_ids)} flagged transactions.")

def zip_project_for_submission(output_zip_name="submission.zip"):
    """Zips the project directory, excluding virtual environments and secrets."""
    ignored_dirs = {'venv', '.env', '__pycache__', 'data', '.git'}
    
    with zipfile.ZipFile(output_zip_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk("."):
            # Modify dirs in-place to skip ignored directories
            dirs[:] = [d for d in dirs if d not in ignored_dirs]
            
            for file in files:
                # Don't zip the zip file itself or the .env file
                if file == output_zip_name or file == '.env':
                    continue
                
                file_path = os.path.join(root, file)
                # Add file to zip maintaining folder structure
                zipf.write(file_path, arcname=file_path)
                
    print(f"CHECK: Project successfully zipped into '{output_zip_name}' ready for upload.")